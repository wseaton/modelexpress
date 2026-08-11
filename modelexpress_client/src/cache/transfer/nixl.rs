// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL FFI containment for the NVMe transfer.
//!
//! The ONLY module that touches `nixl-sys`. Everything above it (the protocol,
//! the reconcile loop) is plain async Rust. Built only with the `nixl` feature
//! so the default library/CLI/tests need no `libnixl` or RDMA hardware.
//!
//! Operations mirror the validated Python benchmark exactly:
//! - whole regions are registered once; only the per-leg [`XferDescList`] is
//!   chunked (that is where the POSIX backend gets its IO parallelism),
//! - a FILE region carries its fd in `device_id` and offset 0 in the pointer,
//! - metadata moves as opaque blobs (`get_local_md`/`load_remote_md`) carried by
//!   the ModelExpress registry + the in-band first notification, so we never
//!   touch NIXL's etcd path,
//! - notifications are drained as raw bytes (our tags + the puller's md blob and
//!   buffer address are binary), not via the lossy string `take_notifs`.

use std::os::unix::io::RawFd;
use std::time::Duration;

use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, NixlError, NotificationMap,
    OptArgs, RegistrationHandle, XferDescList, XferOp, XferRequest,
};

use super::CHUNK;

/// A whole host-memory region (the staging buffer) described to NIXL as DRAM.
#[derive(Debug)]
struct HostRegion {
    addr: usize,
    len: usize,
}

impl MemoryRegion for HostRegion {
    fn size(&self) -> usize {
        self.len
    }
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }
}

impl NixlDescriptor for HostRegion {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }
    fn device_id(&self) -> u64 {
        0
    }
}

/// A whole file described to NIXL as a FILE region: offset 0, fd in `device_id`.
#[derive(Debug)]
struct FileRegion {
    len: usize,
    fd: RawFd,
}

impl MemoryRegion for FileRegion {
    fn size(&self) -> usize {
        self.len
    }
    unsafe fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }
}

impl NixlDescriptor for FileRegion {
    fn mem_type(&self) -> MemType {
        MemType::File
    }
    fn device_id(&self) -> u64 {
        u64::try_from(self.fd).unwrap_or(0)
    }
}

/// `base + i * CHUNK`, overflow-checked, as the offset of chunk `i`.
fn chunk_offset(base: usize, i: u64) -> Result<usize, NixlError> {
    let delta = i.checked_mul(CHUNK).ok_or(NixlError::InvalidParam)?;
    let off = (base as u64)
        .checked_add(delta)
        .ok_or(NixlError::InvalidParam)?;
    usize::try_from(off).map_err(|_| NixlError::InvalidParam)
}

/// A NIXL agent wired for the GPU-less storage transfer (UCX + POSIX backends,
/// listen thread enabled). DRAM registration handles are retained for the
/// agent's lifetime (the staging buffer is registered once); file registrations
/// are returned to the caller and deregister on drop.
pub struct NixlAgent {
    agent: Agent,
    ucx: Backend,
    posix: Backend,
    name: String,
    dram_regs: Vec<RegistrationHandle>,
    /// Concurrent POSIX transfer requests a posted file write is striped across.
    /// One request behaves like a single synchronous stream (~2 GB/s on the
    /// target RAID); striping lets the array absorb parallel writers.
    write_streams: usize,
}

impl NixlAgent {
    /// Create the agent with a listen thread on `listen_port` and bring up the
    /// UCX (network) and POSIX (storage) backends.
    pub fn new(name: &str, listen_port: u16) -> Result<Self, NixlError> {
        let cfg = AgentConfig {
            enable_prog_thread: true,
            enable_listen_thread: true,
            listen_port: i32::from(listen_port),
            ..Default::default()
        };
        let agent = Agent::new_configured(name, &cfg)?;
        let (_, ucx_params) = agent.get_plugin_params("UCX")?;
        let ucx = agent.create_backend("UCX", &ucx_params)?;
        // The POSIX plugin's default IO queue executes the transfer synchronously
        // inside postXfer, which serializes the puller's write leg with the next
        // shard's receive (measured: the whole disk write lands in `post`, none in
        // `wait`). The uring/aio queues make postXfer actually asynchronous; not
        // every libnixl build carries them, so fall back in order.
        let posix = ["use_uring", "use_aio"]
            .iter()
            .find_map(|flag| {
                let (_, mut params) = agent.get_plugin_params("POSIX").ok()?;
                params.set(flag, "true").ok()?;
                let backend = agent.create_backend("POSIX", &params).ok()?;
                tracing::info!(flag, "POSIX backend using async IO queue");
                Some(backend)
            })
            .map_or_else(
                || {
                    tracing::warn!("POSIX async IO queues unavailable; writes serialize in post");
                    let (_, params) = agent.get_plugin_params("POSIX")?;
                    agent.create_backend("POSIX", &params)
                },
                Ok,
            )?;
        Ok(Self {
            agent,
            ucx,
            posix,
            name: name.to_string(),
            dram_regs: Vec::new(),
            write_streams: 1,
        })
    }

    /// Set how many concurrent transfer requests a posted file write is striped
    /// across (clamped to at least 1).
    pub fn with_write_streams(mut self, streams: usize) -> Self {
        self.write_streams = streams.max(1);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Names of the plugins NIXL discovered; a quick liveness probe.
    pub fn plugin_names(&self) -> Result<Vec<String>, NixlError> {
        let plugins = self.agent.get_available_plugins()?;
        let mut names = Vec::new();
        for p in plugins.iter() {
            names.push(p?.to_string());
        }
        Ok(names)
    }

    /// Register a host buffer with BOTH backends: UCX (so it is an RDMA endpoint
    /// for the network leg) and POSIX (so it is the memory side of the local
    /// file<->DRAM legs). Registering only one fails the other leg's
    /// `createXferReq` with "no backend had the required registrations".
    pub fn register_dram(&mut self, base: usize, len: usize) -> Result<(), NixlError> {
        let region = HostRegion { addr: base, len };
        let mut opt = OptArgs::new()?;
        opt.add_backend(&self.ucx)?;
        opt.add_backend(&self.posix)?;
        let handle = self.agent.register_memory(&region, Some(&opt))?;
        self.dram_regs.push(handle);
        Ok(())
    }

    /// Register an open file (POSIX backend) as a storage endpoint. The
    /// registration deregisters when the returned handle drops.
    pub fn register_file(
        &mut self,
        fd: RawFd,
        len: usize,
    ) -> Result<RegistrationHandle, NixlError> {
        let region = FileRegion { len, fd };
        let mut opt = OptArgs::new()?;
        opt.add_backend(&self.posix)?;
        self.agent.register_memory(&region, Some(&opt))
    }

    /// This agent's metadata blob, published to the registry / sent in-band so a
    /// peer can `load_remote` it.
    pub fn local_md(&self) -> Result<Vec<u8>, NixlError> {
        self.agent.get_local_md()
    }

    /// Load a peer's metadata blob; returns the peer's agent name.
    pub fn load_remote(&self, blob: &[u8]) -> Result<String, NixlError> {
        self.agent.load_remote_md(blob)
    }

    /// Drop a loaded peer's metadata so a same-named reconnect can load fresh.
    pub fn invalidate_remote(&self, peer: &str) -> Result<(), NixlError> {
        self.agent.invalidate_remote_md(peer)
    }

    /// Whether this agent currently holds `peer`'s metadata (rkeys resolvable).
    pub fn has_remote(&self, peer: &str) -> bool {
        self.agent.check_remote_metadata(peer, None)
    }

    pub fn send_notif(&self, peer: &str, msg: &[u8]) -> Result<(), NixlError> {
        self.agent.send_notification(peer, msg, Some(&self.ucx))
    }

    /// Drain pending notifications as raw bytes: `(sender, payload)` pairs.
    pub fn drain_notifs(&self) -> Result<Vec<(String, Vec<u8>)>, NixlError> {
        let mut map = NotificationMap::new()?;
        self.agent.get_notifications(&mut map, None)?;
        // Collect sender names first so the immutable `agents()` borrow is
        // released before the per-sender `get_notifications` borrow below.
        let senders = map
            .agents()
            .map(|a| a.map(|s| s.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        senders
            .into_iter()
            .map(|sender| {
                map.get_notifications(&sender)?
                    .map(|note| note.map(|payload| (sender.clone(), payload)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|nested| nested.into_iter().flatten().collect())
    }

    /// Local storage leg: read exactly `size` bytes of a registered file into
    /// the registered host buffer via POSIX.
    pub fn read_file_to_dram(
        &self,
        dram_base: usize,
        fd: RawFd,
        size: u64,
    ) -> Result<(), NixlError> {
        let dram = Self::chunk_dlist(MemType::Dram, dram_base, size, 0)?;
        let file = Self::chunk_dlist(MemType::File, 0, size, u64::try_from(fd).unwrap_or(0))?;
        self.run_leg(XferOp::Read, &dram, &file, &self.name.clone(), &self.posix)
    }

    /// Network leg: RDMA-write exactly `size` bytes from the registered host
    /// buffer into a peer's registered buffer at `peer_base` over UCX.
    pub fn write_dram_to_peer(
        &self,
        dram_base: usize,
        peer_base: usize,
        peer: &str,
        size: u64,
    ) -> Result<(), NixlError> {
        let local = Self::chunk_dlist(MemType::Dram, dram_base, size, 0)?;
        let remote = Self::chunk_dlist(MemType::Dram, peer_base, size, 0)?;
        self.run_leg(XferOp::Write, &local, &remote, peer, &self.ucx)
    }

    /// Local storage leg, posted (non-blocking): begin writing `size` bytes of
    /// the registered host buffer down to a registered file via POSIX, striped
    /// across `write_streams` concurrent transfer requests (chunk-aligned, so
    /// `O_DIRECT` alignment holds per stripe). Returns the in-flight requests to
    /// await with [`NixlAgent::wait_write`]; the puller posts the next shard's
    /// receive while these drain.
    pub fn post_write_dram_to_file(
        &self,
        dram_base: usize,
        fd: RawFd,
        size: u64,
    ) -> Result<Vec<XferRequest>, NixlError> {
        let dev = u64::try_from(fd).unwrap_or(0);
        let mut opt = OptArgs::new()?;
        opt.add_backend(&self.posix)?;
        let mut reqs = Vec::new();
        for (off, len) in super::stripe_ranges(size, self.write_streams) {
            let stripe_base = chunk_offset(dram_base, off / CHUNK)?;
            let file_base = usize::try_from(off).map_err(|_| NixlError::InvalidParam)?;
            let dram = Self::chunk_dlist(MemType::Dram, stripe_base, len, 0)?;
            let file = Self::chunk_dlist(MemType::File, file_base, len, dev)?;
            let req =
                self.agent
                    .create_xfer_req(XferOp::Write, &dram, &file, &self.name, Some(&opt))?;
            self.agent.post_xfer_req(&req, Some(&opt))?;
            reqs.push(req);
        }
        Ok(reqs)
    }

    /// Block until every stripe posted by [`NixlAgent::post_write_dram_to_file`]
    /// completes.
    pub fn wait_write(&self, reqs: Vec<XferRequest>) -> Result<(), NixlError> {
        for req in reqs {
            loop {
                if self.agent.get_xfer_status(&req)?.is_success() {
                    break;
                }
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        Ok(())
    }

    /// Build a transfer descriptor list covering exactly `size` bytes from
    /// `base`: full `CHUNK` descriptors plus a final partial one. The chunking
    /// is what lets the POSIX backend issue parallel IO; the partial tail keeps
    /// the transfer byte-exact so source files need no padding.
    fn chunk_dlist(
        mem: MemType,
        base: usize,
        size: u64,
        dev_id: u64,
    ) -> Result<XferDescList<'static>, NixlError> {
        let mut dl = XferDescList::new(mem)?;
        let mut i = 0u64;
        let mut moved = 0u64;
        while moved < size {
            let len = CHUNK.min(size.saturating_sub(moved));
            let len = usize::try_from(len).map_err(|_| NixlError::InvalidParam)?;
            dl.add_desc(chunk_offset(base, i)?, len, dev_id);
            moved = moved.checked_add(CHUNK).ok_or(NixlError::InvalidParam)?;
            i = i.checked_add(1).ok_or(NixlError::InvalidParam)?;
        }
        Ok(dl)
    }

    /// Issue one transfer leg and block until it completes.
    fn run_leg(
        &self,
        op: XferOp,
        local: &XferDescList,
        remote: &XferDescList,
        remote_name: &str,
        backend: &Backend,
    ) -> Result<(), NixlError> {
        let mut opt = OptArgs::new()?;
        opt.add_backend(backend)?;
        let req: XferRequest =
            self.agent
                .create_xfer_req(op, local, remote, remote_name, Some(&opt))?;
        self.agent.post_xfer_req(&req, Some(&opt))?;
        loop {
            if self.agent.get_xfer_status(&req)?.is_success() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

/// Map a `NixlError` into `anyhow` so the FFI implements the feature-independent
/// [`super::Transport`] trait the protocol is written against.
fn ax<T>(r: Result<T, NixlError>) -> anyhow::Result<T> {
    r.map_err(|e| anyhow::anyhow!("nixl: {e:?}"))
}

// Each method delegates to the inherent method of the same name (inherent
// methods take priority in resolution), converting the error.
impl super::Transport for NixlAgent {
    type WriteHandle = Vec<XferRequest>;

    fn name(&self) -> &str {
        self.name()
    }

    fn local_md(&self) -> anyhow::Result<Vec<u8>> {
        ax(self.local_md())
    }

    fn load_remote(&mut self, blob: &[u8]) -> anyhow::Result<String> {
        ax(NixlAgent::load_remote(self, blob))
    }

    fn invalidate_remote(&mut self, peer: &str) -> anyhow::Result<()> {
        ax(NixlAgent::invalidate_remote(self, peer))
    }

    fn send_notif(&self, peer: &str, msg: &[u8]) -> anyhow::Result<()> {
        ax(self.send_notif(peer, msg))
    }

    fn drain_notifs(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
        ax(self.drain_notifs())
    }

    type FileReg = RegistrationHandle;

    fn register_dram(&mut self, base: usize, len: usize) -> anyhow::Result<()> {
        ax(self.register_dram(base, len))
    }

    fn register_file(&mut self, fd: RawFd, len: usize) -> anyhow::Result<RegistrationHandle> {
        ax(self.register_file(fd, len))
    }

    fn read_file_to_dram(&self, dram_base: usize, fd: RawFd, size: u64) -> anyhow::Result<()> {
        ax(self.read_file_to_dram(dram_base, fd, size))
    }

    fn write_dram_to_peer(
        &self,
        dram_base: usize,
        peer_base: usize,
        peer: &str,
        size: u64,
    ) -> anyhow::Result<()> {
        ax(self.write_dram_to_peer(dram_base, peer_base, peer, size))
    }

    fn post_write_dram_to_file(
        &self,
        dram_base: usize,
        fd: RawFd,
        size: u64,
    ) -> anyhow::Result<Vec<XferRequest>> {
        ax(self.post_write_dram_to_file(dram_base, fd, size))
    }

    fn wait_write(&self, handle: Vec<XferRequest>) -> anyhow::Result<()> {
        ax(self.wait_write(handle))
    }
}
