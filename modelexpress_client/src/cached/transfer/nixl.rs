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
/// listen thread enabled). Registration handles are retained so the registered
/// regions stay live for the agent's lifetime.
pub struct NixlAgent {
    agent: Agent,
    ucx: Backend,
    posix: Backend,
    name: String,
    regs: Vec<RegistrationHandle>,
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
        let (_, posix_params) = agent.get_plugin_params("POSIX")?;
        let posix = agent.create_backend("POSIX", &posix_params)?;
        Ok(Self {
            agent,
            ucx,
            posix,
            name: name.to_string(),
            regs: Vec::new(),
        })
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
        self.regs.push(handle);
        Ok(())
    }

    /// Register an open file (POSIX backend) as a storage endpoint.
    pub fn register_file(&mut self, fd: RawFd, len: usize) -> Result<(), NixlError> {
        let region = FileRegion { len, fd };
        let mut opt = OptArgs::new()?;
        opt.add_backend(&self.posix)?;
        let handle = self.agent.register_memory(&region, Some(&opt))?;
        self.regs.push(handle);
        Ok(())
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
        // released before the per-sender `get_notifications` borrow.
        let mut senders = Vec::new();
        for a in map.agents() {
            senders.push(a?.to_string());
        }
        let mut out = Vec::new();
        for sender in senders {
            for note in map.get_notifications(&sender)? {
                out.push((sender.clone(), note?));
            }
        }
        Ok(out)
    }

    /// Local storage leg: read a registered file into the registered host
    /// buffer via POSIX (`O_DIRECT`), `n_chunks * CHUNK` bytes.
    pub fn read_file_to_dram(
        &self,
        dram_base: usize,
        fd: RawFd,
        n_chunks: u64,
    ) -> Result<(), NixlError> {
        let dram = Self::chunk_dlist(MemType::Dram, dram_base, n_chunks, 0)?;
        let file = Self::chunk_dlist(MemType::File, 0, n_chunks, u64::try_from(fd).unwrap_or(0))?;
        self.run_leg(XferOp::Read, &dram, &file, &self.name.clone(), &self.posix)
    }

    /// Network leg: RDMA-write the registered host buffer into a peer's
    /// registered buffer at `peer_base` over UCX.
    pub fn write_dram_to_peer(
        &self,
        dram_base: usize,
        peer_base: usize,
        peer: &str,
        n_chunks: u64,
    ) -> Result<(), NixlError> {
        let local = Self::chunk_dlist(MemType::Dram, dram_base, n_chunks, 0)?;
        let remote = Self::chunk_dlist(MemType::Dram, peer_base, n_chunks, 0)?;
        self.run_leg(XferOp::Write, &local, &remote, peer, &self.ucx)
    }

    /// Local storage leg: write the registered host buffer down to a registered
    /// file via POSIX (`O_DIRECT`).
    pub fn write_dram_to_file(
        &self,
        dram_base: usize,
        fd: RawFd,
        n_chunks: u64,
    ) -> Result<(), NixlError> {
        let dram = Self::chunk_dlist(MemType::Dram, dram_base, n_chunks, 0)?;
        let file = Self::chunk_dlist(MemType::File, 0, n_chunks, u64::try_from(fd).unwrap_or(0))?;
        self.run_leg(XferOp::Write, &dram, &file, &self.name.clone(), &self.posix)
    }

    /// Build a chunked transfer descriptor list over a registered region. The
    /// chunking is what lets the POSIX backend issue parallel IO.
    fn chunk_dlist(
        mem: MemType,
        base: usize,
        n_chunks: u64,
        dev_id: u64,
    ) -> Result<XferDescList<'static>, NixlError> {
        let mut dl = XferDescList::new(mem)?;
        let chunk = usize::try_from(CHUNK).map_err(|_| NixlError::InvalidParam)?;
        let mut i = 0u64;
        while i < n_chunks {
            dl.add_desc(chunk_offset(base, i)?, chunk, dev_id);
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
