// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process [`Transport`] for exercising the transfer protocol without RDMA.
//!
//! It is a loopback, not a fake: the file legs are real `pread`/`pwrite` against
//! the same fds the protocol opens, and the "network" leg is a `memcpy` between
//! the two staging buffers, which live in the same process during a test. So the
//! bytes genuinely move and the puller's SHA check is genuine; only the fabric
//! (notification delivery and metadata) is short-circuited through shared
//! in-memory queues. The NIXL path is validated separately on real hardware via
//! the two-node harness.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::os::raw::c_void;
use std::os::unix::io::RawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::Transport;
use super::stager::ModelLocator;

type Inbox = Arc<Mutex<VecDeque<(String, Vec<u8>)>>>;

/// Shared notification fabric connecting a set of [`Loopback`] agents by name.
#[derive(Clone, Default)]
pub struct Fabric {
    inboxes: Arc<Mutex<HashMap<String, Inbox>>>,
}

impl Fabric {
    fn inbox(&self, name: &str) -> Inbox {
        self.inboxes
            .lock()
            .expect("fabric lock")
            .entry(name.to_string())
            .or_default()
            .clone()
    }

    /// Inject a notification from `from` into `to`'s inbox. Lets a test drive
    /// one side of the protocol in isolation.
    pub fn deliver(&self, from: &str, to: &str, msg: &[u8]) {
        self.inbox(to)
            .lock()
            .expect("inbox lock")
            .push_back((from.to_string(), msg.to_vec()));
    }

    /// Drain `name`'s inbox, so a test can inspect what a side was sent.
    pub fn drain(&self, name: &str) -> Vec<(String, Vec<u8>)> {
        self.inbox(name)
            .lock()
            .expect("inbox lock")
            .drain(..)
            .collect()
    }
}

/// A [`ModelLocator`] backed by a fixed `model -> (dir, revision)` map, for
/// driving [`super::stager::CacheServer`] in tests.
#[derive(Default)]
pub struct MapLocator(pub HashMap<String, (PathBuf, String)>);

impl ModelLocator for MapLocator {
    fn locate(&self, model: &str) -> Option<(PathBuf, String)> {
        self.0.get(model).cloned()
    }
}

/// One agent on a [`Fabric`]. Implements [`Transport`] over real local IO.
pub struct Loopback {
    name: String,
    fabric: Fabric,
    inbox: Inbox,
    /// When set, flip a byte during the network leg so the puller's SHA check
    /// sees corruption (tests data-integrity detection honestly).
    corrupt: bool,
    /// Peers whose metadata is loaded. Mirrors NIXL's semantics: loading an
    /// already-loaded name fails until it is invalidated, so tests exercise the
    /// same reconnect behaviour the hardware shows.
    loaded: HashSet<String>,
}

impl Loopback {
    pub fn new(name: &str, fabric: &Fabric) -> Self {
        Self::build(name, fabric, false)
    }

    /// A `Loopback` whose network leg corrupts one byte, to test that the puller
    /// rejects a corrupted transfer.
    pub fn new_corrupting(name: &str, fabric: &Fabric) -> Self {
        Self::build(name, fabric, true)
    }

    fn build(name: &str, fabric: &Fabric, corrupt: bool) -> Self {
        let inbox = fabric.inbox(name);
        Self {
            name: name.to_string(),
            fabric: fabric.clone(),
            inbox,
            corrupt,
            loaded: HashSet::new(),
        }
    }
}

/// Read exactly `len` bytes from `fd` (starting at offset 0) into `ptr`,
/// handling short reads.
fn pread_exact(fd: RawFd, ptr: *mut u8, len: usize) -> anyhow::Result<()> {
    let mut done = 0usize;
    while done < len {
        let r = unsafe {
            libc::pread(
                fd,
                ptr.add(done).cast::<c_void>(),
                len.saturating_sub(done),
                done as libc::off_t,
            )
        };
        if r < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        if r == 0 {
            break;
        }
        done = done.saturating_add(r as usize);
    }
    Ok(())
}

/// Write exactly `len` bytes from `ptr` to `fd` (starting at offset 0),
/// handling short writes.
fn pwrite_exact(fd: RawFd, ptr: *const u8, len: usize) -> anyhow::Result<()> {
    let mut done = 0usize;
    while done < len {
        let r = unsafe {
            libc::pwrite(
                fd,
                ptr.add(done).cast::<c_void>(),
                len.saturating_sub(done),
                done as libc::off_t,
            )
        };
        if r < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        done = done.saturating_add(r as usize);
    }
    Ok(())
}

fn leg_len(size: u64) -> anyhow::Result<usize> {
    Ok(usize::try_from(size)?)
}

impl Transport for Loopback {
    fn name(&self) -> &str {
        &self.name
    }

    fn local_md(&self) -> anyhow::Result<Vec<u8>> {
        Ok(self.name.as_bytes().to_vec())
    }

    fn load_remote(&mut self, blob: &[u8]) -> anyhow::Result<String> {
        let peer = String::from_utf8(blob.to_vec())?;
        if !self.loaded.insert(peer.clone()) {
            anyhow::bail!("loadRemoteMD: {peer} already loaded (NIXL_ERR_NOT_ALLOWED)");
        }
        Ok(peer)
    }

    fn invalidate_remote(&mut self, peer: &str) -> anyhow::Result<()> {
        self.loaded.remove(peer);
        Ok(())
    }

    fn send_notif(&self, peer: &str, msg: &[u8]) -> anyhow::Result<()> {
        self.fabric.deliver(&self.name, peer, msg);
        Ok(())
    }

    fn drain_notifs(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
        Ok(self.inbox.lock().expect("inbox lock").drain(..).collect())
    }

    fn register_dram(&mut self, _base: usize, _len: usize) -> anyhow::Result<()> {
        Ok(())
    }

    fn register_file(&mut self, _fd: RawFd, _len: usize) -> anyhow::Result<()> {
        Ok(())
    }

    fn read_file_to_dram(&self, dram_base: usize, fd: RawFd, size: u64) -> anyhow::Result<()> {
        pread_exact(fd, dram_base as *mut u8, leg_len(size)?)
    }

    fn write_dram_to_peer(
        &self,
        dram_base: usize,
        peer_base: usize,
        _peer: &str,
        size: u64,
    ) -> anyhow::Result<()> {
        let len = leg_len(size)?;
        // Same process during a test: the peer's staging buffer is a live
        // mmap, so this memcpy is the honest analog of the RDMA write. The
        // PULL/DONE handshake serializes access, so there is no concurrent
        // reader of the destination region.
        unsafe {
            std::ptr::copy_nonoverlapping(dram_base as *const u8, peer_base as *mut u8, len);
            if self.corrupt && len > 0 {
                *(peer_base as *mut u8) ^= 0xff;
            }
        }
        Ok(())
    }

    type WriteHandle = DeferredWrite;

    fn post_write_dram_to_file(
        &self,
        dram_base: usize,
        fd: RawFd,
        size: u64,
    ) -> anyhow::Result<DeferredWrite> {
        // Defer the actual pwrite to `wait_write`, reading the staging buffer at
        // wait time. If the puller wrongly reused this slot before waiting, the
        // bytes would be clobbered and the SHA check would catch it, so this is
        // an honest test of the double-buffer's slot discipline.
        Ok(DeferredWrite {
            dram_base,
            fd,
            size,
        })
    }

    fn wait_write(&self, handle: DeferredWrite) -> anyhow::Result<()> {
        pwrite_exact(
            handle.fd,
            handle.dram_base as *const u8,
            leg_len(handle.size)?,
        )
    }
}

/// A loopback write leg captured at post time and performed at wait time.
pub struct DeferredWrite {
    dram_base: usize,
    fd: RawFd,
    size: u64,
}
