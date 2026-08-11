// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stager side of the transfer protocol: a long-lived per-node server that
//! streams any locally-held model to pullers on request, driving any
//! [`Transport`]. One node holds many models, so each `MANIFEST_REQUEST` names
//! the model and the server resolves it through a [`ModelLocator`].
//!
//! Control flow over the notif channel (one puller session at a time per
//! puller, keyed by the puller's agent name):
//! 1. puller sends `M?` + model name + its metadata blob; we load the blob,
//!    resolve+scan the model, and reply `MANIFEST` + JSON (or `NOPE` if we don't
//!    hold it),
//! 2. per shard the puller sends `PULL` + index + its buffer base; we read the
//!    file into the staging buffer, RDMA-write it into the puller's buffer, and
//!    reply `DONE` + index,
//! 3. puller sends `BYE` and we drop its session.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};

use super::buffer::StagingBuffer;
use super::{Manifest, Transport, cache_layout, gbps, notif};

const POLL: Duration = Duration::from_micros(200);
const GIB: u64 = 1 << 30;

/// Resolves a model name to the on-disk snapshot directory and revision the
/// node currently holds, or `None` if it does not hold the model.
pub trait ModelLocator {
    fn locate(&self, model: &str) -> Option<(PathBuf, String)>;
}

/// `PULL<idx><base_addr_le>` -> `(idx, base_addr)`.
fn parse_pull(msg: &[u8]) -> Option<(usize, u64)> {
    if !msg.starts_with(notif::PULL) {
        return None;
    }
    let idx: usize = std::str::from_utf8(msg.get(4..8)?).ok()?.parse().ok()?;
    let base: [u8; 8] = msg.get(8..16)?.try_into().ok()?;
    Some((idx, u64::from_le_bytes(base)))
}

struct Session {
    model: String,
    // Opened files back the NIXL registrations; kept alive for the session and
    // dropped on BYE.
    open_files: Vec<File>,
}

/// Serves any held model to pullers over one [`Transport`]. The staging buffer
/// is allocated and registered in [`CacheServer::new`], so [`CacheServer::local_md`]
/// returns a post-registration blob ready to advertise.
pub struct CacheServer<'a, T: Transport, L: ModelLocator> {
    agent: &'a mut T,
    buf: StagingBuffer,
    locator: L,
    direct: bool,
    /// Manifests per model, cached by the snapshot's newest mtime. Scanning
    /// re-hashes the whole model so it is cached, but the cache is invalidated
    /// when the snapshot changes on disk, so a model that grew after a first
    /// partial scan (e.g. a download that finished later) is re-scanned rather
    /// than served stale.
    held: HashMap<String, (Option<std::time::SystemTime>, PathBuf, Manifest)>,
    /// Live sessions keyed by puller agent name.
    sessions: HashMap<String, Session>,
    /// Puller agent names whose metadata is loaded. A puller that crashed
    /// without `BYE` and reconnected under the same name must be invalidated
    /// before its fresh metadata can load (NIXL rejects a duplicate load).
    loaded: std::collections::HashSet<String>,
}

impl<'a, T: Transport, L: ModelLocator> CacheServer<'a, T, L> {
    /// Allocate and register a `buf_gib` staging buffer. A shard larger than the
    /// buffer is rejected at serve time (raise `buf_gib`).
    pub fn new(agent: &'a mut T, locator: L, buf_gib: u32, direct: bool) -> anyhow::Result<Self> {
        let cap = u64::from(buf_gib).saturating_mul(GIB / super::CHUNK).max(1);
        let buf = StagingBuffer::new(cap)?;
        agent.register_dram(buf.base_addr(), buf.len())?;
        Ok(Self {
            agent,
            buf,
            locator,
            direct,
            held: HashMap::new(),
            sessions: HashMap::new(),
            loaded: std::collections::HashSet::new(),
        })
    }

    /// This server's metadata blob, captured after the staging buffer is
    /// registered so a puller can resolve its RDMA keys.
    pub fn local_md(&self) -> anyhow::Result<Vec<u8>> {
        self.agent.local_md()
    }

    /// Serve requests until `stop` is set. Blocks; intended to run on its own
    /// thread (the daemon flips `stop` on shutdown).
    pub fn serve(&mut self, stop: &AtomicBool) -> anyhow::Result<()> {
        tracing::info!("cache server ready");
        while !stop.load(Ordering::Relaxed) {
            for (sender, msg) in self.agent.drain_notifs()? {
                // One puller's bad request must not take the server down for
                // every other puller; log it and keep serving.
                if let Err(e) = self.handle(&sender, &msg) {
                    tracing::warn!(puller = %sender, error = %e, "request failed");
                }
            }
            std::thread::sleep(POLL);
        }
        Ok(())
    }

    fn handle(&mut self, sender: &str, msg: &[u8]) -> anyhow::Result<()> {
        if let Some((model, md)) = notif::decode_manifest_request(msg) {
            // A manifest request from a name we already know is a puller that
            // crashed without BYE and restarted: drop the stale session (and its
            // open files) and invalidate the dead peer's metadata, or loading
            // the fresh blob below is rejected and the reconnect times out.
            if self.sessions.remove(sender).is_some() {
                tracing::info!(puller = sender, "dropped stale session on reconnect");
            }
            if self.loaded.contains(sender) {
                self.agent.invalidate_remote(sender)?;
                self.loaded.remove(sender);
            }
            // Register the puller's rkeys so a later PULL can RDMA-write to it.
            self.agent.load_remote(md)?;
            self.loaded.insert(sender.to_string());
            match self.manifest_reply(model) {
                Ok(reply) => {
                    self.agent.send_notif(sender, &reply)?;
                    self.sessions.insert(
                        sender.to_string(),
                        Session {
                            model: model.to_string(),
                            open_files: Vec::new(),
                        },
                    );
                    tracing::info!(model, puller = sender, "serving manifest");
                }
                Err(e) => {
                    tracing::warn!(model, puller = sender, error = %e, "cannot serve model");
                    self.agent.send_notif(sender, notif::NOT_FOUND)?;
                }
            }
        } else if msg.starts_with(notif::PULL) {
            self.stage_shard(sender, msg)?;
        } else if msg.starts_with(notif::BYE) {
            self.sessions.remove(sender);
            tracing::info!(puller = sender, "session closed");
        }
        Ok(())
    }

    /// `MANIFEST` + JSON for `model`, scanning and caching the manifest the first
    /// time. Errors if the node does not hold the model.
    fn manifest_reply(&mut self, model: &str) -> anyhow::Result<Vec<u8>> {
        let (dir, revision) = self
            .locator
            .locate(model)
            .with_context(|| format!("model not held: {model}"))?;
        let mtime = cache_layout::newest_mtime(&dir);
        let cached_fresh = matches!(self.held.get(model), Some((m, _, _)) if *m == mtime);
        if !cached_fresh {
            let manifest = cache_layout::scan_manifest(&dir, &revision)?;
            self.held.insert(model.to_string(), (mtime, dir, manifest));
        }
        let (_, _, manifest) = self.held.get(model).context("manifest vanished")?;
        let mut reply = notif::MANIFEST.to_vec();
        reply.extend_from_slice(&serde_json::to_vec(manifest)?);
        Ok(reply)
    }

    /// Read one shard off disk and RDMA-write it into the puller's buffer.
    fn stage_shard(&mut self, sender: &str, msg: &[u8]) -> anyhow::Result<()> {
        let (idx, peer_base) = parse_pull(msg).context("bad PULL notif")?;
        let model = self
            .sessions
            .get(sender)
            .context("PULL before manifest request")?
            .model
            .clone();
        // Clone the small bits so no map borrow is held across the transfer.
        let (dir, shard) = {
            let (_, dir, manifest) = self.held.get(&model).context("held model vanished")?;
            let shard = manifest
                .shards
                .get(idx)
                .context("PULL index out of range")?
                .clone();
            (dir.clone(), shard)
        };
        let size = shard.true_size;
        if size > self.buf.len() as u64 {
            bail!(
                "shard {} is {size} B > serve buffer {} B; raise buf_gib",
                shard.rel_path,
                self.buf.len()
            );
        }

        let path = dir.join(&shard.rel_path);
        let file = cache_layout::open_direct(&path, false, self.direct)?;
        let t = Instant::now();
        if size > 0 {
            let fd = file.as_raw_fd();
            self.agent.register_file(fd, usize::try_from(size)?)?;
            self.agent
                .read_file_to_dram(self.buf.base_addr(), fd, size)?;
            self.agent.write_dram_to_peer(
                self.buf.base_addr(),
                usize::try_from(peer_base)?,
                sender,
                size,
            )?;
        }
        self.agent
            .send_notif(sender, &notif::indexed(notif::DONE, idx))?;
        if let Some(session) = self.sessions.get_mut(sender) {
            session.open_files.push(file);
        }
        tracing::info!(
            model = %model,
            idx,
            rel_path = %shard.rel_path,
            gbps = gbps(size, t.elapsed()),
            "shard staged"
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cache::transfer::loopback::{Fabric, Loopback, MapLocator};

    #[test]
    fn pull_before_manifest_request_errors() {
        let fabric = Fabric::default();
        let mut agent = Loopback::new("h", &fabric);
        let mut server =
            CacheServer::new(&mut agent, MapLocator(HashMap::new()), 0, false).expect("server");
        let mut pull = notif::indexed(notif::PULL, 0);
        pull.extend_from_slice(&0u64.to_le_bytes());
        let err = server.handle("p", &pull).expect_err("should error");
        assert!(err.to_string().contains("PULL before manifest request"));
    }

    #[test]
    fn unheld_model_replies_not_found() {
        let fabric = Fabric::default();
        let mut agent = Loopback::new("h", &fabric);
        let mut server =
            CacheServer::new(&mut agent, MapLocator(HashMap::new()), 0, false).expect("server");
        let req = notif::encode_manifest_request("absent/model", b"p");
        server.handle("p", &req).expect("handle");
        // The puller is told NOPE, and no session is created.
        let notes = fabric.drain("p");
        assert_eq!(notes.len(), 1);
        assert!(notes[0].1.starts_with(notif::NOT_FOUND));
        assert!(server.sessions.is_empty());
    }

    /// A puller that crashed without BYE and reconnects under the same agent
    /// name must be served, not rejected. The loopback rejects a duplicate
    /// metadata load exactly like NIXL does, so this passing proves the stager
    /// invalidates the dead peer and drops its stale session first.
    #[test]
    fn reconnect_after_crash_replaces_session() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.safetensors"), b"weights").expect("write");
        let fabric = Fabric::default();
        let mut agent = Loopback::new("h", &fabric);
        let mut map = HashMap::new();
        map.insert(
            "m".to_string(),
            (dir.path().to_path_buf(), "rev".to_string()),
        );
        let mut server = CacheServer::new(&mut agent, MapLocator(map), 0, false).expect("server");

        // First connect: session established, metadata loaded.
        server
            .handle("p", &notif::encode_manifest_request("m", b"p"))
            .expect("first manifest");
        assert!(server.sessions.contains_key("p"));

        // The puller dies mid-session (no BYE) and a restarted pod reconnects
        // under the same name with fresh metadata.
        server
            .handle("p", &notif::encode_manifest_request("m", b"p"))
            .expect("reconnect must be served, not rejected");
        assert_eq!(
            server.sessions.len(),
            1,
            "stale session replaced, not leaked"
        );

        // The replacement session is live: a PULL is served.
        let mut pull = notif::indexed(notif::PULL, 0);
        let buf = vec![0u8; 7];
        pull.extend_from_slice(&(buf.as_ptr() as u64).to_le_bytes());
        server.handle("p", &pull).expect("pull on new session");
        assert_eq!(buf.as_slice(), b"weights");
    }

    #[test]
    fn index_out_of_range_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.safetensors"), b"x").expect("write");
        let fabric = Fabric::default();
        let mut agent = Loopback::new("h", &fabric);
        let mut map = HashMap::new();
        map.insert(
            "m".to_string(),
            (dir.path().to_path_buf(), "rev".to_string()),
        );
        let mut server = CacheServer::new(&mut agent, MapLocator(map), 0, false).expect("server");
        server
            .handle("p", &notif::encode_manifest_request("m", b"p"))
            .expect("manifest");
        let mut pull = notif::indexed(notif::PULL, 9);
        pull.extend_from_slice(&0u64.to_le_bytes());
        let err = server.handle("p", &pull).expect_err("should error");
        assert!(err.to_string().contains("PULL index out of range"));
    }
}
