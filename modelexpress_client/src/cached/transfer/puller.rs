// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Puller side of the transfer protocol: pull a whole model snapshot from a
//! stager into the local cache, verify it, and publish it atomically. Generic
//! over [`Transport`] so the protocol is unit-testable against the in-process
//! [`super::loopback`] with no RDMA.
//!
//! Every file in the snapshot lands under a temp name in the revision's snapshot
//! directory, is SHA-verified against the manifest, then renamed into place;
//! once every file is in, the directory gets its
//! [`cache_layout::COMPLETE_SENTINEL`]. The destination directory for the
//! manifest's revision is computed by a caller-supplied resolver, so the puller
//! stays decoupled from the cache layout (the daemon passes `resolve_model_path`,
//! tests pass a tempdir).

use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::bail;

use super::buffer::StagingBuffer;
use super::{Manifest, Transport, cache_layout, gbps, notif};

const POLL: Duration = Duration::from_micros(200);
const NOTIF_TIMEOUT: Duration = Duration::from_secs(300);
const GIB: u64 = 1 << 30;

/// Outcome of a completed pull.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullSummary {
    /// Destination snapshot directory the model landed in.
    pub dest: PathBuf,
    /// Total bytes transferred.
    pub bytes: u64,
    /// Number of files pulled and verified.
    pub files: usize,
}

/// Pulls a whole model from one stager. The staging buffer is allocated and
/// registered in [`Puller::new`], before any metadata is handed out, so the
/// stager can RDMA-write into it.
pub struct Puller<'a, T: Transport> {
    agent: &'a mut T,
    buf: StagingBuffer,
    cap: u64,
    direct: bool,
}

impl<'a, T: Transport> Puller<'a, T> {
    /// Allocate + register a `buf_gib` receive buffer. `direct` selects
    /// `O_DIRECT` writes (false in Phase 3).
    pub fn new(agent: &'a mut T, buf_gib: u32, direct: bool) -> anyhow::Result<Self> {
        let cap = u64::from(buf_gib).saturating_mul(GIB / super::CHUNK).max(1);
        let buf = StagingBuffer::new(cap)?;
        agent.register_dram(buf.base_addr(), buf.len())?;
        Ok(Self {
            agent,
            buf,
            cap,
            direct,
        })
    }

    /// Pull `model` from the holder identified by `holder_md`, landing it under
    /// `dest_for(revision)`. Verifies every file and marks the model complete.
    pub fn pull(
        &mut self,
        holder_md: &[u8],
        model: &str,
        dest_for: impl Fn(&str) -> PathBuf,
    ) -> anyhow::Result<PullSummary> {
        let holder = self.agent.load_remote(holder_md)?;
        let req = notif::encode_manifest_request(model, &self.agent.local_md()?);
        self.agent.send_notif(&holder, &req)?;
        let manifest = self.await_manifest(model)?;

        let dest = dest_for(&manifest.revision);
        std::fs::create_dir_all(&dest)?;
        tracing::info!(model, files = manifest.shards.len(), dest = %dest.display(), "pulling");

        let mut bytes: u64 = 0;
        for (idx, shard) in manifest.shards.iter().enumerate() {
            if shard.n_chunks() > self.cap {
                bail!(
                    "{} needs {} chunks > buffer {}; raise buf_gib",
                    shard.rel_path,
                    shard.n_chunks(),
                    self.cap
                );
            }
            let size = shard.true_size;
            let final_path = dest.join(&shard.rel_path);
            cache_layout::ensure_parent(&final_path)?;
            let tmp = cache_layout::temp_path(&final_path);
            let file = cache_layout::open_direct(&tmp, true, self.direct)?;
            file.set_len(size)?;

            // Ask for the shard (PULL + our buffer base); the holder writes it
            // in. The request->DONE window covers the holder's read + network.
            let mut pull = notif::indexed(notif::PULL, idx);
            pull.extend_from_slice(&(self.buf.base_addr() as u64).to_le_bytes());
            let t = Instant::now();
            self.agent.send_notif(&holder, &pull)?;
            self.await_done(idx)?;

            if size > 0 {
                let fd = file.as_raw_fd();
                self.agent.register_file(fd, usize::try_from(size)?)?;
                self.agent
                    .write_dram_to_file(self.buf.base_addr(), fd, size)?;
            }
            file.sync_all()?;
            drop(file);

            let got = cache_layout::sha256_prefix(&tmp, size)?;
            if got != shard.sha256 {
                bail!("{} sha mismatch: {got} != {}", shard.rel_path, shard.sha256);
            }
            cache_layout::finalize_shard(&tmp, &final_path)?;
            bytes = bytes.saturating_add(size);
            tracing::info!(idx, rel_path = %shard.rel_path, gbps = gbps(size, t.elapsed()), "verified");
        }

        cache_layout::mark_complete(&dest)?;
        self.agent.send_notif(&holder, notif::BYE)?;
        let files = manifest.shards.len();
        tracing::info!(model, bytes, files, "all files verified");
        Ok(PullSummary { dest, bytes, files })
    }

    /// Block until the holder replies with the manifest, mapping `NOPE` to a
    /// clear error (the holder does not hold the model).
    fn await_manifest(&self, model: &str) -> anyhow::Result<Manifest> {
        let start = Instant::now();
        loop {
            for (_, msg) in self.agent.drain_notifs()? {
                if let Some(body) = msg.strip_prefix(notif::MANIFEST) {
                    return Ok(serde_json::from_slice(body)?);
                }
                if msg.starts_with(notif::NOT_FOUND) {
                    bail!("holder does not hold model {model}");
                }
            }
            if start.elapsed() >= NOTIF_TIMEOUT {
                bail!("timed out waiting for manifest of {model}");
            }
            std::thread::sleep(POLL);
        }
    }

    /// Block until the holder reports `DONE` for shard `idx`.
    fn await_done(&self, idx: usize) -> anyhow::Result<()> {
        let want = notif::indexed(notif::DONE, idx);
        let start = Instant::now();
        loop {
            for (_, msg) in self.agent.drain_notifs()? {
                if msg.starts_with(&want) {
                    return Ok(());
                }
            }
            if start.elapsed() >= NOTIF_TIMEOUT {
                bail!("timed out waiting for shard {idx}");
            }
            std::thread::sleep(POLL);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::cached::transfer::cache_layout;
    use crate::cached::transfer::loopback::{Fabric, Loopback, MapLocator};
    use crate::cached::transfer::stager::CacheServer;
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn write_file(path: &Path, bytes: &[u8]) {
        cache_layout::ensure_parent(path).expect("mkdir");
        std::fs::write(path, bytes).expect("write");
    }

    fn locator_for(model: &str, dir: &Path, rev: &str) -> MapLocator {
        let mut m = HashMap::new();
        m.insert(model.to_string(), (dir.to_path_buf(), rev.to_string()));
        MapLocator(m)
    }

    /// Full happy path: a multi-model server streams a model's whole snapshot
    /// (weights + config + a nested file) and the puller lands, verifies (real
    /// SHA over real bytes), and publishes it under the revision's dir.
    #[test]
    fn round_trip_full_snapshot() {
        let src = tempfile::tempdir().expect("src");
        let dst = tempfile::tempdir().expect("dst");
        let files: &[(&str, &[u8])] = &[
            ("model-00001.safetensors", b"the quick brown fox weights"),
            ("config.json", b"{\"hidden\": 4}"),
            ("nested/tokenizer.json", b"tok-bytes"),
        ];
        for (name, bytes) in files {
            write_file(&src.path().join(name), bytes);
        }

        let fabric = Fabric::default();
        let mut holder = Loopback::new("h", &fabric);
        let mut puller_agent = Loopback::new("p", &fabric);
        let mut server =
            CacheServer::new(&mut holder, locator_for("m", src.path(), "rev1"), 0, false)
                .expect("server");
        let mut puller = Puller::new(&mut puller_agent, 0, false).expect("puller");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));
            let dst_root = dst.path().to_path_buf();
            let summary = puller
                .pull(b"h", "m", |rev| dst_root.join(rev))
                .expect("pull");
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");

            assert_eq!(summary.files, 3);
            assert_eq!(summary.dest, dst.path().join("rev1"));
            for (name, bytes) in files {
                let got = std::fs::read(summary.dest.join(name)).expect("read dest");
                assert_eq!(got.as_slice(), *bytes, "{name} round-tripped");
            }
            assert!(cache_layout::is_complete(&summary.dest), "sentinel dropped");
            assert!(!cache_layout::temp_path(&summary.dest.join("config.json")).exists());
        });
    }

    /// The holder doesn't hold the model -> the puller gets NOPE and bails (the
    /// daemon would then fall back to origin).
    #[test]
    fn missing_model_yields_not_found() {
        let dst = tempfile::tempdir().expect("dst");
        let fabric = Fabric::default();
        let mut holder = Loopback::new("h", &fabric);
        let mut puller_agent = Loopback::new("p", &fabric);
        let mut server =
            CacheServer::new(&mut holder, MapLocator::default(), 0, false).expect("server");
        let mut puller = Puller::new(&mut puller_agent, 0, false).expect("puller");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));
            let dst_root = dst.path().to_path_buf();
            let err = puller
                .pull(b"h", "absent", |rev| dst_root.join(rev))
                .expect_err("should bail");
            assert!(err.to_string().contains("does not hold model"));
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");
        });
    }

    /// A corrupted transfer must be caught by the SHA check, leaving the model
    /// unpublished (no sentinel).
    #[test]
    fn corrupted_transfer_is_rejected() {
        let src = tempfile::tempdir().expect("src");
        let dst = tempfile::tempdir().expect("dst");
        write_file(&src.path().join("model.safetensors"), b"real weights here");

        let fabric = Fabric::default();
        let mut holder = Loopback::new_corrupting("h", &fabric);
        let mut puller_agent = Loopback::new("p", &fabric);
        let mut server =
            CacheServer::new(&mut holder, locator_for("m", src.path(), "rev1"), 0, false)
                .expect("server");
        let mut puller = Puller::new(&mut puller_agent, 0, false).expect("puller");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));
            let dst_root = dst.path().to_path_buf();
            let err = puller
                .pull(b"h", "m", |rev| dst_root.join(rev))
                .expect_err("should reject corruption");
            assert!(err.to_string().contains("sha mismatch"));
            assert!(!cache_layout::is_complete(&dst_root.join("rev1")));
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");
        });
    }
}
