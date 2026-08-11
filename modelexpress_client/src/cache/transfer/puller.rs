// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Puller side of the transfer protocol: pull a whole model snapshot from a
//! stager into the local cache, verify it, and publish it atomically. Generic
//! over [`Transport`] so the protocol is unit-testable against the in-process
//! [`super::loopback`] with no RDMA.
//!
//! The receive is double-buffered: the staging buffer is carved into `depth`
//! slots, and the puller keeps the NVMe write of one shard in flight (posted via
//! [`Transport::post_write_dram_to_file`]) while it receives and verifies the
//! next. The write is the bottleneck by far (the holder's read+RDMA is ~7 GB/s);
//! with `O_DIRECT` writes the overlap lets the disk keep moving instead of
//! stalling on the page-cache flush a buffered `sync_all` forces. The measured
//! single-stream write ceiling on the target RAID is ~2 GB/s. Only one shard is
//! ever requested from the holder at a time, so the holder side stays simple.
//!
//! Each shard is hash-verified from the staging buffer the moment it arrives (not
//! by reading the file back off disk, which would compete with the writes), then
//! its write is posted; once the write completes the temp file is renamed into
//! place. When every file is in, the directory gets its
//! [`cache_layout::COMPLETE_SENTINEL`]. The destination directory for the
//! manifest's revision is computed by a caller-supplied resolver, so the puller
//! stays decoupled from the cache layout.

use std::collections::VecDeque;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};

use super::buffer::StagingBuffer;
use super::{CHUNK, Manifest, Shard, Transport, cache_layout, gbps, notif};

const POLL: Duration = Duration::from_micros(200);
const NOTIF_TIMEOUT: Duration = Duration::from_secs(300);
const GIB: u64 = 1 << 30;
/// Logical-block alignment for `O_DIRECT` writes. 4 KiB is a superset of every
/// common block size, so a length rounded up to it is always acceptable.
const O_DIRECT_ALIGN: u64 = 4096;

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

/// A shard whose NVMe write has been posted and is draining while the puller
/// moves on to the next shard. Held until its write completes and the temp file
/// is renamed into place.
struct InFlight<H, R> {
    idx: usize,
    /// Buffer slot this shard occupies; freed for reuse once finalized.
    slot: usize,
    /// Open temp file backing the write; kept alive until the write is awaited.
    file: File,
    /// The posted write, or `None` for a zero-byte file (no write leg).
    handle: Option<H>,
    /// The file's transport registration; deregisters when this shard finalizes
    /// so a long pull does not accumulate registrations.
    reg: Option<R>,
    tmp: PathBuf,
    final_path: PathBuf,
    rel_path: String,
    size: u64,
    started: Instant,
}

/// Pulls a whole model from one stager, double-buffering the receive against the
/// NVMe write. The staging buffer is allocated and registered in [`Puller::new`],
/// before any metadata is handed out, so the stager can RDMA-write into it.
pub struct Puller<'a, T: Transport> {
    agent: &'a mut T,
    buf: StagingBuffer,
    /// Total buffer capacity in chunks; carved into slots per pull, once the
    /// manifest's largest shard is known.
    cap_chunks: u64,
    /// Requested pipeline depth; the effective depth of a pull may be lower so
    /// every slot fits the largest shard.
    requested_depth: usize,
    direct: bool,
}

/// Carve a buffer of `cap` chunks into `(slot_chunks, depth)` such that the
/// largest shard (`max_shard_chunks`) fits one slot, at the deepest pipeline up
/// to `requested_depth` the buffer allows. Errors only when even the whole
/// buffer cannot hold the largest shard.
fn slot_layout(
    cap: u64,
    requested_depth: usize,
    max_shard_chunks: u64,
) -> anyhow::Result<(u64, usize)> {
    let need = max_shard_chunks.max(1);
    if need > cap {
        bail!("largest shard needs {need} chunks > buffer {cap}; raise buf_gib");
    }
    let fit = cap.checked_div(need).unwrap_or(1).max(1);
    let depth = u64::try_from(requested_depth.max(1))
        .unwrap_or(1)
        .min(fit)
        .max(1);
    let slot_chunks = cap.checked_div(depth).unwrap_or(need).max(need);
    let depth = usize::try_from(depth).unwrap_or(1);
    Ok((slot_chunks, depth))
}

impl<'a, T: Transport> Puller<'a, T> {
    /// Allocate + register a receive buffer of `buf_gib` chunks. The buffer is
    /// carved into up to `pool_depth` slots per pull, shrinking the depth when
    /// the manifest's largest shard needs bigger slots. `direct` selects
    /// `O_DIRECT` writes (false in Phase 3/4b).
    pub fn new(
        agent: &'a mut T,
        buf_gib: u32,
        pool_depth: usize,
        direct: bool,
    ) -> anyhow::Result<Self> {
        let cap_chunks = u64::from(buf_gib).saturating_mul(GIB / CHUNK).max(1);
        let buf = StagingBuffer::new(cap_chunks)?;
        agent.register_dram(buf.base_addr(), buf.len())?;
        Ok(Self {
            agent,
            buf,
            cap_chunks,
            requested_depth: pool_depth.max(1),
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
        manifest.ensure_frameable()?;

        let dest = dest_for(&manifest.revision);
        std::fs::create_dir_all(&dest)?;

        // Carve the buffer for this manifest: the largest shard must fit one
        // slot, so the depth shrinks (down to 1) rather than rejecting the pull.
        let max_shard_chunks = manifest
            .shards
            .iter()
            .map(Shard::n_chunks)
            .max()
            .unwrap_or(1);
        let (slot_chunks, depth) =
            slot_layout(self.cap_chunks, self.requested_depth, max_shard_chunks)?;
        let slot_bytes = usize::try_from(
            slot_chunks
                .checked_mul(CHUNK)
                .context("slot byte length overflow")?,
        )
        .context("slot byte length too large for usize")?;
        tracing::info!(model, files = manifest.shards.len(), depth, dest = %dest.display(), "pulling");

        let mut inflight: VecDeque<InFlight<T::WriteHandle, T::FileReg>> = VecDeque::new();
        // Free slots as a stack; starts full.
        let mut free_slots: Vec<usize> = (0..depth).rev().collect();
        let mut bytes: u64 = 0;
        let pull_start = Instant::now();
        // Per-leg timing comes from these spans (RUST_LOG=...puller=debug); the
        // pull span scopes them and reports the total on close.
        let _pull =
            tracing::info_span!("pull", model, files = manifest.shards.len(), depth).entered();

        for (idx, shard) in manifest.shards.iter().enumerate() {
            // Acquire a slot, finalizing the oldest in-flight write if all slots
            // are busy (this is where the pipeline blocks on the write leg).
            let slot = match free_slots.pop() {
                Some(slot) => slot,
                None => {
                    let done = inflight
                        .pop_front()
                        .context("no in-flight shard to drain for a slot")?;
                    let freed = done.slot;
                    self.finalize(done)?;
                    freed
                }
            };

            let size = shard.true_size;
            let slot_off = slot
                .checked_mul(slot_bytes)
                .context("slot offset overflow")?;
            let slot_base = self
                .buf
                .base_addr()
                .checked_add(slot_off)
                .context("slot base overflow")?;

            // Request the shard into this slot; the holder RDMA-writes it in. The
            // request->DONE window covers the holder's read + network leg.
            let mut pull = notif::indexed(notif::PULL, idx);
            pull.extend_from_slice(&(slot_base as u64).to_le_bytes());
            let started = Instant::now();
            self.agent.send_notif(&holder, &pull)?;
            {
                let _recv = tracing::debug_span!("recv", idx).entered();
                self.await_done(idx)?;
            }

            // Verify from the staging buffer (not a disk readback), before the
            // write is posted, so corrupt data is never written.
            if size > 0 {
                let end = slot_off
                    .checked_add(usize::try_from(size)?)
                    .context("slot end overflow")?;
                let received = self
                    .buf
                    .as_slice()
                    .get(slot_off..end)
                    .context("received range outside staging buffer")?;
                let got = {
                    let _hash = tracing::debug_span!("hash", idx).entered();
                    cache_layout::hash_mem(received)
                };
                if got != shard.hash {
                    bail!("{} hash mismatch: {got} != {}", shard.rel_path, shard.hash);
                }
            }

            // O_DIRECT needs block-aligned write lengths, so the final partial
            // chunk is rounded up; the slot already holds those extra bytes, and
            // `finalize` truncates the file back to the exact size. Buffered
            // writes use the exact size unchanged.
            let write_size = if self.direct {
                cache_layout::align_up(size, O_DIRECT_ALIGN)
            } else {
                size
            };
            let final_path = dest.join(&shard.rel_path);
            cache_layout::ensure_parent(&final_path)?;
            let tmp = cache_layout::temp_path(&final_path);
            let file = {
                let _prep = tracing::debug_span!("prep", idx).entered();
                let file = cache_layout::open_direct(&tmp, true, self.direct)?;
                file.set_len(write_size)?;
                file
            };

            let (handle, reg) = if size > 0 {
                let fd = file.as_raw_fd();
                let reg = {
                    let _reg = tracing::debug_span!("reg", idx).entered();
                    self.agent.register_file(fd, usize::try_from(write_size)?)?
                };
                let _post = tracing::debug_span!("post", idx).entered();
                (
                    Some(
                        self.agent
                            .post_write_dram_to_file(slot_base, fd, write_size)?,
                    ),
                    Some(reg),
                )
            } else {
                (None, None)
            };

            inflight.push_back(InFlight {
                idx,
                slot,
                file,
                handle,
                reg,
                tmp,
                final_path,
                rel_path: shard.rel_path.clone(),
                size,
                started,
            });
            bytes = bytes.saturating_add(size);
        }

        // Drain the tail of the pipeline in request order.
        while let Some(done) = inflight.pop_front() {
            self.finalize(done)?;
        }

        cache_layout::mark_complete(&dest)?;
        self.agent.send_notif(&holder, notif::BYE)?;
        let files = manifest.shards.len();
        let elapsed = pull_start.elapsed();
        tracing::info!(
            model,
            bytes,
            files,
            depth,
            secs = elapsed.as_secs_f64(),
            gbps = gbps(bytes, elapsed),
            "all files verified"
        );
        Ok(PullSummary { dest, bytes, files })
    }

    /// Await a posted write, fsync, and atomically rename the temp file into
    /// place. The SHA was already checked from DRAM when the shard arrived. The
    /// file's registration drops here, after the write has completed.
    fn finalize(&mut self, done: InFlight<T::WriteHandle, T::FileReg>) -> anyhow::Result<()> {
        if let Some(handle) = done.handle {
            let _write = tracing::debug_span!("write", idx = done.idx).entered();
            self.agent.wait_write(handle)?;
        }
        drop(done.reg);
        {
            // Drop any O_DIRECT alignment padding so the file ends at its exact
            // size (a no-op for buffered writes, which wrote exactly `size`).
            let _sync = tracing::debug_span!("sync", idx = done.idx).entered();
            done.file.set_len(done.size)?;
            done.file.sync_all()?;
        }
        drop(done.file);
        cache_layout::finalize_shard(&done.tmp, &done.final_path)?;
        tracing::info!(
            idx = done.idx,
            rel_path = %done.rel_path,
            gbps = gbps(done.size, done.started.elapsed()),
            "verified"
        );
        Ok(())
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
    use crate::cache::transfer::cache_layout;
    use crate::cache::transfer::loopback::{Fabric, Loopback, MapLocator};
    use crate::cache::transfer::stager::CacheServer;
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
        let mut puller = Puller::new(&mut puller_agent, 0, 2, false).expect("puller");
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

    /// The regression case from the Qwen-14B bench: 512-chunk buffer, requested
    /// depth 4, largest shard 232 chunks. The old fixed carve made 128-chunk
    /// slots and rejected the pull; the layout must instead shrink the depth.
    #[test]
    fn slot_layout_shrinks_depth_to_fit_largest_shard() {
        let (slot, depth) = slot_layout(512, 4, 232).expect("layout");
        assert_eq!(depth, 2);
        assert_eq!(slot, 256);

        // Plenty of room: the requested depth is honored.
        let (slot, depth) = slot_layout(512, 4, 10).expect("layout");
        assert_eq!(depth, 4);
        assert_eq!(slot, 128);

        // Shard fills the whole buffer: single slot.
        let (slot, depth) = slot_layout(512, 4, 512).expect("layout");
        assert_eq!(depth, 1);
        assert_eq!(slot, 512);

        // All-empty manifest degrades to a 1-chunk need, not a div-by-zero.
        let (_, depth) = slot_layout(512, 2, 0).expect("layout");
        assert_eq!(depth, 2);
    }

    #[test]
    fn slot_layout_rejects_shard_larger_than_buffer() {
        let err = slot_layout(64, 2, 65).expect_err("too big");
        assert!(err.to_string().contains("raise buf_gib"));
    }

    /// A manifest with more shards than the 4-digit notif framing can address
    /// must be rejected up front, not silently corrupt frame parsing at shard
    /// 10000 mid-pull.
    #[test]
    fn oversized_manifest_is_rejected() {
        let shard = |i: usize| super::super::Shard {
            rel_path: format!("f{i}"),
            true_size: 1,
            hash: String::new(),
        };
        let ok = Manifest {
            revision: "r".into(),
            shards: (0..super::super::MAX_SHARDS).map(shard).collect(),
        };
        ok.ensure_frameable().expect("exactly MAX_SHARDS is fine");

        let too_big = Manifest {
            revision: "r".into(),
            shards: (0..=super::super::MAX_SHARDS).map(shard).collect(),
        };
        let err = too_big.ensure_frameable().expect_err("one too many");
        assert!(err.to_string().contains("at most"));
    }

    /// A completed pull must leave zero live file registrations on either side:
    /// a long-lived stager (or a daemon pulling many models) must not
    /// accumulate transport state per shard served. The totals prove the
    /// assertion is not vacuous.
    #[test]
    fn file_registrations_are_reclaimed() {
        let src = tempfile::tempdir().expect("src");
        let dst = tempfile::tempdir().expect("dst");
        for i in 0..5 {
            write_file(
                &src.path().join(format!("model-0000{i}.safetensors")),
                format!("weights-{i}").as_bytes(),
            );
        }

        let fabric = Fabric::default();
        let mut holder = Loopback::new("h", &fabric);
        let mut puller_agent = Loopback::new("p", &fabric);
        let mut server =
            CacheServer::new(&mut holder, locator_for("m", src.path(), "rev1"), 0, false)
                .expect("server");
        let mut puller = Puller::new(&mut puller_agent, 0, 2, false).expect("puller");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));
            let dst_root = dst.path().to_path_buf();
            puller
                .pull(b"h", "m", |rev| dst_root.join(rev))
                .expect("pull");
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");
        });

        assert_eq!(holder.total_file_regs(), 5, "stager registered every shard");
        assert_eq!(
            puller_agent.total_file_regs(),
            5,
            "puller registered every shard"
        );
        assert_eq!(
            holder.active_file_regs(),
            0,
            "stager reclaimed registrations"
        );
        assert_eq!(
            puller_agent.active_file_regs(),
            0,
            "puller reclaimed registrations"
        );
    }

    /// Crash-and-retry: a puller opens a session (manifest request lands, no
    /// BYE ever sent), dies, and a restarted puller under the same agent name
    /// runs a full pull against the same live server. Reproduces the on-hardware
    /// failure where the stager rejected the reconnect's metadata load and the
    /// retry timed out.
    #[test]
    fn round_trip_after_crashed_predecessor() {
        let src = tempfile::tempdir().expect("src");
        let dst = tempfile::tempdir().expect("dst");
        write_file(
            &src.path().join("model-00001.safetensors"),
            b"crash weights",
        );
        write_file(&src.path().join("config.json"), b"{}");

        let fabric = Fabric::default();
        let mut holder = Loopback::new("h", &fabric);
        let mut server =
            CacheServer::new(&mut holder, locator_for("m", src.path(), "rev1"), 0, false)
                .expect("server");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));

            // First incarnation: requests the manifest, then dies without BYE.
            {
                let crashed = Loopback::new("p", &fabric);
                let req = notif::encode_manifest_request("m", &crashed.local_md().expect("md"));
                crashed.send_notif("h", &req).expect("send");
                // Wait until the server has actually opened the session.
                let start = std::time::Instant::now();
                loop {
                    if crashed
                        .drain_notifs()
                        .expect("drain")
                        .iter()
                        .any(|(_, m)| m.starts_with(notif::MANIFEST))
                    {
                        break;
                    }
                    assert!(start.elapsed() < NOTIF_TIMEOUT, "no manifest reply");
                    std::thread::sleep(POLL);
                }
            }

            // Second incarnation, same name: the full pull must succeed.
            let mut retry_agent = Loopback::new("p", &fabric);
            let mut puller = Puller::new(&mut retry_agent, 0, 2, false).expect("puller");
            let dst_root = dst.path().to_path_buf();
            let summary = puller
                .pull(b"h", "m", |rev| dst_root.join(rev))
                .expect("retry pull after crash");
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");

            assert_eq!(summary.files, 2);
            let got = std::fs::read(summary.dest.join("model-00001.safetensors")).expect("read");
            assert_eq!(got.as_slice(), b"crash weights");
        });
    }

    /// Pipelined path: with depth 2 and more files than slots, slots are reused.
    /// The loopback defers each write to wait time, reading the slot bytes then,
    /// so a slot reused before its write was awaited would corrupt the file and
    /// fail the byte check. Passing therefore proves the double-buffer's slot
    /// discipline is correct.
    #[test]
    fn round_trip_pipelined_reuses_slots() {
        let src = tempfile::tempdir().expect("src");
        let dst = tempfile::tempdir().expect("dst");
        // Five files, distinct content, so slot reuse (depth 2) recycles slots
        // several times; one empty file exercises the zero-byte path.
        let files: &[(&str, &[u8])] = &[
            ("a.safetensors", b"alpha weights are distinctive aaaa"),
            ("b.safetensors", b"bravo weights differ bbbbbbbbbbbb"),
            ("c.json", b"{\"c\": 3}"),
            ("d/empty.txt", b""),
            ("e.safetensors", b"echo weights tail eeeeeeeeeeeeeeee"),
        ];
        for (name, bytes) in files {
            write_file(&src.path().join(name), bytes);
        }

        let fabric = Fabric::default();
        let mut holder = Loopback::new("h", &fabric);
        let mut puller_agent = Loopback::new("p", &fabric);
        let mut server =
            CacheServer::new(&mut holder, locator_for("m", src.path(), "rev1"), 1, false)
                .expect("server");
        // buf_gib=1 -> 64 chunks, depth 2 -> two real slots that get recycled.
        let mut puller = Puller::new(&mut puller_agent, 1, 2, false).expect("puller");
        assert_eq!(puller.requested_depth, 2, "two real slots");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));
            let dst_root = dst.path().to_path_buf();
            let summary = puller
                .pull(b"h", "m", |rev| dst_root.join(rev))
                .expect("pull");
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");

            assert_eq!(summary.files, 5);
            for (name, bytes) in files {
                let got = std::fs::read(summary.dest.join(name)).expect("read dest");
                assert_eq!(
                    got.as_slice(),
                    *bytes,
                    "{name} round-tripped through reused slots"
                );
            }
            assert!(cache_layout::is_complete(&summary.dest), "sentinel dropped");
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
        let mut puller = Puller::new(&mut puller_agent, 0, 2, false).expect("puller");
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

    /// A corrupted transfer must be caught by the SHA check (now over the staging
    /// buffer), leaving the model unpublished (no sentinel) and unwritten.
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
        let mut puller = Puller::new(&mut puller_agent, 0, 2, false).expect("puller");
        let stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let served = s.spawn(|| server.serve(&stop));
            let dst_root = dst.path().to_path_buf();
            let err = puller
                .pull(b"h", "m", |rev| dst_root.join(rev))
                .expect_err("should reject corruption");
            assert!(err.to_string().contains("hash mismatch"));
            assert!(!cache_layout::is_complete(&dst_root.join("rev1")));
            stop.store(true, Ordering::Relaxed);
            served.join().expect("join").expect("serve ok");
        });
    }
}
