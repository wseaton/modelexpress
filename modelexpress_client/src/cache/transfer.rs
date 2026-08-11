// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire protocol for the NVMe cache transfer.
//!
//! The transfer path is `FILE -(POSIX/O_DIRECT)-> DRAM -(UCX/RDMA)-> DRAM
//! -(POSIX)-> FILE`, driven by an active stager (NIXL cannot one-sided
//! RDMA-read a peer's file). Control flow is notification-based: the puller
//! asks for a manifest, then per shard sends its receive descriptors and the
//! stager streams the bytes. This module holds the types shared by both sides;
//! it carries no NIXL dependency so it builds and unit-tests everywhere.

use std::os::unix::io::RawFd;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub mod buffer;
pub mod cache_layout;
pub mod puller;
pub mod stager;

/// NIXL FFI transport. Only compiled with the `nixl` feature (needs `libnixl`).
#[cfg(feature = "nixl")]
pub mod nixl;

/// In-process loopback transport used to exercise the protocol in unit tests
/// without RDMA hardware. It moves real bytes (real file IO, real `memcpy`
/// between two staging buffers), so the hash verification on the puller is
/// genuine; only the fabric (notifications, metadata) is local.
#[cfg(any(test, feature = "test-support"))]
pub mod loopback;

/// Transfer descriptor granularity. Files transfer as `CHUNK`-sized descriptors
/// plus a final partial one, so the POSIX backend gets IO parallelism without
/// padding the file: exactly `true_size` bytes move and source files are never
/// mutated. 16 MiB is 4K-aligned (relevant once Phase 4 re-enables `O_DIRECT`).
pub const CHUNK: u64 = 16 * 1024 * 1024;

/// One file in a model's snapshot, as advertised in a manifest. The whole
/// snapshot is transferred (weights, config, tokenizer, ...), not just weights,
/// so the pulled model is immediately usable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    /// Path relative to the model's snapshot directory (e.g.
    /// `model-00001.safetensors`, `config.json`). Reconstructed verbatim,
    /// subdirectories included, on the puller.
    pub rel_path: String,
    /// Exact size in bytes; the transfer moves exactly this many bytes and the
    /// dest file ends at this size, no padding or truncation.
    pub true_size: u64,
    /// BLAKE3 of the content, verified on the puller before the file is renamed
    /// into place.
    pub hash: String,
}

/// The files a stager will serve for one model, sent in reply to a manifest
/// request over the notification channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Content revision (HuggingFace commit, etc.) the shards belong to; the
    /// puller lands them under this revision's snapshot directory so the cache
    /// layout matches what the origin download path produces.
    pub revision: String,
    pub shards: Vec<Shard>,
}

impl Shard {
    /// Number of `CHUNK`-sized chunks needed to cover this shard (rounded up).
    /// Used only to size the staging buffer; the transfer itself is byte-exact.
    pub fn n_chunks(&self) -> u64 {
        self.true_size.div_ceil(CHUNK)
    }
}

/// Notification tags exchanged over the NIXL notif channel. Tiny and
/// fixed-shape so framing is trivial. `PULL`/`DONE` are followed by a 4-digit
/// zero-padded shard index, then (for `PULL`) the puller's serialized
/// descriptors. Mirrors the validated benchmark protocol.
pub mod notif {
    /// Puller -> stager: `M?` + u32-LE model-name length + model name + the
    /// puller's metadata blob. Names the model so a multi-model stager knows
    /// which snapshot to serve. See [`encode_manifest_request`].
    pub const MANIFEST_REQUEST: &[u8] = b"M?";
    /// Stager -> puller: `MANIFEST` followed by JSON-encoded [`super::Manifest`].
    pub const MANIFEST: &[u8] = b"MANIFEST";
    /// Stager -> puller: the stager does not hold the requested model.
    pub const NOT_FOUND: &[u8] = b"NOPE";
    /// Puller -> stager: end of session.
    pub const BYE: &[u8] = b"BYE";
    /// Puller -> stager: `PULL` + 4-digit index + the puller's buffer base addr.
    pub const PULL: &[u8] = b"PULL";
    /// Stager -> puller: `DONE` + 4-digit index, shard delivered.
    pub const DONE: &[u8] = b"DONE";

    /// Format a `PULL`/`DONE`-style tag with its zero-padded shard index.
    pub fn indexed(tag: &[u8], idx: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(tag.len().saturating_add(4));
        v.extend_from_slice(tag);
        v.extend_from_slice(format!("{idx:04}").as_bytes());
        v
    }

    /// Build a `MANIFEST_REQUEST` notif naming `model` and carrying the puller's
    /// metadata `md` blob (binary, so it is length-framed rather than delimited).
    pub fn encode_manifest_request(model: &str, md: &[u8]) -> Vec<u8> {
        let name_len = u32::try_from(model.len()).unwrap_or(u32::MAX);
        let mut v = MANIFEST_REQUEST.to_vec();
        v.extend_from_slice(&name_len.to_le_bytes());
        v.extend_from_slice(model.as_bytes());
        v.extend_from_slice(md);
        v
    }

    /// Parse a `MANIFEST_REQUEST` notif into `(model, md)`. Returns `None` on a
    /// truncated or malformed frame.
    pub fn decode_manifest_request(msg: &[u8]) -> Option<(&str, &[u8])> {
        let body = msg.strip_prefix(MANIFEST_REQUEST)?;
        let len_bytes: [u8; 4] = body.get(0..4)?.try_into().ok()?;
        let name_len = usize::try_from(u32::from_le_bytes(len_bytes)).ok()?;
        let end = 4usize.checked_add(name_len)?;
        let model = std::str::from_utf8(body.get(4..end)?).ok()?;
        let md = body.get(end..)?;
        Some((model, md))
    }
}

/// Which side of a transfer an agent plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Holds the model on local NVMe and stages it out on request.
    Holder,
    /// Requests the model and writes it to local NVMe.
    Puller,
}

/// The byte-moving surface the transfer protocol drives. The production
/// implementor is [`nixl::NixlAgent`] (real RDMA + `O_DIRECT`); the in-process
/// [`loopback::Loopback`] is used in tests. Keeping the protocol generic over
/// this trait makes the notification ordering unit-testable with no hardware.
///
/// Addresses and file descriptors are passed by value because the protocol owns
/// the [`buffer::StagingBuffer`] and the open shard files; the transport only
/// moves bytes between them. The three legs mirror the active-stager pattern:
/// `read_file_to_dram` (POSIX) -> `write_dram_to_peer` (UCX/RDMA) on the stager,
/// then the puller's POSIX write leg, which is split into post/wait
/// (`post_write_dram_to_file` -> `wait_write`) so the puller can overlap one
/// shard's NVMe write with the next shard's RDMA receive (the double-buffer).
pub trait Transport {
    /// A posted-but-not-yet-complete write leg, returned by
    /// [`Transport::post_write_dram_to_file`] and awaited by
    /// [`Transport::wait_write`]. Opaque so each transport carries whatever it
    /// needs (NIXL keeps the in-flight request; the loopback keeps the args).
    type WriteHandle;

    /// This agent's NIXL name.
    fn name(&self) -> &str;

    /// This agent's metadata blob, handed to a peer so it can `load_remote` us.
    fn local_md(&self) -> anyhow::Result<Vec<u8>>;

    /// Load a peer's metadata blob; returns the peer's agent name.
    fn load_remote(&mut self, blob: &[u8]) -> anyhow::Result<String>;

    /// Send a notification to `peer`.
    fn send_notif(&self, peer: &str, msg: &[u8]) -> anyhow::Result<()>;

    /// Drain pending notifications as `(sender, payload)` pairs.
    fn drain_notifs(&self) -> anyhow::Result<Vec<(String, Vec<u8>)>>;

    /// Register the staging buffer as both an RDMA endpoint and a storage
    /// endpoint. Called once before any transfer.
    fn register_dram(&mut self, base: usize, len: usize) -> anyhow::Result<()>;

    /// Register an open file as a storage endpoint for the POSIX legs. `len` is
    /// the file's exact size.
    fn register_file(&mut self, fd: RawFd, len: usize) -> anyhow::Result<()>;

    /// Storage leg: read exactly `size` bytes of a registered file into the
    /// registered staging buffer (full `CHUNK` descriptors plus a final partial).
    fn read_file_to_dram(&self, dram_base: usize, fd: RawFd, size: u64) -> anyhow::Result<()>;

    /// Network leg: RDMA-write exactly `size` bytes from the staging buffer into
    /// `peer`'s registered buffer at `peer_base`.
    fn write_dram_to_peer(
        &self,
        dram_base: usize,
        peer_base: usize,
        peer: &str,
        size: u64,
    ) -> anyhow::Result<()>;

    /// Storage leg, posted (non-blocking): begin writing exactly `size` bytes of
    /// the staging buffer at `dram_base` down to a registered file, returning a
    /// handle to await later. The `dram_base` region MUST stay intact until the
    /// returned handle is passed to [`Transport::wait_write`]; the puller's slot
    /// discipline guarantees that.
    fn post_write_dram_to_file(
        &self,
        dram_base: usize,
        fd: RawFd,
        size: u64,
    ) -> anyhow::Result<Self::WriteHandle>;

    /// Block until a write posted by [`Transport::post_write_dram_to_file`]
    /// completes.
    fn wait_write(&self, handle: Self::WriteHandle) -> anyhow::Result<()>;
}

/// Split `size` bytes into up to `streams` contiguous chunk-aligned stripes as
/// `(offset, len)` pairs. Stripes land on `CHUNK` boundaries so `O_DIRECT`
/// alignment is preserved; only the final stripe carries the partial tail. The
/// stripe count is capped at the chunk count, so tiny transfers degrade to one
/// stripe rather than zero-length ones.
pub fn stripe_ranges(size: u64, streams: usize) -> Vec<(u64, u64)> {
    if size == 0 {
        return Vec::new();
    }
    let n_chunks = size.div_ceil(CHUNK);
    let streams = u64::try_from(streams.max(1)).unwrap_or(1).min(n_chunks);
    let chunks_per_stripe = n_chunks.div_ceil(streams);
    let stripe_bytes = chunks_per_stripe.saturating_mul(CHUNK);
    let mut out = Vec::new();
    let mut off = 0u64;
    while off < size {
        let len = stripe_bytes.min(size.saturating_sub(off));
        out.push((off, len));
        off = off.saturating_add(stripe_bytes);
    }
    out
}

/// Throughput in GB/s (decimal) for moving `bytes` in `dur`. Used for the
/// per-leg transfer telemetry.
pub fn gbps(bytes: u64, dur: Duration) -> f64 {
    let secs = dur.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / secs / 1e9
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn shard(rel_path: &str, true_size: u64) -> Shard {
        Shard {
            rel_path: rel_path.into(),
            true_size,
            hash: String::new(),
        }
    }

    #[test]
    fn n_chunks_rounds_up() {
        assert_eq!(shard("a", CHUNK.saturating_add(1)).n_chunks(), 2);
        assert_eq!(shard("a", CHUNK.saturating_mul(2)).n_chunks(), 2);
        assert_eq!(shard("a", 1).n_chunks(), 1);
    }

    #[test]
    fn indexed_tag_is_zero_padded() {
        assert_eq!(notif::indexed(notif::PULL, 7), b"PULL0007");
        assert_eq!(notif::indexed(notif::DONE, 1234), b"DONE1234");
    }

    #[test]
    fn stripe_ranges_cover_exactly_once() {
        for (size, streams) in [
            (1u64, 1usize),
            (CHUNK, 4),
            (CHUNK * 7 + 5, 4),
            (CHUNK * 232 + 12345, 8),
            (CHUNK * 3, 16),
        ] {
            let stripes = stripe_ranges(size, streams);
            assert!(stripes.len() <= streams.max(1));
            let mut expect_off = 0u64;
            for (off, len) in &stripes {
                assert_eq!(*off, expect_off, "contiguous");
                assert_eq!(off % CHUNK, 0, "chunk-aligned start");
                assert!(*len > 0);
                expect_off += len;
            }
            assert_eq!(expect_off, size, "byte-exact coverage");
        }
    }

    #[test]
    fn stripe_ranges_empty_for_zero_size() {
        assert!(stripe_ranges(0, 4).is_empty());
    }

    #[test]
    fn stripe_ranges_single_stream_is_one_stripe() {
        assert_eq!(stripe_ranges(CHUNK * 5 + 1, 1).len(), 1);
    }

    #[test]
    fn manifest_round_trips_json() {
        let m = Manifest {
            revision: "abc123".into(),
            shards: vec![shard("model-00001.safetensors", 42)],
        };
        let bytes = serde_json::to_vec(&m).expect("serialize");
        let back: Manifest = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(m, back);
    }

    #[test]
    fn manifest_request_frames_model_and_md() {
        let md = vec![0u8, 1, 2, 255, 254];
        let msg = notif::encode_manifest_request("google-t5/t5-small", &md);
        let (model, got_md) = notif::decode_manifest_request(&msg).expect("decode");
        assert_eq!(model, "google-t5/t5-small");
        assert_eq!(got_md, md.as_slice());
    }

    #[test]
    fn decode_manifest_request_rejects_truncated() {
        assert!(notif::decode_manifest_request(b"M?").is_none());
        assert!(notif::decode_manifest_request(b"nope").is_none());
    }
}
