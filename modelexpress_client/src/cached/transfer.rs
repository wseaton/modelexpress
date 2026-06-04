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

use serde::{Deserialize, Serialize};

pub mod buffer;

/// NIXL FFI transport. Only compiled with the `nixl` feature (needs `libnixl`).
#[cfg(feature = "nixl")]
pub mod nixl;

/// Phase-0 holder/puller spike (a Rust port of the validated benchmark).
#[cfg(feature = "nixl")]
pub mod spike;

/// Transfer and registration granularity. 16 MiB is 4K-aligned, keeping every
/// `O_DIRECT` read/write and NIXL FILE descriptor aligned. Matches the
/// validated Python benchmark.
pub const CHUNK: u64 = 16 * 1024 * 1024;

/// One model-weight shard as advertised in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    /// File name within the model's snapshot directory.
    pub name: String,
    /// True (unpadded) size in bytes; the dest file is truncated back to this
    /// after the padded `O_DIRECT` transfer.
    pub true_size: u64,
    /// SHA-256 of the true-size content, verified on the puller before the file
    /// is renamed into place.
    pub sha256: String,
}

/// The shards a stager will serve for one model, sent in reply to a manifest
/// request over the notification channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub shards: Vec<Shard>,
}

impl Shard {
    /// Number of `CHUNK`-sized chunks needed to cover this shard, padded up.
    pub fn n_chunks(&self) -> u64 {
        self.true_size.div_ceil(CHUNK)
    }

    /// Padded transfer size: `n_chunks * CHUNK`.
    pub fn padded_size(&self) -> u64 {
        self.n_chunks().saturating_mul(CHUNK)
    }
}

/// Notification tags exchanged over the NIXL notif channel. Tiny and
/// fixed-shape so framing is trivial. `PULL`/`DONE` are followed by a 4-digit
/// zero-padded shard index, then (for `PULL`) the puller's serialized
/// descriptors. Mirrors the validated benchmark protocol.
pub mod notif {
    /// Puller -> stager: "send me your manifest".
    pub const MANIFEST_REQUEST: &[u8] = b"M?";
    /// Stager -> puller: `MANIFEST` followed by JSON-encoded [`super::Manifest`].
    pub const MANIFEST: &[u8] = b"MANIFEST";
    /// Puller -> stager: end of session.
    pub const BYE: &[u8] = b"BYE";
    /// Puller -> stager: `PULL` + 4-digit index + serialized receive descs.
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
}

/// Which side of a transfer an agent plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Holds the model on local NVMe and stages it out on request.
    Holder,
    /// Requests the model and writes it to local NVMe.
    Puller,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn n_chunks_rounds_up() {
        let s = Shard {
            name: "a".into(),
            true_size: CHUNK.saturating_add(1),
            sha256: String::new(),
        };
        assert_eq!(s.n_chunks(), 2);
        assert_eq!(s.padded_size(), CHUNK.saturating_mul(2));
    }

    #[test]
    fn exact_multiple_does_not_overpad() {
        let s = Shard {
            name: "a".into(),
            true_size: CHUNK.saturating_mul(2),
            sha256: String::new(),
        };
        assert_eq!(s.n_chunks(), 2);
    }

    #[test]
    fn indexed_tag_is_zero_padded() {
        assert_eq!(notif::indexed(notif::PULL, 7), b"PULL0007");
        assert_eq!(notif::indexed(notif::DONE, 1234), b"DONE1234");
    }

    #[test]
    fn manifest_round_trips_json() {
        let m = Manifest {
            shards: vec![Shard {
                name: "model-00001.safetensors".into(),
                true_size: 42,
                sha256: "ab".into(),
            }],
        };
        let bytes = serde_json::to_vec(&m).expect("serialize");
        let back: Manifest = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(m, back);
    }
}
