// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, pre-registered staging buffer for the NVMe transfer.
//!
//! One page-aligned anonymous mapping, allocated once and reused for every
//! shard. Page alignment satisfies `O_DIRECT`; the fixed size is the daemon's
//! entire DRAM cap for a transfer, so model size never enters the footprint.
//! `mmap` is 64-bit, so buffers larger than 2 GB work (NIXL's bundled
//! `malloc_passthru` is 32-bit and rejects them, which is why we map our own).

use memmap2::MmapMut;

use super::CHUNK;

/// A reusable, page-aligned host buffer sized to a whole number of [`CHUNK`]s.
///
/// The base address is stable for the lifetime of the buffer, so it can be
/// registered with NIXL once and referenced by every transfer.
pub struct StagingBuffer {
    map: MmapMut,
    n_chunks: u64,
}

impl StagingBuffer {
    /// Allocate `n_chunks * CHUNK` bytes of page-aligned anonymous host memory.
    pub fn new(n_chunks: u64) -> std::io::Result<Self> {
        let bytes = n_chunks.checked_mul(CHUNK).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "staging buffer size overflow",
            )
        })?;
        let len = usize::try_from(bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "staging buffer too large for usize",
            )
        })?;
        let map = MmapMut::map_anon(len)?;
        Ok(Self { map, n_chunks })
    }

    /// Base virtual address, stable for the buffer's lifetime. Passed to NIXL
    /// at registration and used to compute per-chunk transfer descriptors.
    pub fn base_addr(&self) -> usize {
        self.map.as_ptr() as usize
    }

    /// Total size in bytes (`n_chunks * CHUNK`).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the buffer has zero length (only true for `new(0)`).
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Number of [`CHUNK`]-sized chunks the buffer holds.
    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.map
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.map
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn allocates_chunk_multiple_and_stable_address() {
        let buf = StagingBuffer::new(1).expect("alloc");
        assert_eq!(buf.len() as u64, CHUNK);
        assert_eq!(buf.n_chunks(), 1);
        assert!(!buf.is_empty());
        assert_ne!(buf.base_addr(), 0);
        // Address is stable across calls (NIXL registration relies on this).
        assert_eq!(buf.base_addr(), buf.base_addr());
    }

    #[test]
    fn round_trips_bytes() {
        let mut buf = StagingBuffer::new(1).expect("alloc");
        let pattern: Vec<u8> = (0..256u32)
            .map(|i| (i % 251) as u8)
            .cycle()
            .take(buf.len())
            .collect();
        buf.as_mut_slice().copy_from_slice(&pattern);
        assert_eq!(buf.as_slice(), pattern.as_slice());
    }

    #[test]
    fn page_aligned_for_o_direct() {
        let buf = StagingBuffer::new(1).expect("alloc");
        // Anonymous mmap is page-aligned; O_DIRECT needs at least 4K alignment.
        assert_eq!(buf.base_addr() % 4096, 0);
    }
}
