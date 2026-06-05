// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer discovery: turn a model identity into a NIXL metadata blob to pull from,
//! via the P2P registry. This replaces the hand-passed `HOLDER_MD` of the spike;
//! the blob feeds straight into [`crate::cached::transfer::puller::Puller::pull`].
//!
//! Selection spreads load across the READY holders instead of always taking the
//! first: each node walks the holder list starting from a deterministic offset
//! seeded by its own id, so when N nodes wipe and refill together they fan out
//! across the available sources rather than stampeding one. As more nodes finish
//! and advertise, the holder set grows and the fan-out widens (the cascade).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use modelexpress_common::grpc::p2p::SourceIdentity;
use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;

use super::registry::Registry;

/// Find a peer holding `identity` and return its NIXL metadata blob, spreading
/// the choice across READY holders by `seed` (the puller's stable worker id).
/// Walks holders in the seeded rotation and returns the first that carries a
/// NIXL blob, skipping any that don't; `None` when no holder currently serves
/// the model (the caller then falls back to origin).
pub async fn discover_blob(
    registry: &mut Registry,
    identity: SourceIdentity,
    seed: &str,
) -> anyhow::Result<Option<Vec<u8>>> {
    let instances = registry.list_ready(identity).await?;
    for idx in peer_order(seed, instances.len()) {
        let instance = &instances[idx];
        let Some(worker) = registry
            .get_worker(instance.mx_source_id.clone(), instance.worker_id.clone())
            .await?
        else {
            continue;
        };
        if let Some(BackendMetadata::NixlMetadata(blob)) = worker.backend_metadata {
            return Ok(Some(blob));
        }
    }
    Ok(None)
}

/// A deterministic visiting order over `n` holders, rotated by a hash of `seed`.
/// Different seeds start at different holders (load spread), the same seed is
/// stable across calls (so retries are predictable and the logic is testable),
/// and every index appears exactly once (so a holder lacking a blob is still
/// tried as a fallback). Empty when `n == 0`.
pub fn peer_order(seed: &str, n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let start = usize::try_from(hasher.finish().checked_rem(n as u64).unwrap_or(0)).unwrap_or(0);
    // Rotate by chaining ranges rather than modular arithmetic: start..n then
    // 0..start visits every holder once, starting at `start`. No overflow risk.
    (start..n).chain(0..start).collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn peer_order_is_empty_for_no_holders() {
        assert!(peer_order("node-a", 0).is_empty());
    }

    #[test]
    fn peer_order_is_a_full_permutation() {
        for seed in ["node-a", "node-b", "pod-xyz-123", ""] {
            let order = peer_order(seed, 5);
            assert_eq!(order.len(), 5);
            let unique: HashSet<usize> = order.iter().copied().collect();
            assert_eq!(unique.len(), 5, "{seed}: every holder visited once");
            assert!(order.iter().all(|&i| i < 5));
        }
    }

    #[test]
    fn peer_order_is_stable_for_a_given_seed() {
        assert_eq!(peer_order("node-a", 7), peer_order("node-a", 7));
    }

    #[test]
    fn different_seeds_spread_the_starting_holder() {
        // Across a spread of node ids, the first-picked holder is not always 0;
        // that fan-out is the whole point of seeding by worker id.
        let starts: HashSet<usize> = (0..32)
            .map(|i| peer_order(&format!("node-{i}"), 8)[0])
            .collect();
        assert!(
            starts.len() > 1,
            "seeds should land on more than one starting holder, got {starts:?}"
        );
    }

    #[test]
    fn single_holder_is_always_chosen() {
        assert_eq!(peer_order("anything", 1), vec![0]);
    }
}
