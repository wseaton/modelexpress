// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-peer transfer planning.
//!
//! For each tensor the requester needs, the planner picks an owning peer
//! using greedy minimum-load assignment: among peers that own that tensor,
//! the one with the smallest running byte total wins (tie-break: worker_id
//! ascending, for deterministic plans across retries).
//!
//! Unlike the previous version, the planner takes the **union** of peer
//! catalogs rather than the intersection. This is the only correct choice
//! for MoE models with expert-parallel sharding, where different peers own
//! disjoint subsets of expert weights.
//!
//! Tensors no peer owns are returned in `TransferPlan::uncovered` so the
//! requester can disk-load them (or skip if already present locally).
//!
//! Dtype conflict policy: if multiple peers advertise the same tensor name
//! with different dtypes, that tensor is treated as uncovered (we don't
//! silently pick one). The caller surfaces this in `PlanDiagnostics.note`.

use std::cmp::Reverse;
use std::collections::HashMap;

use crate::p2p::backend::WorkerRecord;

/// One tensor's planning-relevant metadata. Mirrors the proto
/// `TensorCatalogEntry` but lives in the server's domain layer.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub name: String,
    pub byte_len: u64,
    pub dtype: String,
}

/// Bundled peer data the RPC handler assembles before calling the planner.
#[derive(Debug, Clone)]
pub struct PeerCandidate {
    pub source_id: String,
    pub worker_id: String,
    /// Used to materialize the response (NIXL metadata, endpoints, addrs).
    pub worker: WorkerRecord,
    /// What this peer owns. Empty means "advertised nothing"; such peers
    /// are eligible only as a destination for symmetric advertise but
    /// can't be assigned tensors to serve.
    pub catalog: Vec<CatalogEntry>,
}

/// One peer's assignment in the computed transfer plan.
#[derive(Debug, Clone)]
pub struct PeerTransferAssignment {
    pub source_id: String,
    pub worker_id: String,
    pub worker: WorkerRecord,
    pub assigned_tensor_names: Vec<String>,
}

/// Full plan: per-peer assignments plus the names of tensors no peer
/// owned. The receiver disk-loads `uncovered` itself.
#[derive(Debug, Clone, Default)]
pub struct TransferPlan {
    pub assignments: Vec<PeerTransferAssignment>,
    pub uncovered: Vec<String>,
    /// Tensors that were dropped because peers disagreed on dtype.
    /// Included in `uncovered` too; this list is for diagnostics.
    pub dtype_conflicts: Vec<String>,
}

/// Compute a transfer plan.
///
/// - `requested`: tensors the receiver needs (name + byte_len + dtype).
///   Empty list returns an empty plan (the caller falls back to disk for
///   everything).
/// - `peers`: candidates filtered by source_id + matching rank, with the
///   requester already excluded.
/// - `max_peers`: optional cap on how many peers appear in the response.
///   Peers are kept in input order before capping (caller decides
///   ordering — typically by recency or load).
///
/// Each needed tensor is assigned to its least-loaded owning peer, ties
/// broken by `worker_id` for deterministic plans across retries.
pub fn compute_transfer_plan(
    requested: &[CatalogEntry],
    peers: &[PeerCandidate],
    max_peers: Option<u32>,
) -> TransferPlan {
    if peers.is_empty() || requested.is_empty() {
        return TransferPlan::default();
    }

    let active_peers: &[PeerCandidate] = match max_peers {
        Some(n) if (n as usize) < peers.len() => &peers[..n as usize],
        _ => peers,
    };

    let n = active_peers.len();

    // Build name -> [(peer_idx, dtype)] inverted index across active peers.
    let mut owners: HashMap<&str, Vec<(usize, &str)>> = HashMap::new();
    for (idx, peer) in active_peers.iter().enumerate() {
        for entry in &peer.catalog {
            owners
                .entry(entry.name.as_str())
                .or_default()
                .push((idx, entry.dtype.as_str()));
        }
    }

    // Sort requested tensors by descending byte_len, tie-break by name
    // ascending, so the heaviest tensors get placed first (better balance)
    // and the plan is deterministic across calls.
    let mut order: Vec<&CatalogEntry> = requested.iter().collect();
    order.sort_by(|a, b| {
        Reverse(a.byte_len)
            .cmp(&Reverse(b.byte_len))
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut load = vec![0u64; n];
    let mut assignments: Vec<Vec<String>> = vec![Vec::new(); n];
    let mut uncovered: Vec<String> = Vec::new();
    let mut dtype_conflicts: Vec<String> = Vec::new();

    for need in order {
        let Some(candidates) = owners.get(need.name.as_str()) else {
            uncovered.push(need.name.clone());
            continue;
        };

        // Reject if peers disagree on dtype for this name. We don't try
        // to be clever — silently picking a peer with a different dtype
        // than the receiver expects produces corrupted weights.
        let dtypes: Vec<&str> = candidates.iter().map(|(_, d)| *d).collect();
        let first_dtype = dtypes[0];
        let dtype_ok = dtypes.iter().all(|d| *d == first_dtype) && first_dtype == need.dtype;
        if !dtype_ok {
            dtype_conflicts.push(need.name.clone());
            uncovered.push(need.name.clone());
            continue;
        }

        // Pick the owning peer with the smallest running load, tie-broken
        // by worker_id for deterministic plans across retries.
        let pick = candidates
            .iter()
            .min_by(|(a, _), (b, _)| {
                (load[*a], active_peers[*a].worker_id.as_str())
                    .cmp(&(load[*b], active_peers[*b].worker_id.as_str()))
            })
            .map(|(idx, _)| *idx);

        let Some(idx) = pick else {
            uncovered.push(need.name.clone());
            continue;
        };

        load[idx] = load[idx].saturating_add(need.byte_len);
        assignments[idx].push(need.name.clone());
    }

    let plan_assignments = active_peers
        .iter()
        .zip(assignments)
        .map(|(peer, assigned)| PeerTransferAssignment {
            source_id: peer.source_id.clone(),
            worker_id: peer.worker_id.clone(),
            worker: peer.worker.clone(),
            assigned_tensor_names: assigned,
        })
        .collect();

    TransferPlan {
        assignments: plan_assignments,
        uncovered,
        dtype_conflicts,
    }
}

/// Build a synthetic catalog from a worker's `tensors` (the addresses
/// from PublishMetadata). Used as a backward-compat shim for peers that
/// don't call AdvertiseTensorCatalog: planning sees their tensor set
/// the same way the old intersection-based planner did.
pub fn synthetic_catalog_from_worker(worker: &WorkerRecord) -> Vec<CatalogEntry> {
    worker
        .tensors
        .iter()
        .map(|t| CatalogEntry {
            name: t.name.clone(),
            byte_len: t.size,
            dtype: t.dtype.clone(),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{CatalogEntry, PeerCandidate, TransferPlan, compute_transfer_plan};
    use crate::p2p::backend::{BackendMetadataRecord, TensorRecord, WorkerRecord};

    fn entry(name: &str, byte_len: u64) -> CatalogEntry {
        CatalogEntry {
            name: name.to_string(),
            byte_len,
            dtype: "bfloat16".to_string(),
        }
    }

    fn entry_with_dtype(name: &str, byte_len: u64, dtype: &str) -> CatalogEntry {
        CatalogEntry {
            name: name.to_string(),
            byte_len,
            dtype: dtype.to_string(),
        }
    }

    fn make_peer(name: &str, catalog: &[(&str, u64)]) -> PeerCandidate {
        let cat: Vec<CatalogEntry> = catalog.iter().map(|(n, s)| entry(n, *s)).collect();
        PeerCandidate {
            source_id: "test-source".to_string(),
            worker_id: name.to_string(),
            worker: WorkerRecord {
                worker_rank: 0,
                backend_metadata: BackendMetadataRecord::Nixl(vec![0xCA, 0xFE]),
                tensors: cat
                    .iter()
                    .map(|c| TensorRecord {
                        name: c.name.clone(),
                        addr: 0x1000,
                        size: c.byte_len,
                        device_id: 0,
                        dtype: c.dtype.clone(),
                    })
                    .collect(),
                status: 2, // READY
                updated_at: 0,
                metadata_endpoint: String::new(),
                agent_name: format!("{name}-agent"),
                worker_grpc_endpoint: String::new(),
                labels: std::collections::HashMap::new(),
                tensor_catalog: None,
            },
            catalog: cat,
        }
    }

    fn make_peer_dtypes(name: &str, catalog: &[(&str, u64, &str)]) -> PeerCandidate {
        let mut p = make_peer(name, &[]);
        p.catalog = catalog
            .iter()
            .map(|(n, s, d)| entry_with_dtype(n, *s, d))
            .collect();
        p
    }

    fn assigned_bytes(asn: &super::PeerTransferAssignment, sizes: &[(&str, u64)]) -> u64 {
        asn.assigned_tensor_names
            .iter()
            .map(|n| {
                sizes
                    .iter()
                    .find(|(t, _)| t == n)
                    .expect("tensor should exist")
                    .1
            })
            .sum()
    }

    fn requested(sizes: &[(&str, u64)]) -> Vec<CatalogEntry> {
        sizes.iter().map(|(n, s)| entry(n, *s)).collect()
    }

    #[test]
    fn empty_peers_returns_empty_plan() {
        let plan = compute_transfer_plan(&requested(&[("a", 100)]), &[], None);
        assert!(plan.assignments.is_empty());
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn empty_request_returns_empty_plan() {
        let peers = vec![make_peer("p0", &[("a", 100)])];
        let plan = compute_transfer_plan(&[], &peers, None);
        assert!(plan.assignments.is_empty());
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn single_peer_gets_all_tensors() {
        let tensors = &[("a", 100), ("b", 80)];
        let peers = vec![make_peer("p0", tensors)];
        let plan = compute_transfer_plan(&requested(tensors), &peers, None);
        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0].assigned_tensor_names.len(), 2);
        assert_eq!(assigned_bytes(&plan.assignments[0], tensors), 180);
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn two_peers_balanced() {
        // All four tensors on both peers — greedy should split 140/140.
        let tensors = &[
            ("layer.0", 100),
            ("layer.1", 80),
            ("layer.2", 60),
            ("layer.3", 40),
        ];
        let peers = vec![make_peer("p0", tensors), make_peer("p1", tensors)];
        let plan = compute_transfer_plan(&requested(tensors), &peers, None);
        assert_eq!(plan.assignments.len(), 2);
        let b0 = assigned_bytes(&plan.assignments[0], tensors);
        let b1 = assigned_bytes(&plan.assignments[1], tensors);
        assert_eq!(b0 + b1, 280);
        // Allow any balanced split — tie-break is deterministic but exact
        // index assignment depends on input order. We assert balance only.
        assert!(b0.abs_diff(b1) <= 20, "expected balanced, got {b0}/{b1}");
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn moe_disjoint_experts_all_get_assigned() {
        // Realistic MoE-EP layout: shared params on every peer, experts
        // split disjointly. Each expert tensor must end up on the one
        // peer that owns it.
        let p0 = make_peer(
            "p0",
            &[
                ("shared.embed", 1000),
                ("layer.0.mlp.experts.0.w", 500),
                ("layer.0.mlp.experts.1.w", 500),
            ],
        );
        let p1 = make_peer(
            "p1",
            &[
                ("shared.embed", 1000),
                ("layer.0.mlp.experts.2.w", 500),
                ("layer.0.mlp.experts.3.w", 500),
            ],
        );
        let need = requested(&[
            ("shared.embed", 1000),
            ("layer.0.mlp.experts.0.w", 500),
            ("layer.0.mlp.experts.1.w", 500),
            ("layer.0.mlp.experts.2.w", 500),
            ("layer.0.mlp.experts.3.w", 500),
        ]);

        let plan = compute_transfer_plan(&need, &[p0, p1], None);
        assert_eq!(plan.assignments.len(), 2);
        assert!(plan.uncovered.is_empty(), "MoE plan should cover all needs");

        // Each expert is assigned to its owner.
        let p0_assigned: Vec<&str> = plan.assignments[0]
            .assigned_tensor_names
            .iter()
            .map(String::as_str)
            .collect();
        let p1_assigned: Vec<&str> = plan.assignments[1]
            .assigned_tensor_names
            .iter()
            .map(String::as_str)
            .collect();

        assert!(p0_assigned.contains(&"layer.0.mlp.experts.0.w"));
        assert!(p0_assigned.contains(&"layer.0.mlp.experts.1.w"));
        assert!(p1_assigned.contains(&"layer.0.mlp.experts.2.w"));
        assert!(p1_assigned.contains(&"layer.0.mlp.experts.3.w"));

        // shared.embed lands on exactly one peer.
        let shared_count = p0_assigned.iter().filter(|n| **n == "shared.embed").count()
            + p1_assigned.iter().filter(|n| **n == "shared.embed").count();
        assert_eq!(shared_count, 1);
    }

    #[test]
    fn three_peers_reasonably_balanced() {
        let tensors = &[
            ("a", 100),
            ("b", 90),
            ("c", 80),
            ("d", 70),
            ("e", 60),
            ("f", 50),
        ];
        let peers = vec![
            make_peer("p0", tensors),
            make_peer("p1", tensors),
            make_peer("p2", tensors),
        ];
        let plan = compute_transfer_plan(&requested(tensors), &peers, None);
        assert_eq!(plan.assignments.len(), 3);
        let totals: Vec<u64> = plan
            .assignments
            .iter()
            .map(|a| assigned_bytes(a, tensors))
            .collect();
        let total: u64 = totals.iter().sum();
        assert_eq!(total, 450);
        let max = *totals.iter().max().expect("at least one peer");
        let min = *totals.iter().min().expect("at least one peer");
        assert!(max - min <= 20, "expected balanced, got {totals:?}");
    }

    #[test]
    fn disjoint_peers_partial_coverage_no_uncovered() {
        // p0 owns "a", p1 owns "b". Both are needed. The plan must
        // pin "a" to p0 and "b" to p1 — no uncovered, no disagreement.
        let p0 = make_peer("p0", &[("a", 100)]);
        let p1 = make_peer("p1", &[("b", 100)]);
        let plan = compute_transfer_plan(&requested(&[("a", 100), ("b", 100)]), &[p0, p1], None);
        assert!(plan.uncovered.is_empty());
        let a_owner: Vec<&str> = plan
            .assignments
            .iter()
            .filter(|p| p.assigned_tensor_names.iter().any(|n| n == "a"))
            .map(|p| p.worker_id.as_str())
            .collect();
        assert_eq!(a_owner, vec!["p0"]);
        let b_owner: Vec<&str> = plan
            .assignments
            .iter()
            .filter(|p| p.assigned_tensor_names.iter().any(|n| n == "b"))
            .map(|p| p.worker_id.as_str())
            .collect();
        assert_eq!(b_owner, vec!["p1"]);
    }

    #[test]
    fn uncovered_tensors_are_reported() {
        let p0 = make_peer("p0", &[("a", 100), ("b", 80)]);
        let need = requested(&[("a", 100), ("b", 80), ("missing", 50)]);
        let plan = compute_transfer_plan(&need, &[p0], None);
        assert_eq!(plan.uncovered, vec!["missing".to_string()]);
        let assigned: Vec<&str> = plan.assignments[0]
            .assigned_tensor_names
            .iter()
            .map(String::as_str)
            .collect();
        assert!(assigned.contains(&"a"));
        assert!(assigned.contains(&"b"));
        assert!(!assigned.contains(&"missing"));
    }

    #[test]
    fn partial_overlap_uses_union_not_intersection() {
        // p0 owns {a,b,c}, p1 owns {b,c,d}. Need={a,b,c,d}.
        // Old planner would assign only {b,c} (intersection); the new
        // one must assign all four.
        let p0 = make_peer("p0", &[("a", 100), ("b", 80), ("c", 60)]);
        let p1 = make_peer("p1", &[("b", 80), ("c", 60), ("d", 40)]);
        let plan = compute_transfer_plan(
            &requested(&[("a", 100), ("b", 80), ("c", 60), ("d", 40)]),
            &[p0, p1],
            None,
        );
        assert!(plan.uncovered.is_empty());
        let mut all: Vec<&str> = plan
            .assignments
            .iter()
            .flat_map(|p| p.assigned_tensor_names.iter().map(String::as_str))
            .collect();
        all.sort();
        assert_eq!(all, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn dtype_conflict_drops_tensor_to_uncovered() {
        // p0 advertises "a" as bf16, p1 advertises "a" as fp8. The
        // planner refuses to pick — adds "a" to uncovered + dtype_conflicts.
        let p0 = make_peer_dtypes("p0", &[("a", 100, "bfloat16")]);
        let p1 = make_peer_dtypes("p1", &[("a", 100, "float8_e4m3fn")]);
        let plan = compute_transfer_plan(&requested(&[("a", 100)]), &[p0, p1], None);
        assert_eq!(plan.uncovered, vec!["a".to_string()]);
        assert_eq!(plan.dtype_conflicts, vec!["a".to_string()]);
        for asn in &plan.assignments {
            assert!(asn.assigned_tensor_names.is_empty());
        }
    }

    #[test]
    fn requested_dtype_mismatch_drops_tensor() {
        // Receiver wants bf16, peer offers fp8: do not silently corrupt.
        let p0 = make_peer_dtypes("p0", &[("a", 100, "float8_e4m3fn")]);
        let need = vec![entry_with_dtype("a", 100, "bfloat16")];
        let plan = compute_transfer_plan(&need, &[p0], None);
        assert_eq!(plan.uncovered, vec!["a".to_string()]);
        assert_eq!(plan.dtype_conflicts, vec!["a".to_string()]);
    }

    #[test]
    fn max_peers_limits_peer_count() {
        let tensors = &[("a", 100), ("b", 80), ("c", 60)];
        let peers = vec![
            make_peer("p0", tensors),
            make_peer("p1", tensors),
            make_peer("p2", tensors),
        ];
        let plan = compute_transfer_plan(&requested(tensors), &peers, Some(2));
        assert_eq!(plan.assignments.len(), 2);
        let all: Vec<&str> = plan
            .assignments
            .iter()
            .flat_map(|p| p.assigned_tensor_names.iter().map(String::as_str))
            .collect();
        assert_eq!(all.len(), 3);
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn every_requested_tensor_assigned_exactly_once() {
        let tensors = &[("a", 100), ("b", 90), ("c", 80), ("d", 70), ("e", 60)];
        let peers = vec![
            make_peer("p0", tensors),
            make_peer("p1", tensors),
            make_peer("p2", tensors),
        ];
        let plan = compute_transfer_plan(&requested(tensors), &peers, None);
        let mut all: Vec<&str> = plan
            .assignments
            .iter()
            .flat_map(|p| p.assigned_tensor_names.iter().map(String::as_str))
            .collect();
        all.sort();
        assert_eq!(all, vec!["a", "b", "c", "d", "e"]);
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn deterministic_across_repeated_calls() {
        // Same inputs -> same plan. Important so retries don't flap
        // between assignments and waste already-warm RDMA connections.
        let tensors = &[("a", 100), ("b", 80), ("c", 60), ("d", 40)];
        let peers = vec![make_peer("p0", tensors), make_peer("p1", tensors)];
        let p1 = compute_transfer_plan(&requested(tensors), &peers, None);
        let p2 = compute_transfer_plan(&requested(tensors), &peers, None);
        let collect = |t: &TransferPlan| -> Vec<Vec<String>> {
            t.assignments
                .iter()
                .map(|a| a.assigned_tensor_names.clone())
                .collect()
        };
        assert_eq!(collect(&p1), collect(&p2));
    }
}
