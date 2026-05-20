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
//!
//! Shape conflict policy: same idea for shapes. Non-empty shapes that
//! disagree (peer-vs-peer, or peer-vs-requester) mean someone's layout
//! drifted; the tensor is dropped to uncovered rather than transferred as
//! mismatched bytes. An unspecified shape (`None`) skips this check.

use std::cmp::{Ordering, Reverse};
use std::collections::HashMap;

use crate::p2p::backend::WorkerRecord;
use crate::p2p::informer::{InformerRegistry, ScoreCtx};

/// Per-tensor candidate ordering used by the greedy pick.
///
/// We don't derive `Ord` because `score` is an `f64` (no total order on
/// NaN); the explicit `cmp` formalizes the lex ordering and makes it
/// inspectable. Lower load wins, ties broken by higher informer score,
/// then by `worker_id` ascending for deterministic plans.
#[derive(Debug, Clone, Copy)]
pub struct PeerRank<'a> {
    /// Bytes already assigned to this peer in the current plan.
    pub load_bytes: u64,
    /// Composite informer score for this peer from the caller's POV.
    /// `0.0` is the no-signal neutral baseline. Higher = better.
    pub score: f64,
    /// Stable tiebreak. Borrowed from the candidate list.
    pub worker_id: &'a str,
}

impl PeerRank<'_> {
    /// Total ordering for greedy selection: lower load first, then
    /// higher score, then lower worker_id. Treats NaN as Equal so a
    /// misconfigured informer can never panic the planner.
    ///
    /// Inherent method (not `impl Ord`) because `score: f64` has no
    /// total order — implementing `Ord` would require wrapping in
    /// `OrderedFloat` or hashing NaN out, and the planner only ever
    /// uses `cmp` directly. Allow the clippy lint since the method
    /// name is intentional: callers explicitly do `peer.cmp(&other)`.
    #[allow(clippy::should_implement_trait)]
    pub fn cmp(&self, other: &Self) -> Ordering {
        self.load_bytes
            .cmp(&other.load_bytes)
            .then_with(|| {
                // Higher score is better, so reverse the partial_cmp.
                other
                    .score
                    .partial_cmp(&self.score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| self.worker_id.cmp(other.worker_id))
    }
}

/// One tensor's planning-relevant metadata. Mirrors the proto
/// `TensorCatalogEntry` but lives in the server's domain layer.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub name: String,
    pub byte_len: u64,
    pub dtype: String,
    /// Tensor shape. `None` means "unspecified" and skips shape validation.
    pub shape: Option<Vec<i64>>,
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
    /// Tensors that were dropped because peers (or the requester) disagreed
    /// on shape. Included in `uncovered` too; this list is for diagnostics.
    pub shape_conflicts: Vec<String>,
}

/// Optional context for caller-relative scoring. `None` means the
/// planner skips informer scoring entirely and uses the load+worker_id
/// ordering only — preserving the pre-informer behavior.
#[derive(Debug, Clone, Copy)]
pub struct ScoringContext<'a> {
    /// Worker ID of the requesting peer. Informers use this for
    /// caller-relative signals (e.g., topology distance).
    pub caller_worker_id: &'a str,
    /// Labels the caller published. Informers use these for
    /// caller-relative label comparisons (same rack, same tenant).
    pub caller_labels: &'a HashMap<String, String>,
    /// Source of composite scores.
    pub registry: &'a InformerRegistry,
}

/// One peer that owns a given tensor name, with the dtype and (optional)
/// shape it advertised. Used to build the per-name owner index and run
/// dtype/shape conflict checks before assignment.
type TensorOwner<'a> = (usize, &'a str, Option<&'a [i64]>);

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
/// - `scoring`: optional informer-backed scorer. When present, peers
///   that any informer hard-excludes (returns `None`) are filtered out
///   before assignment, and the per-tensor pick uses the composite
///   score as a tiebreaker behind load.
pub fn compute_transfer_plan(
    requested: &[CatalogEntry],
    peers: &[PeerCandidate],
    max_peers: Option<u32>,
    scoring: Option<ScoringContext<'_>>,
) -> TransferPlan {
    if peers.is_empty() || requested.is_empty() {
        return TransferPlan::default();
    }

    let active_peers: &[PeerCandidate] = match max_peers {
        Some(n) if (n as usize) < peers.len() => &peers[..n as usize],
        _ => peers,
    };

    // Resolve per-peer composite scores once up front. Hard-excluded
    // peers (any informer returns None) are filtered out of candidacy
    // entirely. Soft scores default to 0.0 when no scoring context.
    let n = active_peers.len();
    let scores: Vec<Option<f64>> = active_peers
        .iter()
        .map(|p| match scoring {
            Some(sctx) => {
                let ctx = ScoreCtx {
                    caller_worker_id: sctx.caller_worker_id,
                    peer_worker_id: &p.worker_id,
                    caller_labels: sctx.caller_labels,
                    peer_labels: &p.worker.labels,
                };
                sctx.registry.composite_score(&ctx)
            }
            None => Some(0.0),
        })
        .collect();

    // Build name -> [owner] inverted index across active peers, skipping
    // peers that are hard-excluded for this caller. Each owner carries its
    // peer index, advertised dtype, and (optional) shape for conflict checks.
    let mut owners: HashMap<&str, Vec<TensorOwner<'_>>> = HashMap::new();
    for (idx, peer) in active_peers.iter().enumerate() {
        if scores[idx].is_none() {
            continue;
        }
        for entry in &peer.catalog {
            owners.entry(entry.name.as_str()).or_default().push((
                idx,
                entry.dtype.as_str(),
                entry.shape.as_deref(),
            ));
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
    let mut shape_conflicts: Vec<String> = Vec::new();

    for need in order {
        let Some(candidates) = owners.get(need.name.as_str()) else {
            uncovered.push(need.name.clone());
            continue;
        };

        // Reject if peers disagree on dtype for this name. We don't try
        // to be clever — silently picking a peer with a different dtype
        // than the receiver expects produces corrupted weights.
        let dtypes: Vec<&str> = candidates.iter().map(|(_, d, _)| *d).collect();
        let first_dtype = dtypes[0];
        let dtype_ok = dtypes.iter().all(|d| *d == first_dtype) && first_dtype == need.dtype;
        if !dtype_ok {
            dtype_conflicts.push(need.name.clone());
            uncovered.push(need.name.clone());
            continue;
        }

        // Reject if any specified shapes disagree (peer-vs-peer or
        // peer-vs-requester). Unspecified shapes (`None`) are skipped, so a
        // peer that never advertised a shape doesn't block the transfer.
        let mut ref_shape: Option<&[i64]> = need.shape.as_deref();
        let shape_ok = candidates.iter().all(|(_, _, s)| match (ref_shape, s) {
            (_, None) => true,
            (None, Some(peer_shape)) => {
                ref_shape = Some(peer_shape);
                true
            }
            (Some(want), Some(peer_shape)) => want == *peer_shape,
        });
        if !shape_ok {
            shape_conflicts.push(need.name.clone());
            uncovered.push(need.name.clone());
            continue;
        }

        // Pick the candidate peer minimizing PeerRank: load first,
        // composite score second (higher better), worker_id third.
        let pick = candidates
            .iter()
            .min_by(|(a, _, _), (b, _, _)| {
                let ra = PeerRank {
                    load_bytes: load[*a],
                    // Unwrap: any hard-excluded peer was filtered above.
                    score: scores[*a].unwrap_or(0.0),
                    worker_id: active_peers[*a].worker_id.as_str(),
                };
                let rb = PeerRank {
                    load_bytes: load[*b],
                    score: scores[*b].unwrap_or(0.0),
                    worker_id: active_peers[*b].worker_id.as_str(),
                };
                ra.cmp(&rb)
            })
            .map(|(idx, _, _)| *idx);

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
        shape_conflicts,
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
            // PublishMetadata descriptors carry no shape; synthetic
            // entries are shape-unspecified and skip shape validation.
            shape: None,
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
            shape: None,
        }
    }

    fn entry_with_dtype(name: &str, byte_len: u64, dtype: &str) -> CatalogEntry {
        CatalogEntry {
            name: name.to_string(),
            byte_len,
            dtype: dtype.to_string(),
            shape: None,
        }
    }

    fn entry_with_shape(name: &str, byte_len: u64, shape: &[i64]) -> CatalogEntry {
        CatalogEntry {
            name: name.to_string(),
            byte_len,
            dtype: "bfloat16".to_string(),
            shape: Some(shape.to_vec()),
        }
    }

    /// Build a peer whose catalog carries explicit per-tensor shapes.
    fn make_peer_shapes(name: &str, catalog: &[(&str, u64, &[i64])]) -> PeerCandidate {
        let mut p = make_peer(name, &[]);
        p.catalog = catalog
            .iter()
            .map(|(n, s, shape)| entry_with_shape(n, *s, shape))
            .collect();
        p
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
        let plan = compute_transfer_plan(&requested(&[("a", 100)]), &[], None, None);
        assert!(plan.assignments.is_empty());
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn empty_request_returns_empty_plan() {
        let peers = vec![make_peer("p0", &[("a", 100)])];
        let plan = compute_transfer_plan(&[], &peers, None, None);
        assert!(plan.assignments.is_empty());
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn single_peer_gets_all_tensors() {
        let tensors = &[("a", 100), ("b", 80)];
        let peers = vec![make_peer("p0", tensors)];
        let plan = compute_transfer_plan(&requested(tensors), &peers, None, None);
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
        let plan = compute_transfer_plan(&requested(tensors), &peers, None, None);
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

        let plan = compute_transfer_plan(&need, &[p0, p1], None, None);
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
        let plan = compute_transfer_plan(&requested(tensors), &peers, None, None);
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
        let plan =
            compute_transfer_plan(&requested(&[("a", 100), ("b", 100)]), &[p0, p1], None, None);
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
        let plan = compute_transfer_plan(&need, &[p0], None, None);
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
        let plan = compute_transfer_plan(&requested(&[("a", 100)]), &[p0, p1], None, None);
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
        let plan = compute_transfer_plan(&need, &[p0], None, None);
        assert_eq!(plan.uncovered, vec!["a".to_string()]);
        assert_eq!(plan.dtype_conflicts, vec!["a".to_string()]);
    }

    #[test]
    fn shape_conflict_between_peers_drops_tensor() {
        // p0 and p1 both advertise "a" but with divergent shapes (e.g. an
        // elastic-EP shuffle gone wrong). Refuse rather than transfer
        // mismatched bytes.
        let p0 = make_peer_shapes("p0", &[("a", 100, &[2, 50])]);
        let p1 = make_peer_shapes("p1", &[("a", 100, &[5, 20])]);
        let plan = compute_transfer_plan(&requested(&[("a", 100)]), &[p0, p1], None, None);
        assert_eq!(plan.uncovered, vec!["a".to_string()]);
        assert_eq!(plan.shape_conflicts, vec!["a".to_string()]);
        for asn in &plan.assignments {
            assert!(asn.assigned_tensor_names.is_empty());
        }
    }

    #[test]
    fn requested_shape_mismatch_drops_tensor() {
        // Receiver expects [2, 50], peer offers [4, 25]: same byte_len,
        // wrong layout. Drop it.
        let p0 = make_peer_shapes("p0", &[("a", 100, &[4, 25])]);
        let need = vec![entry_with_shape("a", 100, &[2, 50])];
        let plan = compute_transfer_plan(&need, &[p0], None, None);
        assert_eq!(plan.uncovered, vec!["a".to_string()]);
        assert_eq!(plan.shape_conflicts, vec!["a".to_string()]);
    }

    #[test]
    fn unspecified_shape_skips_validation() {
        // Peer advertises a shape, requester leaves it unspecified (None):
        // the transfer proceeds. Mixed specified/unspecified is fine.
        let p0 = make_peer_shapes("p0", &[("a", 100, &[2, 50])]);
        let need = vec![entry("a", 100)]; // entry() leaves shape None
        let plan = compute_transfer_plan(&need, &[p0], None, None);
        assert!(plan.shape_conflicts.is_empty());
        assert_eq!(
            plan.assignments[0].assigned_tensor_names,
            vec!["a".to_string()]
        );
    }

    #[test]
    fn matching_shapes_across_peers_transfer() {
        let p0 = make_peer_shapes("p0", &[("a", 100, &[2, 50])]);
        let p1 = make_peer_shapes("p1", &[("a", 100, &[2, 50])]);
        let need = vec![entry_with_shape("a", 100, &[2, 50])];
        let plan = compute_transfer_plan(&need, &[p0, p1], None, None);
        assert!(plan.shape_conflicts.is_empty());
        assert!(plan.uncovered.is_empty());
    }

    #[test]
    fn max_peers_limits_peer_count() {
        let tensors = &[("a", 100), ("b", 80), ("c", 60)];
        let peers = vec![
            make_peer("p0", tensors),
            make_peer("p1", tensors),
            make_peer("p2", tensors),
        ];
        let plan = compute_transfer_plan(&requested(tensors), &peers, Some(2), None);
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
        let plan = compute_transfer_plan(&requested(tensors), &peers, None, None);
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
        let p1 = compute_transfer_plan(&requested(tensors), &peers, None, None);
        let p2 = compute_transfer_plan(&requested(tensors), &peers, None, None);
        let collect = |t: &TransferPlan| -> Vec<Vec<String>> {
            t.assignments
                .iter()
                .map(|a| a.assigned_tensor_names.clone())
                .collect()
        };
        assert_eq!(collect(&p1), collect(&p2));
    }

    // ----------------------------------------------------------------------
    // Scoring integration
    // ----------------------------------------------------------------------

    use crate::p2p::informer::test_informers::StaticScoreInformer;
    use crate::p2p::informer::{Informer, InformerRegistry};
    use std::sync::Arc;

    #[test]
    fn higher_score_wins_when_load_is_tied() {
        // Both peers own the same single tensor — without scoring, the
        // worker_id tiebreak picks "p0". With scoring that ranks p1
        // higher, p1 should win instead.
        let tensors = &[("a", 100)];
        let peers = vec![make_peer("p0", tensors), make_peer("p1", tensors)];
        let scorer = StaticScoreInformer::new("test");
        scorer.set("p0", Some(1.0));
        scorer.set("p1", Some(10.0));
        let reg = InformerRegistry::new(vec![(Arc::clone(&scorer) as Arc<dyn Informer>, 1.0)]);
        let ctx = super::ScoringContext {
            caller_worker_id: "target",
            caller_labels: &std::collections::HashMap::new(),
            registry: &reg,
        };

        let plan = compute_transfer_plan(&requested(tensors), &peers, None, Some(ctx));
        let winner: Vec<&str> = plan
            .assignments
            .iter()
            .filter(|a| !a.assigned_tensor_names.is_empty())
            .map(|a| a.worker_id.as_str())
            .collect();
        assert_eq!(winner, vec!["p1"], "higher-scored peer should win the tie");
    }

    #[test]
    fn hard_excluded_peer_loses_candidacy_entirely() {
        // p0 is hard-excluded (informer returns None). p0 was the only
        // owner of "a" — so "a" becomes uncovered. p1 still serves "b".
        let p0 = make_peer("p0", &[("a", 100)]);
        let p1 = make_peer("p1", &[("b", 80)]);
        let scorer = StaticScoreInformer::new("maint");
        scorer.set("p0", None); // peer in maintenance
        scorer.set("p1", Some(0.0));
        let reg = InformerRegistry::new(vec![(Arc::clone(&scorer) as Arc<dyn Informer>, 1.0)]);
        let ctx = super::ScoringContext {
            caller_worker_id: "target",
            caller_labels: &std::collections::HashMap::new(),
            registry: &reg,
        };

        let plan = compute_transfer_plan(
            &requested(&[("a", 100), ("b", 80)]),
            &[p0, p1],
            None,
            Some(ctx),
        );

        // p0's only tensor is uncovered.
        assert_eq!(plan.uncovered, vec!["a".to_string()]);
        // p1 is still assigned "b".
        let assigned: Vec<(String, Vec<String>)> = plan
            .assignments
            .iter()
            .map(|a| (a.worker_id.clone(), a.assigned_tensor_names.clone()))
            .collect();
        assert!(
            assigned.contains(&("p1".to_string(), vec!["b".to_string()])),
            "p1 should still serve b: got {assigned:?}"
        );
        // p0 is still listed in the plan but with empty assignments.
        let p0_entry = assigned
            .iter()
            .find(|(w, _)| w == "p0")
            .expect("p0 should still appear in the plan");
        assert!(p0_entry.1.is_empty());
    }

    #[test]
    fn scoring_does_not_override_load_balance() {
        // Two peers, both own all 4 tensors. p0 has a much higher score,
        // but the planner must still load-balance bytes — score is a
        // tertiary tiebreak only.
        let tensors = &[("a", 100), ("b", 80), ("c", 60), ("d", 40)];
        let peers = vec![make_peer("p0", tensors), make_peer("p1", tensors)];
        let scorer = StaticScoreInformer::new("test");
        scorer.set("p0", Some(1000.0));
        scorer.set("p1", Some(0.0));
        let reg = InformerRegistry::new(vec![(Arc::clone(&scorer) as Arc<dyn Informer>, 1.0)]);
        let ctx = super::ScoringContext {
            caller_worker_id: "target",
            caller_labels: &std::collections::HashMap::new(),
            registry: &reg,
        };

        let plan = compute_transfer_plan(&requested(tensors), &peers, None, Some(ctx));
        let p0_bytes = assigned_bytes(&plan.assignments[0], tensors);
        let p1_bytes = assigned_bytes(&plan.assignments[1], tensors);
        assert_eq!(p0_bytes + p1_bytes, 280);
        // p0 wins ties (higher score), but the inner loop still balances
        // overall bytes. With heaviest-first, p0 takes 100, p1 takes 80,
        // p0 would tie at 100/80 → p0's load is higher, so p1 gets 60
        // (its load is 80 < p0's 100), then p0 takes 40 to land 140/140.
        assert_eq!(p0_bytes, 140);
        assert_eq!(p1_bytes, 140);
    }

    #[test]
    fn peer_rank_cmp_lex_order_is_load_score_id() {
        let lo_load = super::PeerRank {
            load_bytes: 100,
            score: 0.0,
            worker_id: "z",
        };
        let hi_load = super::PeerRank {
            load_bytes: 200,
            score: 100.0,
            worker_id: "a",
        };
        // Lower load wins despite worse score and later id.
        assert_eq!(lo_load.cmp(&hi_load), std::cmp::Ordering::Less);

        // Equal load: higher score wins.
        let same_load_a = super::PeerRank {
            load_bytes: 100,
            score: 5.0,
            worker_id: "z",
        };
        let same_load_b = super::PeerRank {
            load_bytes: 100,
            score: 1.0,
            worker_id: "a",
        };
        assert_eq!(same_load_a.cmp(&same_load_b), std::cmp::Ordering::Less);

        // Equal load and score: lower worker_id wins.
        let id_a = super::PeerRank {
            load_bytes: 100,
            score: 5.0,
            worker_id: "a",
        };
        let id_z = super::PeerRank {
            load_bytes: 100,
            score: 5.0,
            worker_id: "z",
        };
        assert_eq!(id_a.cmp(&id_z), std::cmp::Ordering::Less);

        // NaN score is treated as Equal, never panics.
        let nan_score = super::PeerRank {
            load_bytes: 100,
            score: f64::NAN,
            worker_id: "a",
        };
        let ok_score = super::PeerRank {
            load_bytes: 100,
            score: 1.0,
            worker_id: "z",
        };
        // load equal, score comparison falls back to Equal, then "a" < "z"
        assert_eq!(nan_score.cmp(&ok_score), std::cmp::Ordering::Less);
    }
}
