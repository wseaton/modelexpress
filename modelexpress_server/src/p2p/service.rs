// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P Metadata Service implementation for storing and retrieving NIXL/RDMA metadata.
//!
//! Metadata is keyed by mx_source_id, a 16-char hex hash of SourceIdentity.
//! Clients send the full SourceIdentity; the server computes and returns the hash.

use crate::p2p::backend::{
    SourceInstanceInfo, TensorCatalogEntryRecord, TensorCatalogRecord, WorkerRecord,
};
use crate::p2p::informer::InformerRegistry;
use crate::p2p::planner::{
    CatalogEntry, PeerCandidate, ScoringContext, compute_transfer_plan,
    synthetic_catalog_from_worker,
};
use crate::p2p::source_identity::{compute_mx_source_id, validate_identity};
use crate::p2p::state::P2pStateManager;
use modelexpress_common::grpc::p2p::{
    AdvertiseTensorCatalogRequest, AdvertiseTensorCatalogResponse, ComputeTransferPlanRequest,
    ComputeTransferPlanResponse, GetMetadataRequest, GetMetadataResponse, ListSourcesRequest,
    ListSourcesResponse, PeerAssignment, PlanDiagnostics, PublishMetadataRequest,
    PublishMetadataResponse, SourceInstanceRef, SourceStatus, UpdateStatusRequest,
    UpdateStatusResponse, WorkerMetadata, p2p_service_server::P2pService,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};

/// P2P Service implementation
pub struct P2pServiceImpl {
    state: Arc<P2pStateManager>,
    /// External-data informers used to rank peers during transfer planning.
    /// Empty registry = no informer signal applied; planner uses
    /// load+worker_id ordering only.
    informers: Arc<InformerRegistry>,
}

impl P2pServiceImpl {
    /// Create a new P2P service with no informers (planner uses
    /// load+worker_id ordering only).
    pub fn new(state: Arc<P2pStateManager>) -> Self {
        Self {
            state,
            informers: InformerRegistry::empty(),
        }
    }

    /// Create a new P2P service backed by an informer registry. The
    /// caller is responsible for having already called
    /// [`InformerRegistry::start`] before the first plan request.
    pub fn with_informers(state: Arc<P2pStateManager>, informers: Arc<InformerRegistry>) -> Self {
        Self { state, informers }
    }
}

#[tonic::async_trait]
impl P2pService for P2pServiceImpl {
    async fn publish_metadata(
        &self,
        request: Request<PublishMetadataRequest>,
    ) -> Result<Response<PublishMetadataResponse>, Status> {
        let req = request.into_inner();

        let identity = match req.identity {
            Some(id) => id,
            None => {
                return Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: "identity is required".to_string(),
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }));
            }
        };

        if let Err(e) = validate_identity(&identity) {
            return Ok(Response::new(PublishMetadataResponse {
                success: false,
                message: e,
                mx_source_id: String::new(),
                worker_id: String::new(),
            }));
        }

        if req.worker_id.is_empty() {
            return Ok(Response::new(PublishMetadataResponse {
                success: false,
                message: "worker_id is required".to_string(),
                mx_source_id: String::new(),
                worker_id: String::new(),
            }));
        }

        let worker = match req.worker {
            Some(w) => w,
            None => {
                return Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: "worker is required".to_string(),
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }));
            }
        };

        let source_id = compute_mx_source_id(&identity);
        let worker_id = req.worker_id.clone();
        let model_name = identity.model_name.clone();
        let worker_rank = worker.worker_rank;
        let tensor_count = worker.tensors.len();

        match self
            .state
            .publish_metadata(&identity, &worker_id, worker)
            .await
        {
            Ok(()) => {
                info!(
                    "PublishMetadata: model='{}' source_id={} worker_id={} worker_rank={} tensors={}",
                    model_name, source_id, worker_id, worker_rank, tensor_count
                );
                Ok(Response::new(PublishMetadataResponse {
                    success: true,
                    message: format!(
                        "Published metadata for '{}' (source_id={}, worker_id={}, worker_rank={}, {} tensors)",
                        model_name, source_id, worker_id, worker_rank, tensor_count
                    ),
                    mx_source_id: source_id,
                    worker_id,
                }))
            }
            Err(e) => {
                error!("Failed to publish metadata: {}", e);
                Ok(Response::new(PublishMetadataResponse {
                    success: false,
                    message: format!("Failed to publish metadata: {e}"),
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }))
            }
        }
    }

    async fn list_sources(
        &self,
        request: Request<ListSourcesRequest>,
    ) -> Result<Response<ListSourcesResponse>, Status> {
        let req = request.into_inner();

        // Resolve optional source_id filter
        let source_id_filter: Option<String> = req.identity.as_ref().and_then(|id| {
            if id.model_name.is_empty() {
                None
            } else {
                Some(compute_mx_source_id(id))
            }
        });

        // Convert raw proto i32 to typed enum — None means no filter
        let status_filter = req
            .status_filter
            .and_then(|s| SourceStatus::try_from(s).ok());

        let workers: Vec<SourceInstanceInfo> = match self
            .state
            .list_workers(source_id_filter, status_filter)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to list workers: {}", e);
                return Ok(Response::new(ListSourcesResponse {
                    instances: Vec::new(),
                }));
            }
        };

        let refs: Vec<SourceInstanceRef> = workers
            .into_iter()
            .map(|info| SourceInstanceRef {
                mx_source_id: info.source_id,
                worker_id: info.worker_id,
                model_name: info.model_name,
                worker_rank: info.worker_rank,
            })
            .collect();

        debug!("ListSources: returning {} instances", refs.len());

        Ok(Response::new(ListSourcesResponse { instances: refs }))
    }

    async fn get_metadata(
        &self,
        request: Request<GetMetadataRequest>,
    ) -> Result<Response<GetMetadataResponse>, Status> {
        let req = request.into_inner();

        if req.mx_source_id.is_empty() || req.worker_id.is_empty() {
            return Ok(Response::new(GetMetadataResponse {
                found: false,
                worker: None,
                mx_source_id: String::new(),
                worker_id: String::new(),
            }));
        }

        match self
            .state
            .get_metadata(&req.mx_source_id, &req.worker_id)
            .await
        {
            Ok(Some(record)) => {
                // Each worker_id maps to exactly one worker record; take the first.
                let worker = record.workers.into_iter().next().map(WorkerMetadata::from);
                let found = worker.is_some();
                info!(
                    "GetMetadata '{}' (source_id={}, worker_id={}): {} tensors",
                    record.model_name,
                    req.mx_source_id,
                    req.worker_id,
                    worker.as_ref().map_or(0, |w| w.tensors.len()),
                );
                Ok(Response::new(GetMetadataResponse {
                    found,
                    worker,
                    mx_source_id: req.mx_source_id,
                    worker_id: req.worker_id,
                }))
            }
            Ok(None) => {
                info!(
                    "No metadata found for source_id={} worker_id={}",
                    req.mx_source_id, req.worker_id
                );
                Ok(Response::new(GetMetadataResponse {
                    found: false,
                    worker: None,
                    mx_source_id: req.mx_source_id,
                    worker_id: req.worker_id,
                }))
            }
            Err(e) => {
                error!("Failed to get metadata: {}", e);
                Ok(Response::new(GetMetadataResponse {
                    found: false,
                    worker: None,
                    mx_source_id: String::new(),
                    worker_id: String::new(),
                }))
            }
        }
    }

    async fn update_status(
        &self,
        request: Request<UpdateStatusRequest>,
    ) -> Result<Response<UpdateStatusResponse>, Status> {
        let req = request.into_inner();

        if req.mx_source_id.is_empty() {
            return Ok(Response::new(UpdateStatusResponse {
                success: false,
                message: "mx_source_id is required".to_string(),
            }));
        }

        if req.worker_id.is_empty() {
            return Ok(Response::new(UpdateStatusResponse {
                success: false,
                message: "worker_id is required".to_string(),
            }));
        }

        let status = match SourceStatus::try_from(req.status) {
            Ok(s) => s,
            Err(_) => {
                return Ok(Response::new(UpdateStatusResponse {
                    success: false,
                    message: format!("invalid status value: {}", req.status),
                }));
            }
        };

        match self
            .state
            .update_worker_status(&req.mx_source_id, &req.worker_id, req.worker_rank, status)
            .await
        {
            Ok(()) => Ok(Response::new(UpdateStatusResponse {
                success: true,
                message: format!(
                    "Updated status for source '{}' worker_id '{}' rank {}",
                    req.mx_source_id, req.worker_id, req.worker_rank
                ),
            })),
            Err(e) => {
                error!("Failed to update status: {}", e);
                Ok(Response::new(UpdateStatusResponse {
                    success: false,
                    message: format!("Failed to update status: {e}"),
                }))
            }
        }
    }

    async fn compute_transfer_plan(
        &self,
        request: Request<ComputeTransferPlanRequest>,
    ) -> Result<Response<ComputeTransferPlanResponse>, Status> {
        let req = request.into_inner();

        let identity = match req.identity {
            Some(id) => id,
            None => {
                return Ok(Response::new(ComputeTransferPlanResponse {
                    peers: Vec::new(),
                    uncovered_tensor_names: Vec::new(),
                    diagnostics: Some(PlanDiagnostics {
                        candidates_total: 0,
                        candidates_eligible: 0,
                        note: "identity is required".to_string(),
                    }),
                }));
            }
        };

        if let Err(e) = validate_identity(&identity) {
            return Ok(Response::new(ComputeTransferPlanResponse {
                peers: Vec::new(),
                uncovered_tensor_names: Vec::new(),
                diagnostics: Some(PlanDiagnostics {
                    candidates_total: 0,
                    candidates_eligible: 0,
                    note: e,
                }),
            }));
        }

        let source_id = compute_mx_source_id(&identity);

        let workers: Vec<SourceInstanceInfo> = match self
            .state
            .list_workers(Some(source_id.clone()), Some(SourceStatus::Ready))
            .await
        {
            Ok(v) => v,
            Err(e) => {
                error!("ComputeTransferPlan: failed to list workers: {}", e);
                return Ok(Response::new(ComputeTransferPlanResponse {
                    peers: Vec::new(),
                    uncovered_tensor_names: Vec::new(),
                    diagnostics: Some(PlanDiagnostics {
                        candidates_total: 0,
                        candidates_eligible: 0,
                        note: format!("backend error: {e}"),
                    }),
                }));
            }
        };

        let candidates_total = workers.len() as u32;

        let eligible: Vec<SourceInstanceInfo> = workers
            .into_iter()
            .filter(|w| {
                w.worker_rank == req.requester_worker_rank && w.worker_id != req.requester_worker_id
            })
            .collect();

        let candidates_eligible = eligible.len() as u32;

        if eligible.is_empty() {
            info!(
                "ComputeTransferPlan: no eligible peers for source_id={} rank={}",
                source_id, req.requester_worker_rank
            );
            return Ok(Response::new(ComputeTransferPlanResponse {
                peers: Vec::new(),
                uncovered_tensor_names: Vec::new(),
                diagnostics: Some(PlanDiagnostics {
                    candidates_total,
                    candidates_eligible: 0,
                    note: "no eligible peers found".to_string(),
                }),
            }));
        }

        let mut peer_candidates = Vec::with_capacity(eligible.len());
        for info in &eligible {
            match self
                .state
                .get_metadata(&info.source_id, &info.worker_id)
                .await
            {
                Ok(Some(record)) => {
                    if let Some(worker) = record.workers.into_iter().next() {
                        // Backward-compat: peers that didn't call
                        // AdvertiseTensorCatalog (or whose catalog hasn't
                        // landed in the backend yet) get a synthetic
                        // catalog from their PublishMetadata tensors.
                        let catalog = catalog_entries_for_worker(&worker);
                        peer_candidates.push(PeerCandidate {
                            source_id: info.source_id.clone(),
                            worker_id: info.worker_id.clone(),
                            worker,
                            catalog,
                        });
                    }
                }
                Ok(None) => {
                    debug!(
                        "ComputeTransferPlan: metadata not found for worker_id={}",
                        info.worker_id
                    );
                }
                Err(e) => {
                    error!(
                        "ComputeTransferPlan: failed to get metadata for worker_id={}: {}",
                        info.worker_id, e
                    );
                }
            }
        }

        // The requester's own worker record, looked up once and reused for
        // both the implied need-set (empty `requested_tensor_names`) and the
        // caller-relative scoring labels below. The receiver advertises this
        // via PublishMetadata(INITIALIZING) + AdvertiseTensorCatalog before
        // planning; absent that, it's None and we fall back to the union.
        //
        // Only fetched when something actually needs it: an empty request
        // (implied need-set) or active informers (caller labels). The
        // explicit-names path with no informers skips the round-trip.
        let needs_requester_record =
            req.requested_tensor_names.is_empty() || !self.informers.describe().is_empty();
        let requester_record = if needs_requester_record {
            self.state
                .get_metadata(&source_id, &req.requester_worker_id)
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        let requester_worker = requester_record.as_ref().and_then(|rec| {
            rec.workers
                .iter()
                .find(|w| w.worker_rank == req.requester_worker_rank)
                .or_else(|| rec.workers.first())
        });

        // Resolve the requested tensor set. Per the proto contract:
        //   - explicit `requested_tensor_names` wins
        //   - empty list uses the requester's own advertised catalog as the
        //     implied need-set
        //   - union of all peer catalogs is the last resort when the
        //     requester advertised nothing (noted in diagnostics)
        let mut note_parts: Vec<String> = Vec::new();
        let requested: Vec<CatalogEntry> = if !req.requested_tensor_names.is_empty() {
            // Build entries from the union catalog so we have byte_len
            // and dtype for sorting and dtype-conflict detection. If a
            // requested name appears in multiple peers, we use the first
            // peer's entry; mismatches are caught inside the planner.
            let mut by_name: std::collections::HashMap<&str, &CatalogEntry> =
                std::collections::HashMap::new();
            for peer in &peer_candidates {
                for e in &peer.catalog {
                    by_name.entry(e.name.as_str()).or_insert(e);
                }
            }
            let mut missing_from_catalog = 0usize;
            let requested = req
                .requested_tensor_names
                .iter()
                .map(|name| match by_name.get(name.as_str()) {
                    Some(e) => (*e).clone(),
                    None => {
                        missing_from_catalog = missing_from_catalog.saturating_add(1);
                        CatalogEntry {
                            name: name.clone(),
                            byte_len: 0,
                            dtype: String::new(),
                            shape: None,
                        }
                    }
                })
                .collect();
            if missing_from_catalog > 0 {
                note_parts.push(format!(
                    "{} requested tensor(s) absent from all peer catalogs",
                    missing_from_catalog
                ));
            }
            requested
        } else {
            let requester_catalog = requester_worker
                .map(catalog_entries_for_worker)
                .filter(|c| !c.is_empty());
            match requester_catalog {
                Some(catalog) => {
                    note_parts.push(format!(
                        "using requester's advertised catalog ({} tensor(s))",
                        catalog.len()
                    ));
                    catalog
                }
                None => {
                    note_parts.push(
                        "requester has no advertised catalog; \
                         using union of peer catalogs (last resort)"
                            .into(),
                    );
                    let mut seen: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    let mut union = Vec::new();
                    for peer in &peer_candidates {
                        for e in &peer.catalog {
                            if seen.insert(e.name.clone()) {
                                union.push(e.clone());
                            }
                        }
                    }
                    union
                }
            }
        };

        // Build a scoring context when the registry has informers wired
        // up. An empty registry behaves like no scoring (planner uses
        // load+id ordering only), and we skip the caller-label lookup
        // entirely in that case.
        //
        // The caller's own labels (published earlier via PublishMetadata
        // under the same source_id) let label-only informers do
        // caller-relative comparisons. Absent or unpublished => empty map.
        let caller_labels: std::collections::HashMap<String, String> =
            if self.informers.describe().is_empty() {
                std::collections::HashMap::new()
            } else {
                requester_worker
                    .map(|worker| worker.labels.clone())
                    .unwrap_or_default()
            };

        let scoring = if self.informers.describe().is_empty() {
            None
        } else {
            Some(ScoringContext {
                caller_worker_id: &req.requester_worker_id,
                caller_labels: &caller_labels,
                registry: &self.informers,
            })
        };

        let plan = compute_transfer_plan(&requested, &peer_candidates, req.max_peers, scoring);

        if !plan.dtype_conflicts.is_empty() {
            note_parts.push(format!(
                "dtype conflicts dropped {} tensor(s)",
                plan.dtype_conflicts.len()
            ));
        }
        if !plan.shape_conflicts.is_empty() {
            note_parts.push(format!(
                "shape conflicts dropped {} tensor(s)",
                plan.shape_conflicts.len()
            ));
        }
        if !plan.uncovered.is_empty() {
            note_parts.push(format!("{} tensor(s) uncovered", plan.uncovered.len()));
        }

        let peers: Vec<PeerAssignment> = plan
            .assignments
            .into_iter()
            .map(|a| {
                let backend_md = match &a.worker.backend_metadata {
                    crate::p2p::backend::BackendMetadataRecord::Nixl(data) => data.clone(),
                    _ => Vec::new(),
                };
                PeerAssignment {
                    mx_source_id: a.source_id,
                    worker_id: a.worker_id,
                    agent_name: a.worker.agent_name,
                    metadata_endpoint: a.worker.metadata_endpoint,
                    nixl_metadata: backend_md,
                    tensors: a
                        .worker
                        .tensors
                        .into_iter()
                        .map(modelexpress_common::grpc::p2p::TensorDescriptor::from)
                        .collect(),
                    assigned_tensor_names: a.assigned_tensor_names,
                    worker_grpc_endpoint: a.worker.worker_grpc_endpoint,
                }
            })
            .collect();

        info!(
            "ComputeTransferPlan: source_id={} rank={} peers={} (of {} eligible, {} total) uncovered={}",
            source_id,
            req.requester_worker_rank,
            peers.len(),
            candidates_eligible,
            candidates_total,
            plan.uncovered.len(),
        );

        Ok(Response::new(ComputeTransferPlanResponse {
            peers,
            diagnostics: Some(PlanDiagnostics {
                candidates_total,
                candidates_eligible,
                note: note_parts.join("; "),
            }),
            uncovered_tensor_names: plan.uncovered,
        }))
    }

    async fn advertise_tensor_catalog(
        &self,
        request: Request<AdvertiseTensorCatalogRequest>,
    ) -> Result<Response<AdvertiseTensorCatalogResponse>, Status> {
        let req = request.into_inner();

        let identity = match req.identity {
            Some(id) => id,
            None => {
                return Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: false,
                    message: "identity is required".to_string(),
                    entries_accepted: 0,
                    total_bytes: 0,
                    generation: 0,
                }));
            }
        };

        if let Err(e) = validate_identity(&identity) {
            return Ok(Response::new(AdvertiseTensorCatalogResponse {
                success: false,
                message: e,
                entries_accepted: 0,
                total_bytes: 0,
                generation: 0,
            }));
        }

        if req.worker_id.is_empty() {
            return Ok(Response::new(AdvertiseTensorCatalogResponse {
                success: false,
                message: "worker_id is required".to_string(),
                entries_accepted: 0,
                total_bytes: 0,
                generation: 0,
            }));
        }

        let source_id = compute_mx_source_id(&identity);
        let existing = match self.state.get_metadata(&source_id, &req.worker_id).await {
            Ok(Some(record)) => record,
            Ok(None) => {
                return Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: false,
                    message: format!(
                        "worker '{}' has not published metadata for source_id={}",
                        req.worker_id, source_id
                    ),
                    entries_accepted: 0,
                    total_bytes: 0,
                    generation: 0,
                }));
            }
            Err(e) => {
                error!(
                    "AdvertiseTensorCatalog: failed to read metadata for worker_id={}: {}",
                    req.worker_id, e
                );
                return Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: false,
                    message: format!("backend error: {e}"),
                    entries_accepted: 0,
                    total_bytes: 0,
                    generation: 0,
                }));
            }
        };

        let Some(worker) = existing
            .workers
            .iter()
            .find(|worker| worker.worker_rank == req.worker_rank)
        else {
            return Ok(Response::new(AdvertiseTensorCatalogResponse {
                success: false,
                message: format!(
                    "worker '{}' rank {} was not found for source_id={}",
                    req.worker_id, req.worker_rank, source_id
                ),
                entries_accepted: 0,
                total_bytes: 0,
                generation: 0,
            }));
        };

        if let Some(catalog) = &worker.tensor_catalog
            && req.generation <= catalog.generation
        {
            return Ok(Response::new(AdvertiseTensorCatalogResponse {
                success: false,
                message: format!(
                    "stale catalog generation {}; current generation is {}",
                    req.generation, catalog.generation
                ),
                entries_accepted: 0,
                total_bytes: catalog.total_bytes(),
                generation: catalog.generation,
            }));
        }

        let mut seen = std::collections::HashSet::new();
        let mut entries = Vec::with_capacity(req.entries.len());
        let mut total_bytes = 0_u64;
        for entry in req.entries {
            if entry.name.is_empty() {
                return Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: false,
                    message: "catalog entry name is required".to_string(),
                    entries_accepted: 0,
                    total_bytes: 0,
                    generation: 0,
                }));
            }
            if !seen.insert(entry.name.clone()) {
                return Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: false,
                    message: format!("duplicate catalog entry '{}'", entry.name),
                    entries_accepted: 0,
                    total_bytes: 0,
                    generation: 0,
                }));
            }
            total_bytes = total_bytes.saturating_add(entry.byte_len);
            entries.push(TensorCatalogEntryRecord {
                name: entry.name,
                byte_len: entry.byte_len,
                dtype: entry.dtype,
                shape: entry.shape,
            });
        }

        let entries_accepted = match u32::try_from(entries.len()) {
            Ok(count) => count,
            Err(_) => u32::MAX,
        };
        let catalog = TensorCatalogRecord {
            generation: req.generation,
            entries,
        };

        match self
            .state
            .put_tensor_catalog(&source_id, &req.worker_id, req.worker_rank, catalog)
            .await
        {
            Ok(()) => {
                info!(
                    "AdvertiseTensorCatalog: source_id={} worker_id={} rank={} entries={} bytes={} gen={}",
                    source_id,
                    req.worker_id,
                    req.worker_rank,
                    entries_accepted,
                    total_bytes,
                    req.generation,
                );
                Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: true,
                    message: "catalog accepted".into(),
                    entries_accepted,
                    total_bytes,
                    generation: req.generation,
                }))
            }
            Err(e) => {
                error!(
                    "AdvertiseTensorCatalog: failed to persist catalog for worker_id={}: {}",
                    req.worker_id, e
                );
                Ok(Response::new(AdvertiseTensorCatalogResponse {
                    success: false,
                    message: format!("failed to persist catalog: {e}"),
                    entries_accepted: 0,
                    total_bytes: 0,
                    generation: req.generation,
                }))
            }
        }
    }
}

fn catalog_entries_for_worker(worker: &WorkerRecord) -> Vec<CatalogEntry> {
    worker
        .tensor_catalog
        .as_ref()
        .map(|catalog| {
            catalog
                .entries
                .iter()
                .map(|entry| CatalogEntry {
                    name: entry.name.clone(),
                    byte_len: entry.byte_len,
                    dtype: entry.dtype.clone(),
                    // Empty shape on the wire/record means "unspecified".
                    shape: if entry.shape.is_empty() {
                        None
                    } else {
                        Some(entry.shape.clone())
                    },
                })
                .collect()
        })
        .unwrap_or_else(|| synthetic_catalog_from_worker(worker))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::p2p::backend::{
        BackendMetadataRecord, MockMetadataBackend, ModelMetadataRecord, TensorCatalogEntryRecord,
        TensorCatalogRecord, TensorRecord, WorkerRecord,
    };
    use crate::p2p::state::P2pStateManager;
    use modelexpress_common::grpc::p2p::{
        MxSourceType, SourceIdentity, SourceStatus, TensorCatalogEntry,
    };

    fn make_service(mock: MockMetadataBackend) -> P2pServiceImpl {
        P2pServiceImpl::new(Arc::new(P2pStateManager::with_backend(Arc::new(mock))))
    }

    fn test_identity() -> SourceIdentity {
        SourceIdentity {
            mx_version: "0.3.0".to_string(),
            mx_source_type: MxSourceType::Weights as i32,
            model_name: "my-model".to_string(),
            backend_framework: 1,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 0,
            dtype: "bfloat16".to_string(),
            quantization: String::new(),
            extra_parameters: Default::default(),
            revision: String::new(),
        }
    }

    // ── publish_metadata ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_publish_metadata_missing_identity() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: None,
                worker: None,
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.mx_source_id.is_empty());
    }

    #[tokio::test]
    async fn test_publish_metadata_empty_model_name() {
        let svc = make_service(MockMetadataBackend::new());
        let mut id = test_identity();
        id.model_name = String::new();
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(id),
                worker: None,
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_publish_metadata_missing_worker_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: None,
                worker_id: String::new(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("worker_id"));
    }

    #[tokio::test]
    async fn test_publish_metadata_success() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _, _| Ok(()));

        let svc = make_service(mock);
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: Some(WorkerMetadata {
                    worker_rank: 0,
                    backend_metadata: Some(
                        modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata::NixlMetadata(vec![1, 2, 3]),
                    ),
                    tensors: vec![],
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                    ..Default::default()
                }),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.success);
        assert!(!resp.mx_source_id.is_empty());
        assert_eq!(resp.mx_source_id.len(), 16);
        assert_eq!(resp.worker_id, "worker-uuid-1");
    }

    #[tokio::test]
    async fn test_publish_metadata_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_publish_metadata()
            .once()
            .returning(|_, _, _| Err("storage unavailable".into()));

        let svc = make_service(mock);
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: Some(WorkerMetadata {
                    worker_rank: 0,
                    backend_metadata: None,
                    tensors: vec![],
                    status: SourceStatus::Initializing as i32,
                    updated_at: 0,
                    ..Default::default()
                }),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("storage unavailable"));
    }

    // ── get_metadata ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_metadata_empty_source_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: String::new(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
    }

    #[tokio::test]
    async fn test_get_metadata_found() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|source_id, worker_id| {
                Ok(Some(ModelMetadataRecord {
                    source_id: source_id.to_string(),
                    worker_id: worker_id.to_string(),
                    model_name: "my-model".to_string(),
                    workers: vec![WorkerRecord {
                        worker_rank: 0,
                        backend_metadata: BackendMetadataRecord::None,
                        tensors: vec![],
                        status: SourceStatus::Ready as i32,
                        updated_at: 1234567890000,
                        metadata_endpoint: String::new(),
                        agent_name: String::new(),
                        worker_grpc_endpoint: String::new(),
                        labels: std::collections::HashMap::new(),
                        tensor_catalog: None,
                    }],
                    published_at: 1234567890,
                }))
            });

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.found);
        assert!(resp.worker.is_some());
        assert_eq!(
            resp.worker.expect("worker should be present").status,
            SourceStatus::Ready as i32
        );
        assert_eq!(resp.mx_source_id, "abc123def456abcd");
        assert_eq!(resp.worker_id, "worker-uuid-1");
    }

    #[tokio::test]
    async fn test_get_metadata_not_found() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata().once().returning(|_, _| Ok(None));

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
        assert_eq!(resp.mx_source_id, "abc123def456abcd");
    }

    // ── update_status ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_update_status_invalid_status_value() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                status: 99,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("99"));
    }

    #[tokio::test]
    async fn test_update_status_empty_source_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: String::new(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_update_status_empty_worker_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: String::new(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
    }

    #[tokio::test]
    async fn test_update_status_success() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _, _| Ok(()));

        let svc = make_service(mock);
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 3,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.success);
    }

    // ── publish_metadata (missing worker) ────────────────────────────────

    #[tokio::test]
    async fn test_publish_metadata_missing_worker() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .publish_metadata(Request::new(PublishMetadataRequest {
                identity: Some(test_identity()),
                worker: None,
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("worker is required"));
    }

    // ── list_sources ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_sources_returns_instances() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning(|_, _| {
            Ok(vec![
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                },
                SourceInstanceInfo {
                    source_id: "abc123def456abcd".to_string(),
                    worker_id: "w2".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 1,
                    status: SourceStatus::Ready as i32,
                    updated_at: 1234567890000,
                },
            ])
        });

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(test_identity()),
                status_filter: Some(SourceStatus::Ready as i32),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert_eq!(resp.instances.len(), 2);
        assert_eq!(resp.instances[0].worker_id, "w1");
        assert_eq!(resp.instances[0].worker_rank, 0);
        assert_eq!(resp.instances[1].worker_id, "w2");
        assert_eq!(resp.instances[1].worker_rank, 1);
    }

    #[tokio::test]
    async fn test_list_sources_no_identity() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Ok(vec![]));

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: None,
                status_filter: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.instances.is_empty());
    }

    #[tokio::test]
    async fn test_list_sources_backend_error_returns_empty() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Err("backend down".into()));

        let svc = make_service(mock);
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(test_identity()),
                status_filter: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.instances.is_empty());
    }

    #[tokio::test]
    async fn test_list_sources_empty_model_name_no_filter() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .withf(|source_id, _| source_id.is_none())
            .once()
            .returning(|_, _| Ok(vec![]));

        let svc = make_service(mock);
        let mut id = test_identity();
        id.model_name = String::new();
        let resp = svc
            .list_sources(Request::new(ListSourcesRequest {
                identity: Some(id),
                status_filter: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.instances.is_empty());
    }

    // ── get_metadata (additional) ───────────────────────────────────────────

    #[tokio::test]
    async fn test_get_metadata_empty_worker_id() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: String::new(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
    }

    #[tokio::test]
    async fn test_get_metadata_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|_, _| Err("storage error".into()));

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
        assert!(resp.mx_source_id.is_empty());
    }

    #[tokio::test]
    async fn test_get_metadata_record_with_empty_workers() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(|source_id, worker_id| {
                Ok(Some(ModelMetadataRecord {
                    source_id: source_id.to_string(),
                    worker_id: worker_id.to_string(),
                    model_name: "my-model".to_string(),
                    workers: vec![],
                    published_at: 0,
                }))
            });

        let svc = make_service(mock);
        let resp = svc
            .get_metadata(Request::new(GetMetadataRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.found);
        assert!(resp.worker.is_none());
    }

    // ── update_status (additional) ──────────────────────────────────────────

    #[tokio::test]
    async fn test_update_status_backend_error() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_update_status()
            .once()
            .returning(|_, _, _, _, _| Err("write failed".into()));

        let svc = make_service(mock);
        let resp = svc
            .update_status(Request::new(UpdateStatusRequest {
                mx_source_id: "abc123def456abcd".to_string(),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                status: SourceStatus::Ready as i32,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(!resp.success);
        assert!(resp.message.contains("write failed"));
    }

    // ── compute_transfer_plan ──────────────────────────────────────────────

    fn make_worker_record(rank: u32, tensors: &[(&str, u64)]) -> WorkerRecord {
        WorkerRecord {
            worker_rank: rank,
            backend_metadata: BackendMetadataRecord::Nixl(vec![0xCA, 0xFE]),
            tensors: tensors
                .iter()
                .map(|(name, size)| TensorRecord {
                    name: name.to_string(),
                    addr: 0x1000,
                    size: *size,
                    device_id: 0,
                    dtype: "bfloat16".to_string(),
                })
                .collect(),
            status: SourceStatus::Ready as i32,
            updated_at: 1234567890000,
            metadata_endpoint: "10.0.0.1:12345".to_string(),
            agent_name: "test-agent".to_string(),
            worker_grpc_endpoint: "10.0.0.1:50051".to_string(),
            labels: std::collections::HashMap::new(),
            tensor_catalog: None,
        }
    }

    fn make_model_record(
        source_id: &str,
        worker_id: &str,
        worker: WorkerRecord,
    ) -> ModelMetadataRecord {
        ModelMetadataRecord {
            source_id: source_id.to_string(),
            worker_id: worker_id.to_string(),
            model_name: "my-model".to_string(),
            workers: vec![worker],
            published_at: 0,
        }
    }

    fn make_proto_catalog_entry(name: &str, byte_len: u64) -> TensorCatalogEntry {
        TensorCatalogEntry {
            name: name.to_string(),
            byte_len,
            dtype: "bfloat16".to_string(),
            shape: vec![4, 8],
        }
    }

    // ── advertise_tensor_catalog ───────────────────────────────────────────

    #[tokio::test]
    async fn test_advertise_tensor_catalog_success_persists_entries() {
        let source_id = compute_mx_source_id(&test_identity());
        let worker = make_worker_record(0, &[("legacy", 100)]);

        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata().once().returning({
            let source_id = source_id.clone();
            let worker = worker.clone();
            move |sid, wid| {
                assert_eq!(sid, source_id);
                assert_eq!(wid, "worker-uuid-1");
                Ok(Some(make_model_record(sid, wid, worker.clone())))
            }
        });
        mock.expect_put_tensor_catalog().once().returning({
            let source_id = source_id.clone();
            move |sid, wid, rank, catalog| {
                assert_eq!(sid, source_id);
                assert_eq!(wid, "worker-uuid-1");
                assert_eq!(rank, 0);
                assert_eq!(catalog.generation, 7);
                assert_eq!(catalog.entries.len(), 2);
                assert_eq!(catalog.entries[0].name, "a");
                assert_eq!(catalog.entries[0].shape, vec![4, 8]);
                assert_eq!(catalog.entries[1].byte_len, 20);
                Ok(())
            }
        });

        let svc = make_service(mock);
        let resp = svc
            .advertise_tensor_catalog(Request::new(AdvertiseTensorCatalogRequest {
                identity: Some(test_identity()),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                entries: vec![
                    make_proto_catalog_entry("a", 10),
                    make_proto_catalog_entry("b", 20),
                ],
                generation: 7,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert!(resp.success);
        assert_eq!(resp.entries_accepted, 2);
        assert_eq!(resp.total_bytes, 30);
        assert_eq!(resp.generation, 7);
    }

    #[tokio::test]
    async fn test_advertise_tensor_catalog_rejects_stale_generation() {
        let mut worker = make_worker_record(0, &[("legacy", 100)]);
        worker.tensor_catalog = Some(TensorCatalogRecord {
            generation: 2,
            entries: vec![TensorCatalogEntryRecord {
                name: "existing".to_string(),
                byte_len: 64,
                dtype: "bfloat16".to_string(),
                shape: Vec::new(),
            }],
        });

        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(move |sid, wid| Ok(Some(make_model_record(sid, wid, worker.clone()))));

        let svc = make_service(mock);
        let resp = svc
            .advertise_tensor_catalog(Request::new(AdvertiseTensorCatalogRequest {
                identity: Some(test_identity()),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                entries: vec![make_proto_catalog_entry("newer", 32)],
                generation: 2,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert!(!resp.success);
        assert!(resp.message.contains("stale catalog generation"));
        assert_eq!(resp.generation, 2);
        assert_eq!(resp.total_bytes, 64);
    }

    #[tokio::test]
    async fn test_advertise_tensor_catalog_rejects_duplicate_names() {
        let worker = make_worker_record(0, &[("legacy", 100)]);

        let mut mock = MockMetadataBackend::new();
        mock.expect_get_metadata()
            .once()
            .returning(move |sid, wid| Ok(Some(make_model_record(sid, wid, worker.clone()))));

        let svc = make_service(mock);
        let resp = svc
            .advertise_tensor_catalog(Request::new(AdvertiseTensorCatalogRequest {
                identity: Some(test_identity()),
                worker_id: "worker-uuid-1".to_string(),
                worker_rank: 0,
                entries: vec![
                    make_proto_catalog_entry("dup", 10),
                    make_proto_catalog_entry("dup", 20),
                ],
                generation: 3,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert!(!resp.success);
        assert!(resp.message.contains("duplicate catalog entry"));
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_missing_identity() {
        let svc = make_service(MockMetadataBackend::new());
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: None,
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.peers.is_empty());
        let diag = resp.diagnostics.expect("diagnostics");
        assert!(diag.note.contains("identity is required"));
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_no_eligible_peers() {
        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers()
            .once()
            .returning(|_, _| Ok(vec![]));

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.peers.is_empty());
        let diag = resp.diagnostics.expect("diagnostics");
        assert_eq!(diag.candidates_total, 0);
        assert_eq!(diag.candidates_eligible, 0);
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_excludes_requester() {
        let tensors = &[("a", 100), ("b", 80)];
        let source_id = compute_mx_source_id(&test_identity());

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning({
            let source_id = source_id.clone();
            move |_, _| {
                Ok(vec![SourceInstanceInfo {
                    source_id: source_id.clone(),
                    worker_id: "requester".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 0,
                }])
            }
        });

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();
        assert!(resp.peers.is_empty());
        let diag = resp.diagnostics.expect("diagnostics");
        assert_eq!(diag.candidates_total, 1);
        assert_eq!(diag.candidates_eligible, 0);
        let _ = tensors;
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_two_peers() {
        let tensors = &[("a", 100), ("b", 80), ("c", 60), ("d", 40)];
        let source_id = compute_mx_source_id(&test_identity());

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning({
            let source_id = source_id.clone();
            move |_, _| {
                Ok(vec![
                    SourceInstanceInfo {
                        source_id: source_id.clone(),
                        worker_id: "peer-1".to_string(),
                        model_name: "my-model".to_string(),
                        worker_rank: 0,
                        status: SourceStatus::Ready as i32,
                        updated_at: 0,
                    },
                    SourceInstanceInfo {
                        source_id: source_id.clone(),
                        worker_id: "peer-2".to_string(),
                        model_name: "my-model".to_string(),
                        worker_rank: 0,
                        status: SourceStatus::Ready as i32,
                        updated_at: 0,
                    },
                ])
            }
        });

        // Two peer lookups plus the requester's own catalog lookup (empty
        // requested_tensor_names triggers the implied need-set path).
        mock.expect_get_metadata().times(3).returning({
            move |sid, wid| {
                Ok(Some(ModelMetadataRecord {
                    source_id: sid.to_string(),
                    worker_id: wid.to_string(),
                    model_name: "my-model".to_string(),
                    workers: vec![make_worker_record(0, tensors)],
                    published_at: 0,
                }))
            }
        });

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert_eq!(resp.peers.len(), 2);

        let all_assigned: Vec<&str> = resp
            .peers
            .iter()
            .flat_map(|p| p.assigned_tensor_names.iter().map(|s| s.as_str()))
            .collect();
        assert_eq!(all_assigned.len(), 4);

        for peer in &resp.peers {
            assert!(!peer.nixl_metadata.is_empty());
            assert!(!peer.agent_name.is_empty());
        }

        let diag = resp.diagnostics.expect("diagnostics");
        assert_eq!(diag.candidates_total, 2);
        assert_eq!(diag.candidates_eligible, 2);
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_reports_explicit_missing_tensor_uncovered() {
        let source_id = compute_mx_source_id(&test_identity());

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning({
            let source_id = source_id.clone();
            move |_, _| {
                Ok(vec![SourceInstanceInfo {
                    source_id: source_id.clone(),
                    worker_id: "peer-1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 0,
                }])
            }
        });

        mock.expect_get_metadata().once().returning(|sid, wid| {
            Ok(Some(ModelMetadataRecord {
                source_id: sid.to_string(),
                worker_id: wid.to_string(),
                model_name: "my-model".to_string(),
                workers: vec![make_worker_record(0, &[("a", 100)])],
                published_at: 0,
            }))
        });

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: vec!["a".to_string(), "missing".to_string()],
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert_eq!(resp.peers.len(), 1);
        assert_eq!(resp.peers[0].assigned_tensor_names, vec!["a".to_string()]);
        assert_eq!(resp.uncovered_tensor_names, vec!["missing".to_string()]);
        let diag = resp.diagnostics.expect("diagnostics");
        assert!(
            diag.note
                .contains("1 requested tensor(s) absent from all peer catalogs")
        );
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_uses_persisted_catalog() {
        let source_id = compute_mx_source_id(&test_identity());
        let mut worker = make_worker_record(0, &[("synthetic-only", 999)]);
        worker.tensor_catalog = Some(TensorCatalogRecord {
            generation: 5,
            entries: vec![TensorCatalogEntryRecord {
                name: "catalog-only".to_string(),
                byte_len: 42,
                dtype: "bfloat16".to_string(),
                shape: vec![2, 21],
            }],
        });

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning({
            let source_id = source_id.clone();
            move |_, _| {
                Ok(vec![SourceInstanceInfo {
                    source_id: source_id.clone(),
                    worker_id: "peer-1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 0,
                }])
            }
        });
        // One peer lookup plus the requester's own catalog lookup (empty
        // requested_tensor_names triggers the implied need-set path).
        mock.expect_get_metadata()
            .times(2)
            .returning(move |sid, wid| Ok(Some(make_model_record(sid, wid, worker.clone()))));

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert_eq!(resp.peers.len(), 1);
        assert_eq!(
            resp.peers[0].assigned_tensor_names,
            vec!["catalog-only".to_string()]
        );
        assert!(resp.uncovered_tensor_names.is_empty());
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_scopes_to_requester_catalog() {
        // Peer owns {a, b}; the requester advertised a catalog of only {a}.
        // With empty requested_tensor_names the plan must scope to the
        // requester's need-set: "a" assigned, "b" never considered (not in
        // the plan, not in uncovered).
        let source_id = compute_mx_source_id(&test_identity());

        let mut requester = make_worker_record(0, &[]);
        requester.tensor_catalog = Some(TensorCatalogRecord {
            generation: 1,
            entries: vec![TensorCatalogEntryRecord {
                name: "a".to_string(),
                byte_len: 100,
                dtype: "bfloat16".to_string(),
                shape: vec![],
            }],
        });

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning({
            let source_id = source_id.clone();
            move |_, _| {
                Ok(vec![SourceInstanceInfo {
                    source_id: source_id.clone(),
                    worker_id: "peer-1".to_string(),
                    model_name: "my-model".to_string(),
                    worker_rank: 0,
                    status: SourceStatus::Ready as i32,
                    updated_at: 0,
                }])
            }
        });
        // The requester lookup returns its advertised {a} catalog; every
        // other lookup returns the peer that owns {a, b}.
        mock.expect_get_metadata()
            .withf(|_, wid| wid == "requester")
            .once()
            .returning(move |sid, wid| Ok(Some(make_model_record(sid, wid, requester.clone()))));
        mock.expect_get_metadata()
            .withf(|_, wid| wid != "requester")
            .once()
            .returning(|sid, wid| {
                Ok(Some(ModelMetadataRecord {
                    source_id: sid.to_string(),
                    worker_id: wid.to_string(),
                    model_name: "my-model".to_string(),
                    workers: vec![make_worker_record(0, &[("a", 100), ("b", 80)])],
                    published_at: 0,
                }))
            });

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert_eq!(resp.peers.len(), 1);
        assert_eq!(resp.peers[0].assigned_tensor_names, vec!["a".to_string()]);
        assert!(resp.uncovered_tensor_names.is_empty());
        let diag = resp.diagnostics.expect("diagnostics");
        assert!(diag.note.contains("requester's advertised catalog"));
    }

    #[tokio::test]
    async fn test_compute_transfer_plan_filters_by_rank() {
        let source_id = compute_mx_source_id(&test_identity());

        let mut mock = MockMetadataBackend::new();
        mock.expect_list_workers().once().returning({
            let source_id = source_id.clone();
            move |_, _| {
                Ok(vec![
                    SourceInstanceInfo {
                        source_id: source_id.clone(),
                        worker_id: "peer-rank0".to_string(),
                        model_name: "my-model".to_string(),
                        worker_rank: 0,
                        status: SourceStatus::Ready as i32,
                        updated_at: 0,
                    },
                    SourceInstanceInfo {
                        source_id: source_id.clone(),
                        worker_id: "peer-rank1".to_string(),
                        model_name: "my-model".to_string(),
                        worker_rank: 1,
                        status: SourceStatus::Ready as i32,
                        updated_at: 0,
                    },
                ])
            }
        });

        // One eligible peer lookup plus the requester's own catalog lookup.
        mock.expect_get_metadata().times(2).returning(|sid, wid| {
            Ok(Some(ModelMetadataRecord {
                source_id: sid.to_string(),
                worker_id: wid.to_string(),
                model_name: "my-model".to_string(),
                workers: vec![make_worker_record(0, &[("a", 100)])],
                published_at: 0,
            }))
        });

        let svc = make_service(mock);
        let resp = svc
            .compute_transfer_plan(Request::new(ComputeTransferPlanRequest {
                identity: Some(test_identity()),
                requester_worker_rank: 0,
                requester_worker_id: "requester".to_string(),
                requested_tensor_names: Vec::new(),
                max_peers: None,
            }))
            .await
            .expect("rpc")
            .into_inner();

        assert_eq!(resp.peers.len(), 1);
        assert_eq!(resp.peers[0].worker_id, "peer-rank0");

        let diag = resp.diagnostics.expect("diagnostics");
        assert_eq!(diag.candidates_total, 2);
        assert_eq!(diag.candidates_eligible, 1);
    }
}
