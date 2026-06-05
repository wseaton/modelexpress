// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer discovery: turn a model identity into a NIXL metadata blob to pull from,
//! via the P2P registry. This replaces the hand-passed `HOLDER_MD` of the spike;
//! the blob feeds straight into [`crate::cached::transfer::puller::Puller::pull`].

use modelexpress_common::grpc::p2p::SourceIdentity;
use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;

use super::registry::Registry;

/// Find a peer holding `identity` and return its NIXL metadata blob. Lists READY
/// workers, takes the first, and fetches its blob; `None` when no peer currently
/// holds the model (the caller then falls back to origin). Peer-selection policy
/// (herd spreading, cascade) is a Phase 4 concern; for now this takes the first.
pub async fn discover_blob(
    registry: &mut Registry,
    identity: SourceIdentity,
) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(instance) = registry.list_ready(identity).await?.into_iter().next() else {
        return Ok(None);
    };
    let Some(worker) = registry
        .get_worker(instance.mx_source_id, instance.worker_id)
        .await?
    else {
        return Ok(None);
    };
    Ok(match worker.backend_metadata {
        Some(BackendMetadata::NixlMetadata(blob)) => Some(blob),
        _ => None,
    })
}
