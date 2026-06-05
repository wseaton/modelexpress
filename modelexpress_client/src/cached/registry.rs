// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin async wrapper over the generated P2P metadata gRPC client, scoped to the
//! four calls the cache daemon needs: publish its residency, heartbeat it,
//! discover peers (`list_ready`), and fetch a peer's metadata blob.
//!
//! No NIXL here, so it builds and is exercised by a real in-process server in
//! CI (see `modelexpress_server/tests`).

use std::time::Duration;

use anyhow::{Context, bail};
use modelexpress_common::grpc::p2p::p2p_service_client::P2pServiceClient;
use modelexpress_common::grpc::p2p::{
    GetMetadataRequest, ListSourcesRequest, PublishMetadataRequest, SourceIdentity,
    SourceInstanceRef, SourceStatus, UpdateStatusRequest, WorkerMetadata,
};
use tonic::transport::{Channel, Endpoint};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Connected handle to a ModelExpress server's P2P registry. Cheap to clone
/// (the tonic client shares the channel), so a heartbeat task can hold its own.
#[derive(Clone)]
pub struct Registry {
    client: P2pServiceClient<Channel>,
}

impl Registry {
    /// Connect to a registry at `endpoint` (e.g. `http://host:8001`).
    pub async fn connect(endpoint: impl Into<String>) -> anyhow::Result<Self> {
        let channel = Endpoint::new(endpoint.into())?
            .timeout(CONNECT_TIMEOUT)
            .connect()
            .await
            .context("connect to P2P registry")?;
        Ok(Self::from_channel(channel))
    }

    /// Wrap an already-connected channel (used by in-process tests).
    pub fn from_channel(channel: Channel) -> Self {
        Self {
            client: P2pServiceClient::new(channel),
        }
    }

    /// Publish one worker's metadata; returns the server-computed `mx_source_id`.
    pub async fn publish(
        &mut self,
        identity: SourceIdentity,
        worker: WorkerMetadata,
        worker_id: impl Into<String>,
    ) -> anyhow::Result<String> {
        let resp = self
            .client
            .publish_metadata(PublishMetadataRequest {
                identity: Some(identity),
                worker: Some(worker),
                worker_id: worker_id.into(),
            })
            .await?
            .into_inner();
        if !resp.success {
            bail!("registry rejected publish: {}", resp.message);
        }
        Ok(resp.mx_source_id)
    }

    /// Refresh a worker's status; the heartbeat keeps it out of the reaper's
    /// reach (the server re-stamps the timestamp).
    pub async fn update_status(
        &mut self,
        mx_source_id: impl Into<String>,
        worker_rank: u32,
        status: SourceStatus,
        worker_id: impl Into<String>,
    ) -> anyhow::Result<()> {
        let resp = self
            .client
            .update_status(UpdateStatusRequest {
                mx_source_id: mx_source_id.into(),
                worker_rank,
                status: status as i32,
                worker_id: worker_id.into(),
            })
            .await?
            .into_inner();
        if !resp.success {
            bail!("registry rejected status update: {}", resp.message);
        }
        Ok(())
    }

    /// List the READY workers matching `identity`.
    pub async fn list_ready(
        &mut self,
        identity: SourceIdentity,
    ) -> anyhow::Result<Vec<SourceInstanceRef>> {
        let resp = self
            .client
            .list_sources(ListSourcesRequest {
                identity: Some(identity),
                status_filter: Some(SourceStatus::Ready as i32),
            })
            .await?
            .into_inner();
        Ok(resp.instances)
    }

    /// Fetch one worker's full metadata; `None` if the server has no such worker.
    pub async fn get_worker(
        &mut self,
        mx_source_id: impl Into<String>,
        worker_id: impl Into<String>,
    ) -> anyhow::Result<Option<WorkerMetadata>> {
        let resp = self
            .client
            .get_metadata(GetMetadataRequest {
                mx_source_id: mx_source_id.into(),
                worker_id: worker_id.into(),
            })
            .await?
            .into_inner();
        Ok(resp.found.then_some(resp.worker).flatten())
    }
}
