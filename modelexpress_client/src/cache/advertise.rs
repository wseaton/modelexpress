// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building blocks for advertising a node's cached models to the P2P registry:
//! the `FILE_CACHE` [`SourceIdentity`], the [`WorkerMetadata`] carrying the NIXL
//! blob, and a stable per-node worker id.
//!
//! The daemon loop (Phase 3) drives these: publish once, then heartbeat via
//! [`super::registry::Registry::update_status`] inside the reaper's timeout.

use std::time::{SystemTime, UNIX_EPOCH};

use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
use modelexpress_common::grpc::p2p::{MxSourceType, SourceIdentity, SourceStatus, WorkerMetadata};

/// Protocol version this daemon speaks; part of the published identity so peers
/// on an incompatible version don't match.
const MX_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A node holds a single cache identity, so its registry worker id is stable
/// across restarts: a restart re-publishes the same id (overwrite) instead of
/// leaking a fresh entry the reaper has to clean up. Prefers `POD_NAME`, then
/// `HOSTNAME`, then `gethostname(2)`.
pub fn worker_id() -> String {
    resolve_worker_id(
        std::env::var("POD_NAME").ok(),
        std::env::var("HOSTNAME").ok().or_else(gethostname),
    )
}

fn resolve_worker_id(pod_name: Option<String>, hostname: Option<String>) -> String {
    pod_name
        .filter(|s| !s.is_empty())
        .or_else(|| hostname.filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "mx-cache".to_string())
}

fn gethostname() -> Option<String> {
    let mut buf = [0u8; 256];
    // SAFETY: `buf` is a valid, writable 256-byte buffer; gethostname writes at
    // most `buf.len()` bytes and NUL-terminates on success.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast::<libc::c_char>(), buf.len()) };
    if rc != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8(buf[..end].to_vec())
        .ok()
        .filter(|s| !s.is_empty())
}

/// The identity a cached model is published under. Keyed on `model_name` +
/// `FILE_CACHE` + protocol version (+ `revision` when known); the GPU-layout
/// fields stay at their defaults because a file cache is framework- and
/// parallelism-agnostic.
pub fn file_cache_identity(
    model_name: impl Into<String>,
    revision: impl Into<String>,
) -> SourceIdentity {
    SourceIdentity {
        mx_version: MX_VERSION.to_string(),
        mx_source_type: MxSourceType::FileCache as i32,
        model_name: model_name.into(),
        revision: revision.into(),
        // GPU-layout and compile-artifact fields stay at their defaults: a file
        // cache is framework- and parallelism-agnostic.
        ..Default::default()
    }
}

/// The worker metadata for a cached model: the NIXL blob a puller loads, the
/// agent name, and the listen endpoint. `updated_at` is stamped now because the
/// publish path trusts the client's timestamp (only `update_status` is
/// server-stamped); an unstamped record would be reaped on the first sweep. The
/// file manifest is not carried here, it moves in-band over the NIXL notif
/// channel.
pub fn cache_worker(
    nixl_md: Vec<u8>,
    agent_name: impl Into<String>,
    metadata_endpoint: impl Into<String>,
) -> WorkerMetadata {
    WorkerMetadata {
        worker_rank: 0,
        backend_metadata: Some(BackendMetadata::NixlMetadata(nixl_md)),
        status: SourceStatus::Ready as i32,
        updated_at: now_millis(),
        metadata_endpoint: metadata_endpoint.into(),
        agent_name: agent_name.into(),
        ..Default::default()
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn worker_id_prefers_pod_then_host_then_fallback() {
        assert_eq!(
            resolve_worker_id(Some("pod-7".into()), Some("host-1".into())),
            "pod-7"
        );
        assert_eq!(
            resolve_worker_id(None, Some("host-1".into())),
            "host-1",
            "falls back to hostname"
        );
        assert_eq!(
            resolve_worker_id(Some(String::new()), Some("host-1".into())),
            "host-1",
            "empty pod name is ignored"
        );
        assert_eq!(resolve_worker_id(None, None), "mx-cache");
    }

    #[test]
    fn identity_is_file_cache_typed() {
        let id = file_cache_identity("google-t5/t5-small", "");
        assert_eq!(id.mx_source_type, MxSourceType::FileCache as i32);
        assert_eq!(id.model_name, "google-t5/t5-small");
        assert_eq!(id.mx_version, MX_VERSION);
        // GPU-layout fields stay at defaults.
        assert_eq!(id.tensor_parallel_size, 0);
        assert!(id.dtype.is_empty());
        assert!(id.quantization.is_empty());
    }

    #[test]
    fn worker_carries_blob_ready_and_stamped() {
        let worker = cache_worker(vec![1, 2, 3, 4], "agent-a", "10.0.0.1:7000");
        assert_eq!(worker.worker_rank, 0);
        assert_eq!(worker.status, SourceStatus::Ready as i32);
        assert_eq!(worker.agent_name, "agent-a");
        assert_eq!(worker.metadata_endpoint, "10.0.0.1:7000");
        assert!(
            worker.source_payload.is_none(),
            "manifest moves in-band, not here"
        );
        assert!(worker.updated_at > 0, "must be stamped or it is born stale");
        match worker.backend_metadata {
            Some(BackendMetadata::NixlMetadata(blob)) => assert_eq!(blob, vec![1, 2, 3, 4]),
            other => panic!("expected NixlMetadata, got {other:?}"),
        }
    }
}
