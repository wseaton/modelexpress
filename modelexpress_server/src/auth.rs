// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authn/authz for the P2P gRPC interface. Verifies a bound, audience-scoped projected
//! ServiceAccount token via Kubernetes `TokenReview`, then confirms the caller's pod
//! actually requests a fabric device (IB / RoCE / EFA). The server is the enforcement
//! point.

mod device;
mod layer;
mod store;
mod token;

pub use layer::AuthLayer;

use std::time::Duration;

use http::HeaderMap;
use kube::client::Client;
use moka::future::Cache;
use secrecy::ExposeSecret;
use sha2::{Digest, Sha256};

use crate::config::{AuthMode, SecurityConfig};
use device::DeviceResource;
use store::{DeviceDecision, DeviceStore};
use token::{CallerIdentity, extract_bearer, review_token};

/// Reason a gated request was rejected.
#[derive(Debug, thiserror::Error)]
pub enum Denial {
    #[error("missing or invalid service account token")]
    Unauthenticated,
    #[error("{0}")]
    PermissionDenied(String),
}

impl Denial {
    pub(crate) fn into_status(self) -> tonic::Status {
        match self {
            Self::Unauthenticated => {
                tonic::Status::unauthenticated("missing or invalid service account token")
            }
            Self::PermissionDenied(message) => tonic::Status::permission_denied(message),
        }
    }
}

/// Shared verification state: Kubernetes client, configured policy, and short-TTL caches.
pub struct AuthState {
    pub mode: AuthMode,
    client: Client,
    audiences: Vec<String>,
    devices: Vec<DeviceResource>,
    device_classes: Vec<String>,
    store: DeviceStore,
    token_cache: Cache<[u8; 32], CallerIdentity>,
    /// `try_get_with` does not cache failures, so without this a flood of distinct bad
    /// tokens becomes a flood of `TokenReview` calls. Only definitive rejections land here.
    negative_cache: Cache<[u8; 32], ()>,
}

impl AuthState {
    /// Build the auth state and start the pod (and, for DRA, ResourceClaim) reflectors.
    /// Errors if the reflectors cannot complete their initial sync so the caller can fail
    /// closed.
    pub async fn new(
        client: Client,
        config: &SecurityConfig,
        mode: AuthMode,
    ) -> Result<Self, String> {
        let ttl = Duration::from_secs(config.cache_ttl_secs);
        let dra_enabled = !config.device_classes.is_empty();
        let store =
            DeviceStore::new(&client, dra_enabled, config.pod_label_selector.as_deref()).await?;
        Ok(Self {
            mode,
            client,
            audiences: config.token_audiences.clone(),
            devices: config
                .device_resources
                .iter()
                .cloned()
                .map(DeviceResource)
                .collect(),
            device_classes: config.device_classes.clone(),
            store,
            token_cache: Cache::builder().time_to_live(ttl).build(),
            negative_cache: Cache::builder().time_to_live(ttl).build(),
        })
    }

    /// Verify a request's credentials. Fails closed; the tower layer decides whether a
    /// denial blocks (Enforce) or is only logged (Permissive).
    pub(crate) async fn verify(&self, headers: &HeaderMap) -> Result<CallerIdentity, Denial> {
        let token = extract_bearer(headers).ok_or(Denial::Unauthenticated)?;

        // Cache key is a hash of the token, never the token itself.
        let mut hasher = Sha256::new();
        hasher.update(token.expose_secret().as_bytes());
        let key: [u8; 32] = hasher.finalize().into();

        if self.negative_cache.get(&key).await.is_some() {
            return Err(Denial::Unauthenticated);
        }

        let client = self.client.clone();
        let audiences = self.audiences.clone();
        let identity = match self
            .token_cache
            .try_get_with(key, async move {
                review_token(&client, token.expose_secret(), &audiences).await
            })
            .await
        {
            Ok(identity) => identity,
            Err(error) => {
                if error.is_token_rejection() {
                    self.negative_cache.insert(key, ()).await;
                }
                return Err(Denial::Unauthenticated);
            }
        };

        let pod_name = identity.pod_name.as_deref().ok_or_else(|| {
            Denial::PermissionDenied("caller token carries no pod identity".into())
        })?;
        let pod_uid = identity.pod_uid.as_deref().ok_or_else(|| {
            Denial::PermissionDenied("caller token carries no pod identity".into())
        })?;

        match self.store.decide(
            &identity.namespace,
            pod_name,
            pod_uid,
            &self.devices,
            &self.device_classes,
        ) {
            DeviceDecision::Allowed => Ok(identity),
            DeviceDecision::PodNotFound => Err(Denial::PermissionDenied(format!(
                "caller pod {}/{pod_name} not found",
                identity.namespace
            ))),
            DeviceDecision::NoDevice => Err(Denial::PermissionDenied(format!(
                "caller pod {}/{pod_name} does not hold a required device",
                identity.namespace
            ))),
            DeviceDecision::Unavailable => Err(Denial::PermissionDenied(
                "device store unavailable, failing closed".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unauthenticated_maps_to_grpc_code() {
        let status = Denial::Unauthenticated.into_status();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn permission_denied_maps_to_grpc_code() {
        let status = Denial::PermissionDenied("no device".to_string()).into_status();
        assert_eq!(status.code(), tonic::Code::PermissionDenied);
        assert_eq!(status.message(), "no device");
    }
}
