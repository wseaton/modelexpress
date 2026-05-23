// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background-synced stores of caller pods (and, for DRA, ResourceClaims) via kube-rs
//! reflectors. The device check is then an O(1) in-memory lookup with no API round-trip
//! in the request path. Watched cluster-wide since caller pods may live in any namespace.

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::resource::v1beta1::ResourceClaim;
use kube::ResourceExt;
use kube::api::Api;
use kube::client::Client;
use kube::runtime::WatchStreamExt;
use kube::runtime::reflector::{self, ObjectRef, Store};
use kube::runtime::watcher;
use serde::de::DeserializeOwned;
use tracing::{error, warn};

use crate::auth::device::{
    DeviceResource, claim_matches_class, pod_requests_device, resolve_claim_names,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceDecision {
    Allowed,
    /// Caller pod is not in the store (deleted, or uid mismatch).
    PodNotFound,
    NoDevice,
    /// A backing reflector stopped updating; fail closed.
    Unavailable,
}

/// A reflector's store plus a `healthy` flag the task clears if its watch stream ends, so
/// a stale store can't fail-open by "finding" deleted pods.
#[derive(Clone)]
struct Reflector<K: kube::Resource<DynamicType = ()> + 'static> {
    store: Store<K>,
    healthy: Arc<AtomicBool>,
}

impl<K: kube::Resource<DynamicType = ()> + 'static> Reflector<K> {
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
pub(crate) struct DeviceStore {
    pods: Reflector<Pod>,
    claims: Option<Reflector<ResourceClaim>>,
}

impl DeviceStore {
    /// Wait for both reflectors' initial list to sync. The ResourceClaim reflector only
    /// starts when DRA is in use.
    ///
    /// `pod_label_selector` narrows the pod reflector (smaller store on large clusters);
    /// pods not matching it are absent and therefore fail closed. Claims aren't
    /// label-filtered since they're looked up by the exact name the pod references.
    pub(crate) async fn new(
        client: &Client,
        dra_enabled: bool,
        pod_label_selector: Option<&str>,
    ) -> Result<Self, String> {
        let pods = start_reflector::<Pod>(client.clone(), "pods", pod_label_selector).await?;
        let claims = if dra_enabled {
            Some(start_reflector::<ResourceClaim>(client.clone(), "resourceclaims", None).await?)
        } else {
            None
        };
        Ok(Self { pods, claims })
    }

    pub(crate) fn decide(
        &self,
        namespace: &str,
        pod_name: &str,
        pod_uid: &str,
        devices: &[DeviceResource],
        classes: &[String],
    ) -> DeviceDecision {
        if !self.pods.is_healthy() {
            return DeviceDecision::Unavailable;
        }
        let Some(pod) = self
            .pods
            .store
            .get(&ObjectRef::new(pod_name).within(namespace))
        else {
            return DeviceDecision::PodNotFound;
        };
        // Guard against name reuse: same incarnation the token was bound to.
        if pod.metadata.uid.as_deref() != Some(pod_uid) {
            return DeviceDecision::PodNotFound;
        }
        if pod_requests_device(&pod, devices) {
            return DeviceDecision::Allowed;
        }
        if classes.is_empty() {
            return DeviceDecision::NoDevice;
        }
        if let Some(claims) = self.claims.as_ref() {
            if !claims.is_healthy() {
                return DeviceDecision::Unavailable;
            }
            for claim_name in resolve_claim_names(&pod) {
                if let Some(claim) = claims
                    .store
                    .get(&ObjectRef::new(&claim_name).within(namespace))
                    && claim_matches_class(&claim, classes)
                {
                    return DeviceDecision::Allowed;
                }
            }
        }
        DeviceDecision::NoDevice
    }
}

/// Drain `stream` into `writer`, opening the `healthy` gate on `InitDone`. Returns when
/// the stream ends; `true` if it synced at least once.
async fn run_watch<K, S>(
    kind: &str,
    stream: S,
    writer: &mut reflector::store::Writer<K>,
    healthy: &AtomicBool,
) -> bool
where
    K: kube::Resource<DynamicType = ()> + Clone + 'static,
    S: futures::Stream<Item = Result<watcher::Event<K>, watcher::Error>>,
{
    futures::pin_mut!(stream);
    let mut synced = false;
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => {
                let init_done = matches!(event, watcher::Event::InitDone);
                writer.apply_watcher_event(&event);
                if init_done {
                    healthy.store(true, Ordering::Relaxed);
                    synced = true;
                }
            }
            Err(e) => warn!("{kind} reflector watch error: {e}"),
        }
    }
    synced
}

/// Start a cluster-wide reflector for `K` and wait for its initial list to sync.
async fn start_reflector<K>(
    client: Client,
    kind: &'static str,
    label_selector: Option<&str>,
) -> Result<Reflector<K>, String>
where
    K: kube::Resource<DynamicType = ()> + Clone + DeserializeOwned + Debug + Send + Sync + 'static,
{
    let (reader, mut writer) = reflector::store::<K>();
    let mut config = watcher::Config::default();
    if let Some(selector) = label_selector {
        config = config.labels(selector);
    }
    let api: Api<K> = Api::all(client);
    let healthy = Arc::new(AtomicBool::new(false));
    let task_healthy = healthy.clone();
    let task_api = api.clone();
    let task_config = config.clone();
    // watcher retries transient errors itself; this loop only fires if the stream ends.
    // Keep the same Writer (so the caller's Store stays valid), fail closed while
    // resyncing, and reconnect rather than tear down the server.
    tokio::spawn(async move {
        let mut backoff = Duration::from_secs(1);
        const MAX_BACKOFF: Duration = Duration::from_secs(30);
        loop {
            let stream = watcher(task_api.clone(), task_config.clone())
                .default_backoff()
                // managedFields is large and unused; clearing it shrinks the store.
                .modify(|obj| obj.managed_fields_mut().clear());
            let synced = run_watch(kind, stream, &mut writer, &task_healthy).await;
            task_healthy.store(false, Ordering::Relaxed);
            if synced {
                backoff = Duration::from_secs(1);
            }
            error!("{kind} reflector stream ended; store unhealthy, reconnecting in {backoff:?}");
            tokio::time::sleep(backoff).await;
            backoff = backoff.saturating_mul(2).min(MAX_BACKOFF);
        }
    });
    reader
        .wait_until_ready()
        .await
        .map_err(|e| format!("{kind} reflector failed to become ready: {e}"))?;
    healthy.store(true, Ordering::Relaxed);
    Ok(Reflector {
        store: reader,
        healthy,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{Container, PodSpec, ResourceRequirements};
    use k8s_openapi::api::resource::v1beta1::{DeviceClaim, DeviceRequest, ResourceClaimSpec};
    use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
    use std::collections::BTreeMap;

    fn pod(name: &str, ns: &str, uid: &str, device: Option<&str>, claim: Option<&str>) -> Pod {
        let resources = device.map(|d| {
            let mut limits = BTreeMap::new();
            limits.insert(d.to_string(), Quantity("1".to_string()));
            ResourceRequirements {
                limits: Some(limits),
                ..Default::default()
            }
        });
        let resource_claims = claim.map(|c| {
            vec![k8s_openapi::api::core::v1::PodResourceClaim {
                name: "ib".to_string(),
                resource_claim_name: Some(c.to_string()),
                ..Default::default()
            }]
        });
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                namespace: Some(ns.to_string()),
                uid: Some(uid.to_string()),
                ..Default::default()
            },
            spec: Some(PodSpec {
                containers: vec![Container {
                    name: "c".to_string(),
                    resources,
                    ..Default::default()
                }],
                resource_claims,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn claim_obj(name: &str, ns: &str, class: &str) -> ResourceClaim {
        ResourceClaim {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                namespace: Some(ns.to_string()),
                ..Default::default()
            },
            spec: ResourceClaimSpec {
                devices: Some(DeviceClaim {
                    requests: Some(vec![DeviceRequest {
                        name: "ib".to_string(),
                        device_class_name: class.to_string(),
                        ..Default::default()
                    }]),
                    ..Default::default()
                }),
            },
            ..Default::default()
        }
    }

    fn seed_store<K>(objects: Vec<K>) -> Store<K>
    where
        K: kube::Resource<DynamicType = ()> + Clone + std::fmt::Debug + Send + Sync + 'static,
    {
        let mut writer = reflector::store::Writer::<K>::default();
        for obj in objects {
            writer.apply_watcher_event(&watcher::Event::Apply(obj));
        }
        writer.as_reader()
    }

    fn reflector<K>(objects: Vec<K>) -> Reflector<K>
    where
        K: kube::Resource<DynamicType = ()> + Clone + std::fmt::Debug + Send + Sync + 'static,
    {
        Reflector {
            store: seed_store(objects),
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    fn devices() -> Vec<DeviceResource> {
        vec![DeviceResource("rdma/ib".to_string())]
    }

    #[test]
    fn device_plugin_pod_is_allowed() {
        let store = DeviceStore {
            pods: reflector(vec![pod("w0", "ns", "uid-0", Some("rdma/ib"), None)]),
            claims: None,
        };
        assert_eq!(
            store.decide("ns", "w0", "uid-0", &devices(), &[]),
            DeviceDecision::Allowed
        );
    }

    #[test]
    fn pod_without_device_is_nodevice() {
        let store = DeviceStore {
            pods: reflector(vec![pod("w0", "ns", "uid-0", None, None)]),
            claims: None,
        };
        assert_eq!(
            store.decide("ns", "w0", "uid-0", &devices(), &[]),
            DeviceDecision::NoDevice
        );
    }

    #[test]
    fn missing_pod_is_podnotfound() {
        let store = DeviceStore {
            pods: reflector(Vec::<Pod>::new()),
            claims: None,
        };
        assert_eq!(
            store.decide("ns", "ghost", "uid-x", &devices(), &[]),
            DeviceDecision::PodNotFound
        );
    }

    #[test]
    fn uid_mismatch_is_podnotfound() {
        let store = DeviceStore {
            pods: reflector(vec![pod("w0", "ns", "uid-new", Some("rdma/ib"), None)]),
            claims: None,
        };
        // Token was bound to an older incarnation (uid-old) of the same name.
        assert_eq!(
            store.decide("ns", "w0", "uid-old", &devices(), &[]),
            DeviceDecision::PodNotFound
        );
    }

    #[test]
    fn dra_pod_with_matching_class_is_allowed() {
        let store = DeviceStore {
            pods: reflector(vec![pod("w0", "ns", "uid-0", None, Some("ib-claim"))]),
            claims: Some(reflector(vec![claim_obj(
                "ib-claim",
                "ns",
                "rdma.nvidia.com",
            )])),
        };
        assert_eq!(
            store.decide("ns", "w0", "uid-0", &[], &["rdma.nvidia.com".to_string()]),
            DeviceDecision::Allowed
        );
    }

    #[test]
    fn dra_pod_with_unknown_class_is_nodevice() {
        let store = DeviceStore {
            pods: reflector(vec![pod("w0", "ns", "uid-0", None, Some("ib-claim"))]),
            claims: Some(reflector(vec![claim_obj(
                "ib-claim",
                "ns",
                "gpu.nvidia.com",
            )])),
        };
        assert_eq!(
            store.decide("ns", "w0", "uid-0", &[], &["rdma.nvidia.com".to_string()]),
            DeviceDecision::NoDevice
        );
    }

    #[test]
    fn unhealthy_pod_store_fails_closed() {
        // Pod holds a device, but its reflector died: deny, not stale Allowed.
        let pods = reflector(vec![pod("w0", "ns", "uid-0", Some("rdma/ib"), None)]);
        pods.healthy.store(false, Ordering::Relaxed);
        let store = DeviceStore { pods, claims: None };
        assert_eq!(
            store.decide("ns", "w0", "uid-0", &devices(), &[]),
            DeviceDecision::Unavailable
        );
    }

    #[tokio::test]
    async fn run_watch_syncs_store_and_opens_gate() {
        let (reader, mut writer) = reflector::store::<Pod>();
        let healthy = AtomicBool::new(false);
        let events: Vec<Result<watcher::Event<Pod>, watcher::Error>> = vec![
            Ok(watcher::Event::Init),
            Ok(watcher::Event::InitApply(pod(
                "w0",
                "ns",
                "uid-0",
                Some("rdma/ib"),
                None,
            ))),
            Ok(watcher::Event::InitDone),
        ];
        let synced = run_watch("pods", futures::stream::iter(events), &mut writer, &healthy).await;
        assert!(synced, "InitDone should report a completed sync");
        assert!(healthy.load(Ordering::Relaxed), "gate should be open");
        assert!(
            reader.get(&ObjectRef::new("w0").within("ns")).is_some(),
            "store should hold the applied pod"
        );
    }

    #[test]
    fn unhealthy_claim_store_fails_closed() {
        // Claim reflector died: don't fall through to NoDevice for a DRA caller.
        let claims = reflector(vec![claim_obj("ib-claim", "ns", "rdma.nvidia.com")]);
        claims.healthy.store(false, Ordering::Relaxed);
        let store = DeviceStore {
            pods: reflector(vec![pod("w0", "ns", "uid-0", None, Some("ib-claim"))]),
            claims: Some(claims),
        };
        assert_eq!(
            store.decide("ns", "w0", "uid-0", &[], &["rdma.nvidia.com".to_string()]),
            DeviceDecision::Unavailable
        );
    }
}
