// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Renders the model-cache volume for the server pod.

use crate::crd::{CacheStorage, ModelExpressServerSpec};
use crate::labels::managed_labels;
use k8s_openapi::api::core::v1::{
    EmptyDirVolumeSource, PersistentVolumeClaim, PersistentVolumeClaimVolumeSource, Volume,
    VolumeMount,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use kube::api::ObjectMeta;

pub const VOLUME_NAME: &str = "model-cache";
/// Not /root: the chart relied on the server resolving its cache under HOME,
/// but the images run as UID 1000 and OpenShift's restricted SCC assigns an
/// arbitrary UID with no passwd entry, so HOME is unwritable either way.
pub const DEFAULT_MOUNT_PATH: &str = "/var/cache/modelexpress";
pub const PVC_NAME_SUFFIX: &str = "model-cache";

/// Everything the Deployment (and reconciler) needs for the cache mount. The
/// PVC is only present for operator-managed storage; ownerReferences are the
/// reconciler's job.
pub struct CacheVolume {
    pub volume: Volume,
    pub mount: VolumeMount,
    pub pvc: Option<PersistentVolumeClaim>,
}

pub fn managed_pvc_name(cr_name: &str) -> String {
    format!("{cr_name}-{PVC_NAME_SUFFIX}")
}

/// Also what the server is told via MODEL_EXPRESS_CACHE_DIRECTORY, so the
/// mount and the env var cannot drift.
pub fn mount_path(spec: &ModelExpressServerSpec) -> &str {
    spec.cache
        .as_ref()
        .and_then(|c| c.directory.as_deref())
        .unwrap_or(DEFAULT_MOUNT_PATH)
}

pub fn render_cache_volume(cr_name: &str, spec: &ModelExpressServerSpec) -> CacheVolume {
    let cache = spec.cache.as_ref();

    let mount = VolumeMount {
        name: VOLUME_NAME.to_string(),
        mount_path: mount_path(spec).to_string(),
        ..VolumeMount::default()
    };

    let storage = cache.and_then(|c| c.storage.as_ref());
    let (volume, pvc) = match storage {
        // default: ephemeral cache, safe at any replica count
        None => (empty_dir_volume(None), None),
        Some(CacheStorage::EmptyDir(empty_dir)) => (
            empty_dir_volume(empty_dir.size_limit.clone().map(Quantity)),
            None,
        ),
        Some(CacheStorage::ExistingClaim(existing)) => {
            (claim_volume(existing.claim_name.clone()), None)
        }
        Some(CacheStorage::Pvc(managed)) => {
            let claim_name = managed_pvc_name(cr_name);
            let mut claim_spec = managed.spec.clone();
            if claim_spec.access_modes.is_none() {
                claim_spec.access_modes = Some(vec!["ReadWriteOnce".to_string()]);
            }
            let claim_meta = managed.metadata.clone().unwrap_or_default();
            // user labels first: the operator's own must win, or the claim
            // drops out of the controller's watch
            let mut labels = claim_meta.labels.unwrap_or_default();
            labels.extend(managed_labels(cr_name));
            let pvc = PersistentVolumeClaim {
                metadata: ObjectMeta {
                    name: Some(claim_name.clone()),
                    labels: Some(labels),
                    annotations: claim_meta.annotations,
                    ..ObjectMeta::default()
                },
                spec: Some(claim_spec),
                ..PersistentVolumeClaim::default()
            };
            (claim_volume(claim_name), Some(pvc))
        }
    };

    CacheVolume { volume, mount, pvc }
}

fn empty_dir_volume(size_limit: Option<Quantity>) -> Volume {
    Volume {
        name: VOLUME_NAME.to_string(),
        empty_dir: Some(EmptyDirVolumeSource {
            size_limit,
            ..EmptyDirVolumeSource::default()
        }),
        ..Volume::default()
    }
}

fn claim_volume(claim_name: String) -> Volume {
    Volume {
        name: VOLUME_NAME.to_string(),
        persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
            claim_name,
            read_only: None,
        }),
        ..Volume::default()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::crd::{
        CacheConfig, EmptyDirStorage, ExistingClaimStorage, ManagedPvcStorage, MetadataBackend,
        MetadataOverrides,
    };
    use k8s_openapi::api::core::v1::{PersistentVolumeClaimSpec, VolumeResourceRequirements};

    fn claim_spec(access_modes: Option<Vec<&str>>) -> PersistentVolumeClaimSpec {
        PersistentVolumeClaimSpec {
            access_modes: access_modes.map(|modes| modes.into_iter().map(String::from).collect()),
            storage_class_name: Some("ceph-fs".into()),
            resources: Some(VolumeResourceRequirements {
                requests: Some(
                    [("storage".to_string(), Quantity("100Gi".into()))]
                        .into_iter()
                        .collect(),
                ),
                ..VolumeResourceRequirements::default()
            }),
            ..PersistentVolumeClaimSpec::default()
        }
    }

    fn spec_with_cache(cache: Option<CacheConfig>) -> ModelExpressServerSpec {
        ModelExpressServerSpec {
            image: Some("img".into()),
            replicas: 1,
            metadata_backend: MetadataBackend::Kubernetes {},
            port: 8001,
            log: None,
            cache,
            security: None,
            tls: None,
            reaper: None,
            credentials: None,
            pod_metadata: None,
            service_metadata: None,
            resources: None,
            node_selector: None,
            tolerations: None,
            affinity: None,
            network_policy: None,
            service_account_name: None,
            image_pull_secrets: None,
        }
    }

    #[test]
    fn default_is_empty_dir_at_the_default_path() {
        let cv = render_cache_volume("mx", &spec_with_cache(None));
        assert_eq!(cv.mount.mount_path, DEFAULT_MOUNT_PATH);
        assert_ne!(
            cv.mount.mount_path, "/root",
            "a non-root UID cannot write HOME"
        );
        assert_eq!(cv.mount.name, VOLUME_NAME);
        assert!(cv.volume.empty_dir.is_some());
        assert!(cv.pvc.is_none());
    }

    #[test]
    fn directory_overrides_mount_path() {
        let cv = render_cache_volume(
            "mx",
            &spec_with_cache(Some(CacheConfig {
                directory: Some("/cache".into()),
                ..CacheConfig::default()
            })),
        );
        assert_eq!(cv.mount.mount_path, "/cache");
    }

    #[test]
    fn empty_dir_size_limit_is_applied() {
        let cv = render_cache_volume(
            "mx",
            &spec_with_cache(Some(CacheConfig {
                storage: Some(CacheStorage::EmptyDir(EmptyDirStorage {
                    size_limit: Some("200Gi".into()),
                })),
                ..CacheConfig::default()
            })),
        );
        let limit = cv.volume.empty_dir.expect("emptyDir").size_limit;
        assert_eq!(limit, Some(Quantity("200Gi".into())));
    }

    #[test]
    fn existing_claim_mounts_without_creating_pvc() {
        let cv = render_cache_volume(
            "mx",
            &spec_with_cache(Some(CacheConfig {
                storage: Some(CacheStorage::ExistingClaim(ExistingClaimStorage {
                    claim_name: "shared-models".into(),
                })),
                ..CacheConfig::default()
            })),
        );
        let claim = cv
            .volume
            .persistent_volume_claim
            .expect("pvc volume source");
        assert_eq!(claim.claim_name, "shared-models");
        assert!(cv.pvc.is_none());
    }

    #[test]
    fn managed_pvc_renders_claim_and_volume() {
        let cv = render_cache_volume(
            "mx",
            &spec_with_cache(Some(CacheConfig {
                storage: Some(CacheStorage::Pvc(Box::new(ManagedPvcStorage {
                    metadata: None,
                    spec: claim_spec(Some(vec!["ReadWriteMany"])),
                }))),
                ..CacheConfig::default()
            })),
        );
        let pvc = cv.pvc.expect("managed PVC");
        assert_eq!(pvc.metadata.name.as_deref(), Some("mx-model-cache"));
        let pvc_spec = pvc.spec.expect("pvc spec");
        assert_eq!(
            pvc_spec.access_modes,
            Some(vec!["ReadWriteMany".to_string()])
        );
        assert_eq!(pvc_spec.storage_class_name.as_deref(), Some("ceph-fs"));
        let size = pvc_spec
            .resources
            .and_then(|r| r.requests)
            .and_then(|mut r| r.remove("storage"));
        assert_eq!(size, Some(Quantity("100Gi".into())));
        assert_eq!(
            cv.volume
                .persistent_volume_claim
                .expect("source")
                .claim_name,
            "mx-model-cache"
        );
    }

    #[test]
    fn managed_pvc_defaults_to_rwo() {
        let cv = render_cache_volume(
            "mx",
            &spec_with_cache(Some(CacheConfig {
                storage: Some(CacheStorage::Pvc(Box::new(ManagedPvcStorage {
                    metadata: None,
                    spec: claim_spec(None),
                }))),
                ..CacheConfig::default()
            })),
        );
        let modes = cv.pvc.expect("pvc").spec.expect("spec").access_modes;
        assert_eq!(modes, Some(vec!["ReadWriteOnce".to_string()]));
    }

    #[test]
    fn managed_pvc_carries_labels_and_annotations() {
        let cv = render_cache_volume(
            "mx",
            &spec_with_cache(Some(CacheConfig {
                storage: Some(CacheStorage::Pvc(Box::new(ManagedPvcStorage {
                    metadata: Some(MetadataOverrides {
                        labels: Some(
                            [("team".to_string(), "llm-d".to_string())]
                                .into_iter()
                                .collect(),
                        ),
                        annotations: Some(
                            [("backup".to_string(), "false".to_string())]
                                .into_iter()
                                .collect(),
                        ),
                    }),
                    spec: claim_spec(Some(vec!["ReadWriteMany"])),
                }))),
                ..CacheConfig::default()
            })),
        );
        let meta = cv.pvc.expect("pvc").metadata;
        assert_eq!(
            meta.labels.expect("labels").get("team").map(String::as_str),
            Some("llm-d")
        );
        assert_eq!(
            meta.annotations
                .expect("annotations")
                .get("backup")
                .map(String::as_str),
            Some("false")
        );
    }
}
