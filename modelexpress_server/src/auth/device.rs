// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device-possession check. Two signals matched as an OR:
//!
//! - **Device plugin** (`device_resources`, e.g. `rdma/ib`): scan the pod's container
//!   resource requests/limits.
//! - **DRA** (`device_classes`): resolve the pod's `resourceClaims` to `ResourceClaim`
//!   objects and match each request's `deviceClassName` against the configured classes.
//!
//! Pure inspection over `Pod` / `ResourceClaim` objects already cached by `store.rs`.

use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::resource::v1beta1::ResourceClaim;

/// A device-plugin resource name that proves fabric possession, e.g. `rdma/ib`,
/// `rdma/roce`, `vpc.amazonaws.com/efa`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceResource(pub(crate) String);

/// Does this pod request any of the configured device resources? Scans regular and
/// init containers, in both `requests` and `limits`, treating a zero/empty quantity as
/// "not requested".
pub(crate) fn pod_requests_device(pod: &Pod, devices: &[DeviceResource]) -> bool {
    let Some(spec) = pod.spec.as_ref() else {
        return false;
    };
    let init_containers = spec.init_containers.iter().flatten();
    spec.containers
        .iter()
        .chain(init_containers)
        .any(|container| {
            let Some(resources) = container.resources.as_ref() else {
                return false;
            };
            requests_any(resources.requests.as_ref(), devices)
                || requests_any(resources.limits.as_ref(), devices)
        })
}

/// True if the resource map contains a non-zero quantity for any configured device.
fn requests_any(
    map: Option<
        &std::collections::BTreeMap<
            String,
            k8s_openapi::apimachinery::pkg::api::resource::Quantity,
        >,
    >,
    devices: &[DeviceResource],
) -> bool {
    let Some(map) = map else {
        return false;
    };
    devices.iter().any(|device| {
        map.get(&device.0)
            .is_some_and(|quantity| !quantity.0.is_empty() && quantity.0 != "0")
    })
}

/// Resolve the names of the actual `ResourceClaim` objects a pod uses. A pod entry
/// either names a claim directly or via a template, in which case the generated claim
/// name is published in `pod.status.resourceClaimStatuses`.
pub(crate) fn resolve_claim_names(pod: &Pod) -> Vec<String> {
    let Some(spec) = pod.spec.as_ref() else {
        return Vec::new();
    };
    let Some(claims) = spec.resource_claims.as_ref() else {
        return Vec::new();
    };
    let statuses = pod
        .status
        .as_ref()
        .and_then(|status| status.resource_claim_statuses.as_ref());
    let mut names = Vec::new();
    for claim in claims {
        if let Some(direct) = claim.resource_claim_name.as_ref() {
            names.push(direct.clone());
        } else if let Some(statuses) = statuses {
            // Template-generated claim: the real name is in the pod status.
            if let Some(generated) = statuses
                .iter()
                .find(|status| status.name == claim.name)
                .and_then(|status| status.resource_claim_name.as_ref())
            {
                names.push(generated.clone());
            }
        }
    }
    names
}

/// Does this claim request any of the configured DRA device classes?
pub(crate) fn claim_matches_class(claim: &ResourceClaim, classes: &[String]) -> bool {
    let Some(devices) = claim.spec.devices.as_ref() else {
        return false;
    };
    let Some(requests) = devices.requests.as_ref() else {
        return false;
    };
    requests.iter().any(|request| {
        classes
            .iter()
            .any(|class| class == &request.device_class_name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{Container, PodSpec, ResourceRequirements};
    use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
    use std::collections::BTreeMap;

    fn devices() -> Vec<DeviceResource> {
        vec![DeviceResource("rdma/ib".to_string())]
    }

    fn pod_with(containers: Vec<Container>, init: Vec<Container>) -> Pod {
        Pod {
            spec: Some(PodSpec {
                containers,
                init_containers: if init.is_empty() { None } else { Some(init) },
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn container_with(map_key: Option<(&str, &str)>, in_limits: bool) -> Container {
        let resources = map_key.map(|(name, qty)| {
            let mut map = BTreeMap::new();
            map.insert(name.to_string(), Quantity(qty.to_string()));
            if in_limits {
                ResourceRequirements {
                    limits: Some(map),
                    ..Default::default()
                }
            } else {
                ResourceRequirements {
                    requests: Some(map),
                    ..Default::default()
                }
            }
        });
        Container {
            name: "c".to_string(),
            resources,
            ..Default::default()
        }
    }

    #[test]
    fn detects_device_in_container_limit() {
        let pod = pod_with(vec![container_with(Some(("rdma/ib", "1")), true)], vec![]);
        assert!(pod_requests_device(&pod, &devices()));
    }

    #[test]
    fn detects_device_in_init_container_request() {
        let pod = pod_with(
            vec![container_with(None, false)],
            vec![container_with(Some(("rdma/ib", "1")), false)],
        );
        assert!(pod_requests_device(&pod, &devices()));
    }

    #[test]
    fn zero_quantity_is_not_a_request() {
        let pod = pod_with(vec![container_with(Some(("rdma/ib", "0")), false)], vec![]);
        assert!(!pod_requests_device(&pod, &devices()));
    }

    #[test]
    fn unconfigured_resource_name_does_not_match() {
        let pod = pod_with(
            vec![container_with(Some(("nvidia.com/gpu", "1")), true)],
            vec![],
        );
        assert!(!pod_requests_device(&pod, &devices()));
    }

    #[test]
    fn no_resources_means_no_device() {
        let pod = pod_with(vec![container_with(None, false)], vec![]);
        assert!(!pod_requests_device(&pod, &devices()));
    }

    #[test]
    fn no_spec_means_no_device() {
        assert!(!pod_requests_device(&Pod::default(), &devices()));
    }

    #[test]
    fn matches_any_of_several_configured_devices() {
        let configured = vec![
            DeviceResource("rdma/ib".to_string()),
            DeviceResource("rdma/roce".to_string()),
        ];
        let pod = pod_with(vec![container_with(Some(("rdma/roce", "1")), true)], vec![]);
        assert!(pod_requests_device(&pod, &configured));
    }

    // ---- DRA (Dynamic Resource Allocation) ----

    use k8s_openapi::api::core::v1::{PodResourceClaim, PodResourceClaimStatus, PodStatus};
    use k8s_openapi::api::resource::v1beta1::{
        DeviceClaim, DeviceRequest, ResourceClaim, ResourceClaimSpec,
    };

    fn claim_with_classes(names: &[&str]) -> ResourceClaim {
        ResourceClaim {
            spec: ResourceClaimSpec {
                devices: Some(DeviceClaim {
                    requests: Some(
                        names
                            .iter()
                            .map(|name| DeviceRequest {
                                name: format!("req-{name}"),
                                device_class_name: (*name).to_string(),
                                ..Default::default()
                            })
                            .collect(),
                    ),
                    ..Default::default()
                }),
            },
            ..Default::default()
        }
    }

    #[test]
    fn claim_matches_configured_device_class() {
        let claim = claim_with_classes(&["rdma.nvidia.com"]);
        assert!(claim_matches_class(
            &claim,
            &["rdma.nvidia.com".to_string()]
        ));
    }

    #[test]
    fn claim_does_not_match_other_class() {
        let claim = claim_with_classes(&["gpu.nvidia.com"]);
        assert!(!claim_matches_class(
            &claim,
            &["rdma.nvidia.com".to_string()]
        ));
    }

    #[test]
    fn claim_with_no_devices_does_not_match() {
        let claim = ResourceClaim::default();
        assert!(!claim_matches_class(
            &claim,
            &["rdma.nvidia.com".to_string()]
        ));
    }

    #[test]
    fn resolves_directly_named_claim() {
        let pod = Pod {
            spec: Some(PodSpec {
                resource_claims: Some(vec![PodResourceClaim {
                    name: "ib".to_string(),
                    resource_claim_name: Some("my-claim".to_string()),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(resolve_claim_names(&pod), vec!["my-claim".to_string()]);
    }

    #[test]
    fn resolves_template_claim_via_status() {
        let pod = Pod {
            spec: Some(PodSpec {
                resource_claims: Some(vec![PodResourceClaim {
                    name: "ib".to_string(),
                    resource_claim_template_name: Some("ib-template".to_string()),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            status: Some(PodStatus {
                resource_claim_statuses: Some(vec![PodResourceClaimStatus {
                    name: "ib".to_string(),
                    resource_claim_name: Some("ib-generated-abc".to_string()),
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            resolve_claim_names(&pod),
            vec!["ib-generated-abc".to_string()]
        );
    }

    #[test]
    fn resolves_empty_when_no_claims() {
        assert!(resolve_claim_names(&Pod::default()).is_empty());
    }
}
