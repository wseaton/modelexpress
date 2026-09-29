// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Renders the Deployment and Service for a ModelExpressServer CR.

use crate::crd::ModelExpressServerSpec;
use crate::env::render_env;
use crate::labels::{managed_labels, selector_labels};
use crate::tls::{MOUNT_PATH, TlsSettings, resolve};
use crate::volume::{CacheVolume, render_cache_volume};
use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec, DeploymentStrategy};
use k8s_openapi::api::core::v1::{
    Capabilities, Container, ContainerPort, GRPCAction, PersistentVolumeClaim, PodSecurityContext,
    PodSpec, PodTemplateSpec, Probe, ResourceRequirements, SeccompProfile, SecretVolumeSource,
    SecurityContext, Service, ServicePort, ServiceSpec, TCPSocketAction, Volume, VolumeMount,
};
use k8s_openapi::api::networking::v1::{
    NetworkPolicy, NetworkPolicyIngressRule, NetworkPolicyPort, NetworkPolicySpec,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::LabelSelector;
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use kube::api::ObjectMeta;
use std::collections::BTreeMap;

pub const CONTAINER_NAME: &str = "server";
pub const PORT_NAME: &str = "grpc";
pub const TLS_VOLUME_NAME: &str = "tls";

/// Everything the reconciler applies for one CR. ownerReferences and
/// namespaces are its job, not the renderer's.
pub struct DesiredState {
    pub deployment: Deployment,
    pub service: Service,
    pub pvc: Option<PersistentVolumeClaim>,
    /// None means "no policy desired": the reconciler deletes any stale one.
    pub network_policy: Option<NetworkPolicy>,
}

/// `tls_defaults` fill in what `spec.tls` leaves unset; they only matter when
/// `spec.tls` is set.
/// `image` is the server image the reconciler resolved from `spec.image` and
/// the operator's default.
pub fn render(
    cr_name: &str,
    spec: &ModelExpressServerSpec,
    image: &str,
    tls_defaults: &TlsSettings,
) -> DesiredState {
    let CacheVolume { volume, mount, pvc } = render_cache_volume(cr_name, spec);
    let strategy = rollout_strategy(&volume);
    let labels = pod_labels(cr_name, spec);
    let pod_annotations = spec
        .pod_metadata
        .as_ref()
        .and_then(|meta| meta.annotations.clone());
    let tls = spec
        .tls
        .as_ref()
        .map(|config| resolve(config, tls_defaults));

    // The server wires tonic_health, so use real gRPC probes instead of the
    // chart's TCP socket checks. Liveness is deliberately laxer than
    // readiness: a slow backend should pull the pod from rotation, not
    // restart it. kubelet's gRPC probe is plaintext only, so a TLS listener
    // gets TCP probes instead.
    let (readiness, liveness) = if tls.is_some() {
        (tcp_probe(spec.port, 5, 10), tcp_probe(spec.port, 15, 30))
    } else {
        (grpc_probe(spec.port, 5, 10), grpc_probe(spec.port, 15, 30))
    };

    let mut volume_mounts = vec![mount];
    let mut volumes = vec![volume];
    if let Some(tls) = &tls {
        volume_mounts.push(VolumeMount {
            name: TLS_VOLUME_NAME.to_string(),
            mount_path: MOUNT_PATH.to_string(),
            read_only: Some(true),
            ..VolumeMount::default()
        });
        volumes.push(Volume {
            name: TLS_VOLUME_NAME.to_string(),
            secret: Some(SecretVolumeSource {
                secret_name: Some(tls.secret_name.clone()),
                ..SecretVolumeSource::default()
            }),
            ..Volume::default()
        });
    }

    let container = Container {
        name: CONTAINER_NAME.to_string(),
        image: Some(image.to_string()),
        ports: Some(vec![ContainerPort {
            name: Some(PORT_NAME.to_string()),
            container_port: spec.port,
            ..ContainerPort::default()
        }]),
        env: Some(render_env(spec, tls.as_ref())),
        volume_mounts: Some(volume_mounts),
        readiness_probe: Some(readiness),
        liveness_probe: Some(liveness),
        security_context: Some(container_security_context()),
        resources: Some(spec.resources.clone().unwrap_or_else(default_resources)),
        ..Container::default()
    };

    let deployment = Deployment {
        metadata: ObjectMeta {
            name: Some(cr_name.to_string()),
            labels: Some(labels.clone()),
            ..ObjectMeta::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(spec.replicas),
            strategy: Some(strategy),
            selector: LabelSelector {
                match_labels: Some(selector_labels(cr_name)),
                ..LabelSelector::default()
            },
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels.clone()),
                    annotations: pod_annotations,
                    ..ObjectMeta::default()
                }),
                spec: Some(PodSpec {
                    containers: vec![container],
                    security_context: Some(pod_security_context()),
                    volumes: Some(volumes),
                    service_account_name: Some(crate::rbac::service_account_name(cr_name, spec)),
                    image_pull_secrets: spec.image_pull_secrets.clone(),
                    node_selector: spec.node_selector.clone(),
                    tolerations: spec.tolerations.clone(),
                    affinity: spec.affinity.clone(),
                    ..PodSpec::default()
                }),
            },
            ..DeploymentSpec::default()
        }),
        status: None,
    };

    let service = Service {
        metadata: ObjectMeta {
            name: Some(cr_name.to_string()),
            labels: Some(service_labels(cr_name, spec)),
            annotations: spec
                .service_metadata
                .as_ref()
                .and_then(|meta| meta.annotations.clone()),
            ..ObjectMeta::default()
        },
        spec: Some(ServiceSpec {
            selector: Some(selector_labels(cr_name)),
            ports: Some(vec![ServicePort {
                name: Some(PORT_NAME.to_string()),
                port: spec.port,
                target_port: Some(IntOrString::String(PORT_NAME.to_string())),
                ..ServicePort::default()
            }]),
            ..ServiceSpec::default()
        }),
        status: None,
    };

    let network_policy = spec.network_policy.as_ref().map(|np| NetworkPolicy {
        metadata: ObjectMeta {
            name: Some(cr_name.to_string()),
            labels: Some(pod_labels(cr_name, spec)),
            ..ObjectMeta::default()
        },
        spec: Some(NetworkPolicySpec {
            pod_selector: LabelSelector {
                match_labels: Some(selector_labels(cr_name)),
                ..LabelSelector::default()
            },
            policy_types: Some(vec!["Ingress".to_string()]),
            ingress: Some(vec![NetworkPolicyIngressRule {
                from: Some(np.allow_from.clone()),
                ports: Some(vec![NetworkPolicyPort {
                    port: Some(IntOrString::Int(spec.port)),
                    protocol: Some("TCP".to_string()),
                    ..NetworkPolicyPort::default()
                }]),
            }]),
            ..NetworkPolicySpec::default()
        }),
    });

    DesiredState {
        deployment,
        service,
        pvc,
        network_policy,
    }
}

/// User podMetadata labels first, operator labels on top: the selector subset
/// must never be overridable or the Deployment orphans its pods.
fn pod_labels(cr_name: &str, spec: &ModelExpressServerSpec) -> BTreeMap<String, String> {
    let mut labels = spec
        .pod_metadata
        .as_ref()
        .and_then(|meta| meta.labels.clone())
        .unwrap_or_default();
    labels.extend(managed_labels(cr_name));
    labels
}

/// Pod labels, then serviceMetadata labels, then operator labels on top.
fn service_labels(cr_name: &str, spec: &ModelExpressServerSpec) -> BTreeMap<String, String> {
    let mut labels = pod_labels(cr_name, spec);
    if let Some(extra) = spec
        .service_metadata
        .as_ref()
        .and_then(|meta| meta.labels.as_ref())
    {
        labels.extend(extra.clone());
    }
    labels.extend(managed_labels(cr_name));
    labels
}

/// runAsUser and fsGroup are deliberately absent: OpenShift's restricted-v2
/// SCC assigns both from the namespace range, and pinning them would conflict
/// with that while buying nothing on vanilla Kubernetes.
fn pod_security_context() -> PodSecurityContext {
    PodSecurityContext {
        run_as_non_root: Some(true),
        seccomp_profile: Some(SeccompProfile {
            type_: "RuntimeDefault".to_string(),
            localhost_profile: None,
        }),
        ..PodSecurityContext::default()
    }
}

/// No readOnlyRootFilesystem: unlike the controller, the server unpacks
/// downloads and the provider SDKs write scratch outside the cache mount.
fn container_security_context() -> SecurityContext {
    SecurityContext {
        allow_privilege_escalation: Some(false),
        run_as_non_root: Some(true),
        capabilities: Some(Capabilities {
            drop: Some(vec!["ALL".to_string()]),
            add: None,
        }),
        ..SecurityContext::default()
    }
}

/// RollingUpdate at replicas: 1 rounds maxUnavailable down to 0, so the new
/// pod starts before the old one goes away and a ReadWriteOnce claim cannot
/// attach to both. Managed PVCs default to RWO.
fn rollout_strategy(volume: &Volume) -> DeploymentStrategy {
    let type_ = if volume.persistent_volume_claim.is_some() {
        "Recreate"
    } else {
        "RollingUpdate"
    };
    DeploymentStrategy {
        type_: Some(type_.to_string()),
        rolling_update: None,
    }
}

/// Requests only. A CPU limit would throttle downloads, and a memory limit
/// risks OOMKills while unpacking a large model, so neither is guessed here;
/// the point is to leave BestEffort, where the pod is evicted first.
fn default_resources() -> ResourceRequirements {
    ResourceRequirements {
        requests: Some(
            [
                ("cpu".to_string(), Quantity("100m".to_string())),
                ("memory".to_string(), Quantity("256Mi".to_string())),
            ]
            .into_iter()
            .collect(),
        ),
        ..ResourceRequirements::default()
    }
}

fn tcp_probe(port: i32, initial_delay: i32, period: i32) -> Probe {
    Probe {
        tcp_socket: Some(TCPSocketAction {
            port: IntOrString::Int(port),
            host: None,
        }),
        initial_delay_seconds: Some(initial_delay),
        period_seconds: Some(period),
        ..Probe::default()
    }
}

fn grpc_probe(port: i32, initial_delay: i32, period: i32) -> Probe {
    Probe {
        grpc: Some(GRPCAction {
            port,
            service: None,
        }),
        initial_delay_seconds: Some(initial_delay),
        period_seconds: Some(period),
        ..Probe::default()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::crd::{CacheConfig, CacheStorage, ManagedPvcStorage, MetadataBackend, RedisBackend};
    use k8s_openapi::api::core::v1::{
        Affinity, LocalObjectReference, NodeAffinity, NodeSelector, NodeSelectorRequirement,
        NodeSelectorTerm, PersistentVolumeClaimSpec, Toleration, VolumeResourceRequirements,
    };
    use k8s_openapi::apimachinery::pkg::api::resource::Quantity;

    fn base_spec() -> ModelExpressServerSpec {
        ModelExpressServerSpec {
            image: Some("nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0".into()),
            replicas: 2,
            metadata_backend: MetadataBackend::Redis(RedisBackend {
                url: Some("redis://mx-redis:6379".into()),
                url_secret: None,
            }),
            port: 8001,
            log: None,
            cache: None,
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

    fn pod_spec(state: &DesiredState) -> &PodSpec {
        state
            .deployment
            .spec
            .as_ref()
            .expect("deployment spec")
            .template
            .spec
            .as_ref()
            .expect("pod spec")
    }

    fn container(state: &DesiredState) -> &Container {
        &state
            .deployment
            .spec
            .as_ref()
            .expect("deployment spec")
            .template
            .spec
            .as_ref()
            .expect("pod spec")
            .containers[0]
    }

    #[test]
    fn selector_matches_pod_labels() {
        let state = render("mx", &base_spec(), "img", &TlsSettings::default());
        let dep_spec = state.deployment.spec.as_ref().expect("spec");
        let selector = dep_spec
            .selector
            .match_labels
            .as_ref()
            .expect("match labels");
        let pod_labels = dep_spec
            .template
            .metadata
            .as_ref()
            .expect("pod meta")
            .labels
            .as_ref()
            .expect("pod labels");
        for (k, v) in selector {
            assert_eq!(pod_labels.get(k), Some(v), "selector key {k} not on pod");
        }
        let svc_selector = state
            .service
            .spec
            .as_ref()
            .expect("svc spec")
            .selector
            .as_ref()
            .expect("svc selector");
        assert_eq!(svc_selector, selector);
    }

    #[test]
    fn replicas_image_and_port_propagate() {
        let state = render(
            "mx",
            &base_spec(),
            "nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0",
            &TlsSettings::default(),
        );
        assert_eq!(
            state.deployment.spec.as_ref().expect("spec").replicas,
            Some(2)
        );
        let c = container(&state);
        assert_eq!(
            c.image.as_deref(),
            Some("nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.0")
        );
        assert_eq!(c.ports.as_ref().expect("ports")[0].container_port, 8001);
    }

    #[test]
    fn scheduling_fields_reach_the_pod_spec() {
        let mut spec = base_spec();
        spec.node_selector = Some(BTreeMap::from([(
            "kubernetes.io/arch".to_string(),
            "arm64".to_string(),
        )]));
        spec.tolerations = Some(vec![Toleration {
            key: Some("nvidia.com/gpu".into()),
            operator: Some("Exists".into()),
            effect: Some("NoSchedule".into()),
            ..Toleration::default()
        }]);
        spec.affinity = Some(Affinity {
            node_affinity: Some(NodeAffinity {
                required_during_scheduling_ignored_during_execution: Some(NodeSelector {
                    node_selector_terms: vec![NodeSelectorTerm {
                        match_expressions: Some(vec![NodeSelectorRequirement {
                            key: "kubernetes.io/os".into(),
                            operator: "In".into(),
                            values: Some(vec!["linux".into()]),
                        }]),
                        ..NodeSelectorTerm::default()
                    }],
                }),
                ..NodeAffinity::default()
            }),
            ..Affinity::default()
        });

        let state = render("mx", &spec, "img", &TlsSettings::default());
        let pod = pod_spec(&state);
        assert_eq!(
            pod.node_selector
                .as_ref()
                .expect("node selector")
                .get("kubernetes.io/arch")
                .map(String::as_str),
            Some("arm64")
        );
        assert_eq!(
            pod.tolerations.as_ref().expect("tolerations")[0]
                .key
                .as_deref(),
            Some("nvidia.com/gpu")
        );
        assert!(
            pod.affinity
                .as_ref()
                .expect("affinity")
                .node_affinity
                .is_some()
        );
    }

    #[test]
    fn scheduling_fields_are_unset_by_default() {
        let pod_owned = render("mx", &base_spec(), "img", &TlsSettings::default());
        let pod = pod_spec(&pod_owned);
        assert!(pod.node_selector.is_none());
        assert!(pod.tolerations.is_none());
        assert!(pod.affinity.is_none());
    }

    #[test]
    fn image_pull_secrets_reach_the_pod_spec() {
        let mut spec = base_spec();
        spec.image_pull_secrets = Some(vec![LocalObjectReference {
            name: "registry-creds".into(),
        }]);

        let state = render("mx", &spec, "img", &TlsSettings::default());
        let pod = pod_spec(&state);
        let secrets = pod.image_pull_secrets.as_ref().expect("image pull secrets");
        assert_eq!(secrets.len(), 1);
        assert_eq!(secrets[0].name, "registry-creds");
    }

    #[test]
    fn image_pull_secrets_preserve_multiple_entries() {
        let mut spec = base_spec();
        spec.image_pull_secrets = Some(vec![
            LocalObjectReference {
                name: "primary-registry-creds".into(),
            },
            LocalObjectReference {
                name: "mirror-registry-creds".into(),
            },
        ]);

        let state = render("mx", &spec, "img", &TlsSettings::default());
        let pod = pod_spec(&state);
        let secrets = pod.image_pull_secrets.as_ref().expect("image pull secrets");
        assert_eq!(
            secrets
                .iter()
                .map(|secret| secret.name.as_str())
                .collect::<Vec<_>>(),
            ["primary-registry-creds", "mirror-registry-creds"]
        );
    }

    // A caller supplying their own ServiceAccount still gets the pod-level
    // secrets: the two are independent, so neither route alone is a trap.
    #[test]
    fn image_pull_secrets_survive_a_custom_service_account() {
        let mut spec = base_spec();
        spec.service_account_name = Some("byo-sa".into());
        spec.image_pull_secrets = Some(vec![LocalObjectReference {
            name: "registry-creds".into(),
        }]);

        let state = render("mx", &spec, "img", &TlsSettings::default());
        let pod = pod_spec(&state);
        assert_eq!(pod.service_account_name.as_deref(), Some("byo-sa"));
        assert_eq!(
            pod.image_pull_secrets.as_ref().expect("image pull secrets")[0].name,
            "registry-creds"
        );
    }

    #[test]
    fn image_pull_secrets_are_unset_by_default() {
        let pod_owned = render("mx", &base_spec(), "img", &TlsSettings::default());
        assert!(pod_spec(&pod_owned).image_pull_secrets.is_none());
    }

    #[test]
    fn grpc_probes_target_the_server_port() {
        let mut spec = base_spec();
        spec.port = 9000;
        let state = render("mx", &spec, "img", &TlsSettings::default());
        let c = container(&state);
        let readiness = c.readiness_probe.as_ref().expect("readiness");
        let liveness = c.liveness_probe.as_ref().expect("liveness");
        assert_eq!(readiness.grpc.as_ref().expect("grpc").port, 9000);
        assert_eq!(liveness.grpc.as_ref().expect("grpc").port, 9000);
    }

    #[test]
    fn env_and_volume_are_wired_into_the_pod() {
        let state = render("mx", &base_spec(), "img", &TlsSettings::default());
        let c = container(&state);
        let env = c.env.as_ref().expect("env");
        assert!(env.iter().any(|e| e.name == "MX_METADATA_BACKEND"));
        assert_eq!(
            c.volume_mounts.as_ref().expect("mounts")[0].name,
            "model-cache"
        );
        let volumes = state
            .deployment
            .spec
            .as_ref()
            .expect("spec")
            .template
            .spec
            .as_ref()
            .expect("pod")
            .volumes
            .as_ref()
            .expect("volumes");
        assert_eq!(volumes[0].name, "model-cache");
    }

    #[test]
    fn service_targets_named_port() {
        let state = render("mx", &base_spec(), "img", &TlsSettings::default());
        let port = &state
            .service
            .spec
            .as_ref()
            .expect("spec")
            .ports
            .as_ref()
            .expect("ports")[0];
        assert_eq!(port.port, 8001);
        assert_eq!(
            port.target_port,
            Some(IntOrString::String("grpc".to_string()))
        );
    }

    #[test]
    fn service_metadata_merges_but_cannot_override_operator_labels() {
        let mut spec = base_spec();
        spec.service_metadata = Some(crate::crd::MetadataOverrides {
            labels: Some(
                [
                    ("team".to_string(), "inference".to_string()),
                    ("app.kubernetes.io/name".to_string(), "evil".to_string()),
                ]
                .into_iter()
                .collect(),
            ),
            annotations: Some(
                [(
                    "certs.example.com/secret-name".to_string(),
                    "mx-tls".to_string(),
                )]
                .into_iter()
                .collect(),
            ),
        });
        let state = render("mx", &spec, "img", &TlsSettings::default());
        let meta = &state.service.metadata;
        let labels = meta.labels.as_ref().expect("labels");
        assert_eq!(labels.get("team").map(String::as_str), Some("inference"));
        assert_eq!(
            labels.get("app.kubernetes.io/name").map(String::as_str),
            Some("modelexpress-server")
        );
        assert_eq!(
            meta.annotations
                .as_ref()
                .and_then(|a| a.get("certs.example.com/secret-name"))
                .map(String::as_str),
            Some("mx-tls")
        );
        assert_eq!(
            state.service.spec.as_ref().expect("spec").selector,
            Some(selector_labels("mx"))
        );
    }

    #[test]
    fn service_without_service_metadata_has_no_annotations() {
        let state = render("mx", &base_spec(), "img", &TlsSettings::default());
        assert!(state.service.metadata.annotations.is_none());
    }

    #[test]
    fn pod_metadata_merges_but_cannot_override_selector() {
        let mut spec = base_spec();
        spec.pod_metadata = Some(crate::crd::MetadataOverrides {
            labels: Some(
                [
                    ("istio.io/dataplane-mode".to_string(), "ambient".to_string()),
                    // attempt to hijack a selector label; must lose
                    ("app.kubernetes.io/name".to_string(), "evil".to_string()),
                ]
                .into_iter()
                .collect(),
            ),
            annotations: Some(
                [("sidecar.istio.io/inject".to_string(), "false".to_string())]
                    .into_iter()
                    .collect(),
            ),
        });
        let state = render("mx", &spec, "img", &TlsSettings::default());
        let template_meta = state
            .deployment
            .spec
            .as_ref()
            .expect("spec")
            .template
            .metadata
            .as_ref()
            .expect("pod meta");
        let labels = template_meta.labels.as_ref().expect("labels");
        assert_eq!(
            labels.get("istio.io/dataplane-mode").map(String::as_str),
            Some("ambient")
        );
        assert_eq!(
            labels.get("app.kubernetes.io/name").map(String::as_str),
            Some("modelexpress-server")
        );
        assert_eq!(
            template_meta
                .annotations
                .as_ref()
                .expect("annotations")
                .get("sidecar.istio.io/inject")
                .map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn ephemeral_cache_rolls_but_pvc_cache_recreates() {
        let strategy = |spec: &ModelExpressServerSpec| {
            render("mx", spec, "img", &TlsSettings::default())
                .deployment
                .spec
                .expect("spec")
                .strategy
                .expect("strategy")
                .type_
        };

        let mut spec = base_spec();
        spec.replicas = 1;
        assert_eq!(
            strategy(&spec),
            Some("RollingUpdate".to_string()),
            "emptyDir attaches anywhere"
        );

        spec.cache = Some(CacheConfig {
            storage: Some(CacheStorage::Pvc(Box::new(ManagedPvcStorage {
                metadata: None,
                spec: PersistentVolumeClaimSpec::default(),
            }))),
            ..CacheConfig::default()
        });
        assert_eq!(
            strategy(&spec),
            Some("Recreate".to_string()),
            "a RWO claim cannot attach to the new pod while the old one holds it"
        );

        spec.cache = Some(CacheConfig {
            storage: Some(CacheStorage::ExistingClaim(
                crate::crd::ExistingClaimStorage {
                    claim_name: "shared-models".into(),
                },
            )),
            ..CacheConfig::default()
        });
        assert_eq!(strategy(&spec), Some("Recreate".to_string()));
    }

    /// Anything rendered without this label stops producing reconcile events.
    #[test]
    fn every_rendered_object_carries_the_managed_by_label() {
        use crate::labels::{MANAGED_BY, MANAGED_BY_LABEL};
        use k8s_openapi::api::networking::v1::NetworkPolicyPeer;

        let mut spec = base_spec();
        spec.replicas = 1;
        spec.metadata_backend = MetadataBackend::Kubernetes {};
        spec.cache = Some(CacheConfig {
            storage: Some(CacheStorage::Pvc(Box::new(ManagedPvcStorage {
                metadata: None,
                spec: PersistentVolumeClaimSpec::default(),
            }))),
            ..CacheConfig::default()
        });
        spec.network_policy = Some(crate::crd::NetworkPolicyConfig {
            allow_from: vec![NetworkPolicyPeer::default()],
        });

        let state = render("mx", &spec, "img", &TlsSettings::default());
        let rbac = crate::rbac::render_rbac("mx", &spec);

        let mut checked = 0;
        for (what, labels) in [
            ("deployment", &state.deployment.metadata.labels),
            ("service", &state.service.metadata.labels),
            ("pvc", &state.pvc.as_ref().expect("pvc").metadata.labels),
            (
                "networkpolicy",
                &state
                    .network_policy
                    .as_ref()
                    .expect("netpol")
                    .metadata
                    .labels,
            ),
            (
                "serviceaccount",
                &rbac.service_account.as_ref().expect("sa").metadata.labels,
            ),
            ("role", &rbac.role.as_ref().expect("role").metadata.labels),
            (
                "rolebinding",
                &rbac.role_binding.as_ref().expect("binding").metadata.labels,
            ),
        ] {
            assert_eq!(
                labels
                    .as_ref()
                    .and_then(|l| l.get(MANAGED_BY_LABEL))
                    .map(String::as_str),
                Some(MANAGED_BY),
                "{what} would never reach the controller's watch"
            );
            checked += 1;
        }
        assert_eq!(
            checked, 7,
            "a resource kind was added without a label check"
        );

        // the pod template too, so the Deployment's own selector still matches
        let pod_meta = state
            .deployment
            .spec
            .expect("spec")
            .template
            .metadata
            .expect("pod meta");
        assert_eq!(
            pod_meta
                .labels
                .expect("labels")
                .get(MANAGED_BY_LABEL)
                .map(String::as_str),
            Some(MANAGED_BY)
        );
    }

    #[test]
    fn network_policy_absent_by_default() {
        assert!(
            render("mx", &base_spec(), "img", &TlsSettings::default())
                .network_policy
                .is_none()
        );
    }

    #[test]
    fn network_policy_restricts_grpc_port_to_peers() {
        use k8s_openapi::api::networking::v1::NetworkPolicyPeer;
        let mut spec = base_spec();
        spec.port = 9000;
        spec.network_policy = Some(crate::crd::NetworkPolicyConfig {
            allow_from: vec![NetworkPolicyPeer {
                pod_selector: Some(LabelSelector {
                    match_labels: Some(
                        [("role".to_string(), "worker".to_string())]
                            .into_iter()
                            .collect(),
                    ),
                    ..LabelSelector::default()
                }),
                ..NetworkPolicyPeer::default()
            }],
        });
        let netpol = render("mx", &spec, "img", &TlsSettings::default())
            .network_policy
            .expect("netpol");
        let np_spec = netpol.spec.expect("spec");
        assert_eq!(
            np_spec
                .pod_selector
                .match_labels
                .expect("labels")
                .get("app.kubernetes.io/instance")
                .map(String::as_str),
            Some("mx")
        );
        assert_eq!(np_spec.policy_types, Some(vec!["Ingress".to_string()]));
        let rule = &np_spec.ingress.expect("ingress")[0];
        assert_eq!(rule.from.as_ref().expect("from").len(), 1);
        let port = &rule.ports.as_ref().expect("ports")[0];
        assert_eq!(port.port, Some(IntOrString::Int(9000)));
    }

    #[test]
    fn managed_pvc_flows_through_desired_state() {
        let mut spec = base_spec();
        spec.replicas = 1;
        spec.cache = Some(CacheConfig {
            storage: Some(CacheStorage::Pvc(Box::new(ManagedPvcStorage {
                metadata: None,
                spec: PersistentVolumeClaimSpec {
                    resources: Some(VolumeResourceRequirements {
                        requests: Some(
                            [("storage".to_string(), Quantity("50Gi".into()))]
                                .into_iter()
                                .collect(),
                        ),
                        ..VolumeResourceRequirements::default()
                    }),
                    ..PersistentVolumeClaimSpec::default()
                },
            }))),
            ..CacheConfig::default()
        });
        let state = render("mx", &spec, "img", &TlsSettings::default());
        let pvc = state.pvc.expect("pvc");
        assert_eq!(pvc.metadata.name.as_deref(), Some("mx-model-cache"));
    }
}
