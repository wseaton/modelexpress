// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ServiceAccount and namespace RBAC for the server pod.
//!
//! The kubernetes metadata backend stores state in modelexpress.nvidia.com
//! CRs plus tensor-descriptor ConfigMaps; without these grants the server
//! exits at startup with a 403. Rules mirror upstream's
//! ci/k8s/server/rbac-modelmetadata.yaml.

use crate::crd::{AuthMode, MetadataBackend, ModelExpressServerSpec};
use crate::labels::managed_labels;
use k8s_openapi::api::core::v1::ServiceAccount;
use k8s_openapi::api::rbac::v1::{
    ClusterRoleBinding, PolicyRule, Role, RoleBinding, RoleRef, Subject,
};
use kube::api::ObjectMeta;

pub const UPSTREAM_API_GROUP: &str = "modelexpress.nvidia.com";

/// Built-in ClusterRole granting tokenreviews and subjectaccessreviews create.
pub const AUTH_DELEGATOR_CLUSTER_ROLE: &str = "system:auth-delegator";

/// SA the server pod runs as. Users can bring their own via
/// spec.serviceAccountName; then nothing here is created and binding the
/// upstream Role to it is their job.
pub fn service_account_name(cr_name: &str, spec: &ModelExpressServerSpec) -> String {
    spec.service_account_name
        .clone()
        .unwrap_or_else(|| format!("{cr_name}-server"))
}

pub fn role_name(cr_name: &str) -> String {
    format!("{cr_name}-metadata")
}

/// ClusterRoleBindings share one namespace-less name space, so the name
/// carries the CR's namespace and its name. Joining them with a separator
/// they may both contain is ambiguous (team-a/mx and team/a-mx read alike),
/// and either can be long enough to overrun a name, so the readable part is
/// truncated and a digest of the pair decides the name.
pub fn auth_delegator_binding_name(cr_name: &str, ns: &str) -> String {
    /// Leaves room for the suffix inside the 253 a name allows.
    const READABLE: usize = 100;
    const DIGEST: usize = 16;

    let digest = crate::digest::hex(&crate::digest::sha256(format!("{ns}/{cr_name}").as_bytes()));
    let mut readable = format!("modelexpress-{ns}-{cr_name}");
    readable.truncate(READABLE);
    let readable = readable.trim_end_matches('-');
    let digest = digest.get(..DIGEST).unwrap_or(digest.as_str());
    format!("{readable}-auth-delegator-{digest}")
}

/// Enforce mode validates client tokens with TokenReview, which is a
/// cluster-scoped API: a namespaced Role cannot grant it. Returns None for
/// every other mode, and the caller then deletes a binding left over from a
/// CR that used to enforce.
pub fn render_auth_delegator_binding(
    cr_name: &str,
    ns: &str,
    spec: &ModelExpressServerSpec,
) -> Option<ClusterRoleBinding> {
    if spec.security.as_ref().map(|s| s.mode) != Some(AuthMode::Enforce) {
        return None;
    }
    Some(ClusterRoleBinding {
        metadata: ObjectMeta {
            name: Some(auth_delegator_binding_name(cr_name, ns)),
            labels: Some(managed_labels(cr_name)),
            ..ObjectMeta::default()
        },
        role_ref: RoleRef {
            api_group: "rbac.authorization.k8s.io".to_string(),
            kind: "ClusterRole".to_string(),
            name: AUTH_DELEGATOR_CLUSTER_ROLE.to_string(),
        },
        subjects: Some(vec![Subject {
            kind: "ServiceAccount".to_string(),
            name: service_account_name(cr_name, spec),
            namespace: Some(ns.to_string()),
            ..Subject::default()
        }]),
    })
}

pub struct ServerRbac {
    pub service_account: Option<ServiceAccount>,
    pub role: Option<Role>,
    pub role_binding: Option<RoleBinding>,
}

/// The verbs the server needs on the upstream CRs and ConfigMaps. Public so
/// the operator's own ClusterRole can include them: RBAC escalation
/// prevention means the operator can only grant what it holds.
pub fn server_policy_rules() -> Vec<PolicyRule> {
    let crud = vec![
        "get".to_string(),
        "list".to_string(),
        "watch".to_string(),
        "create".to_string(),
        "update".to_string(),
        "patch".to_string(),
        "delete".to_string(),
    ];
    let status = vec!["get".to_string(), "update".to_string(), "patch".to_string()];
    vec![
        PolicyRule {
            api_groups: Some(vec![UPSTREAM_API_GROUP.to_string()]),
            resources: Some(vec![
                "modelmetadatas".to_string(),
                "modelcacheentries".to_string(),
            ]),
            verbs: crud.clone(),
            ..PolicyRule::default()
        },
        PolicyRule {
            api_groups: Some(vec![UPSTREAM_API_GROUP.to_string()]),
            resources: Some(vec![
                "modelmetadatas/status".to_string(),
                "modelcacheentries/status".to_string(),
            ]),
            verbs: status,
            ..PolicyRule::default()
        },
        // blockOwnerDeletion on a tensor-descriptor ConfigMap's ownerReference
        // is an ownership write against the owning CR, so the API server also
        // requires update on that CR's finalizers subresource.
        PolicyRule {
            api_groups: Some(vec![UPSTREAM_API_GROUP.to_string()]),
            resources: Some(vec![
                "modelmetadatas/finalizers".to_string(),
                "modelcacheentries/finalizers".to_string(),
            ]),
            verbs: vec!["update".to_string()],
            ..PolicyRule::default()
        },
        // Tensor descriptors are stored as ConfigMaps; upstream needs the
        // whole namespace, so run MX in a dedicated one on shared clusters.
        PolicyRule {
            api_groups: Some(vec![String::new()]),
            resources: Some(vec!["configmaps".to_string()]),
            verbs: crud,
            ..PolicyRule::default()
        },
    ]
}

pub fn render_rbac(cr_name: &str, spec: &ModelExpressServerSpec) -> ServerRbac {
    if spec.service_account_name.is_some() {
        return ServerRbac {
            service_account: None,
            role: None,
            role_binding: None,
        };
    }

    let sa_name = service_account_name(cr_name, spec);
    let service_account = Some(ServiceAccount {
        metadata: ObjectMeta {
            name: Some(sa_name.clone()),
            labels: Some(managed_labels(cr_name)),
            ..ObjectMeta::default()
        },
        ..ServiceAccount::default()
    });

    // redis backend keeps state out of the cluster; the SA exists for
    // identity (and future TokenReview auth) but needs no grants
    if !matches!(spec.metadata_backend, MetadataBackend::Kubernetes {}) {
        return ServerRbac {
            service_account,
            role: None,
            role_binding: None,
        };
    }

    let role = Role {
        metadata: ObjectMeta {
            name: Some(role_name(cr_name)),
            labels: Some(managed_labels(cr_name)),
            ..ObjectMeta::default()
        },
        rules: Some(server_policy_rules()),
    };
    let role_binding = RoleBinding {
        metadata: ObjectMeta {
            name: Some(role_name(cr_name)),
            labels: Some(managed_labels(cr_name)),
            ..ObjectMeta::default()
        },
        role_ref: RoleRef {
            api_group: "rbac.authorization.k8s.io".to_string(),
            kind: "Role".to_string(),
            name: role_name(cr_name),
        },
        subjects: Some(vec![Subject {
            kind: "ServiceAccount".to_string(),
            name: sa_name,
            ..Subject::default()
        }]),
    };

    ServerRbac {
        service_account,
        role: Some(role),
        role_binding: Some(role_binding),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use crate::rbac::auth_delegator_binding_name;

    #[test]
    fn binding_names_cannot_collide_across_namespaces() {
        assert_ne!(
            auth_delegator_binding_name("mx", "team-a"),
            auth_delegator_binding_name("a-mx", "team")
        );
        assert_eq!(
            auth_delegator_binding_name("mx", "team-a"),
            auth_delegator_binding_name("mx", "team-a"),
            "the name has to be stable for the same CR"
        );
    }

    #[test]
    fn binding_names_stay_within_a_kubernetes_name() {
        let name = auth_delegator_binding_name(&"c".repeat(253), &"n".repeat(63));
        assert!(name.len() <= 253, "{} chars", name.len());
        assert!(
            name.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{name}"
        );
        assert!(!name.contains("--auth-delegator"), "{name}");
    }

    use super::*;
    use crate::crd::{RedisBackend, SecurityConfig};

    fn spec(backend: MetadataBackend, sa: Option<&str>) -> ModelExpressServerSpec {
        ModelExpressServerSpec {
            image: Some("img".into()),
            image_pull_secrets: None,
            replicas: 1,
            metadata_backend: backend,
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
            service_account_name: sa.map(String::from),
        }
    }

    #[test]
    fn kubernetes_backend_gets_sa_role_binding() {
        let rbac = render_rbac("mx", &spec(MetadataBackend::Kubernetes {}, None));
        assert_eq!(
            rbac.service_account.expect("sa").metadata.name.as_deref(),
            Some("mx-server")
        );
        let role = rbac.role.expect("role");
        assert_eq!(role.metadata.name.as_deref(), Some("mx-metadata"));
        let rules = role.rules.expect("rules");
        assert!(rules.iter().any(|r| {
            r.resources
                .as_ref()
                .is_some_and(|res| res.contains(&"modelmetadatas".to_string()))
        }));
        assert!(rules.iter().any(|r| {
            r.resources
                .as_ref()
                .is_some_and(|res| res.contains(&"configmaps".to_string()))
        }));
        let binding = rbac.role_binding.expect("binding");
        assert_eq!(binding.subjects.expect("subjects")[0].name, "mx-server");
    }

    /// P2P metadata publishing sets blockOwnerDeletion on tensor-descriptor
    /// ConfigMaps, which the API server rejects without update on the owning
    /// CRs' finalizers subresource.
    #[test]
    fn finalizers_subresource_gets_update_only() {
        let rule = server_policy_rules()
            .into_iter()
            .find(|r| {
                r.resources
                    .as_ref()
                    .is_some_and(|res| res.contains(&"modelmetadatas/finalizers".to_string()))
            })
            .expect("finalizers rule");
        assert_eq!(
            rule.api_groups.as_deref(),
            Some([UPSTREAM_API_GROUP.to_string()].as_slice())
        );
        assert_eq!(
            rule.resources.as_deref(),
            Some(
                [
                    "modelmetadatas/finalizers".to_string(),
                    "modelcacheentries/finalizers".to_string(),
                ]
                .as_slice()
            )
        );
        assert_eq!(rule.verbs, vec!["update".to_string()]);
    }

    #[test]
    fn redis_backend_gets_sa_only() {
        let rbac = render_rbac(
            "mx",
            &spec(
                MetadataBackend::Redis(RedisBackend {
                    url: Some("redis://r:6379".into()),
                    url_secret: None,
                }),
                None,
            ),
        );
        assert!(rbac.service_account.is_some());
        assert!(rbac.role.is_none());
        assert!(rbac.role_binding.is_none());
    }

    #[test]
    fn enforce_mode_binds_the_server_sa_to_auth_delegator() {
        let mut spec = spec(MetadataBackend::Kubernetes {}, None);
        spec.security = Some(SecurityConfig {
            mode: AuthMode::Enforce,
            token_audiences: vec!["modelexpress".into()],
            allowed_service_accounts: vec![],
            cache_ttl_secs: None,
        });
        let binding = render_auth_delegator_binding("mx", "mx-system", &spec).expect("binding");
        assert_eq!(
            binding.metadata.name.as_deref(),
            Some(auth_delegator_binding_name("mx", "mx-system").as_str())
        );
        assert_eq!(binding.role_ref.kind, "ClusterRole");
        assert_eq!(binding.role_ref.name, AUTH_DELEGATOR_CLUSTER_ROLE);
        let subjects = binding.subjects.expect("subjects");
        assert_eq!(subjects[0].name, "mx-server");
        assert_eq!(subjects[0].namespace.as_deref(), Some("mx-system"));
        assert_eq!(
            binding.metadata.labels.expect("labels"),
            managed_labels("mx"),
            "delete_if_managed proves ownership from these labels alone"
        );
    }

    #[test]
    fn enforce_mode_binds_a_user_supplied_sa() {
        let mut spec = spec(MetadataBackend::Kubernetes {}, Some("my-sa"));
        spec.security = Some(SecurityConfig {
            mode: AuthMode::Enforce,
            token_audiences: vec!["modelexpress".into()],
            allowed_service_accounts: vec![],
            cache_ttl_secs: None,
        });
        let binding = render_auth_delegator_binding("mx", "mx-system", &spec).expect("binding");
        assert_eq!(binding.subjects.expect("subjects")[0].name, "my-sa");
    }

    #[test]
    fn non_enforce_modes_get_no_cluster_binding() {
        let mut spec = spec(MetadataBackend::Kubernetes {}, None);
        assert!(render_auth_delegator_binding("mx", "mx-system", &spec).is_none());
        spec.security = Some(SecurityConfig {
            mode: AuthMode::Disabled,
            token_audiences: vec![],
            allowed_service_accounts: vec![],
            cache_ttl_secs: None,
        });
        assert!(render_auth_delegator_binding("mx", "mx-system", &spec).is_none());
    }

    #[test]
    fn binding_names_are_unique_per_namespace() {
        assert_ne!(
            auth_delegator_binding_name("mx", "team-a"),
            auth_delegator_binding_name("mx", "team-b")
        );
    }

    #[test]
    fn user_supplied_sa_disables_generation() {
        let rbac = render_rbac("mx", &spec(MetadataBackend::Kubernetes {}, Some("my-sa")));
        assert!(rbac.service_account.is_none());
        assert!(rbac.role.is_none());
        assert!(rbac.role_binding.is_none());
        assert_eq!(
            service_account_name("mx", &spec(MetadataBackend::Kubernetes {}, Some("my-sa"))),
            "my-sa"
        );
    }
}
