// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! config/manifests/openshift: the operator as deployed on OpenShift.

use crate::objects::{METRICS_SERVICE_NAME, NAME, PARAMS_CONFIGMAP, SERVER_IMAGE_PARAM, labels};
use modelexpress_operator::telemetry;
use modelexpress_operator_openshift::images::SERVER_IMAGE_ENV;
use modelexpress_operator_openshift::servicemonitor;
use serde_json::json;

/// Lets the operator read the cluster TLS profile and keep its own
/// ServiceMonitor, neither of which the base manifests need.
pub const OPENSHIFT_ROLE: &str = "modelexpress-operator-openshift";

pub const METRICS_TLS_SECRET: &str = "modelexpress-operator-metrics-tls";
pub const METRICS_TLS_MOUNT: &str = "/etc/modelexpress-operator/metrics-tls";
pub const SERVICE_CA_CONFIGMAP: &str = "openshift-service-ca.crt";
/// The namespace `config/manifests/default` installs into.
pub const DEFAULT_NAMESPACE: &str = "modelexpress-operator-system";

/// Directory of [`component`] under config/manifests.
pub const COMPONENT_DIR: &str = "components/openshift";
/// Directory of [`related_image_component`] under config/manifests.
pub const RELATED_IMAGE_COMPONENT_DIR: &str = "components/related-image";

/// The env var [`server_image_replacement`] fills in. Without it the operator
/// has no default image for a CR that leaves spec.image unset.
fn server_image_env() -> serde_json::Value {
    json!({"name": SERVER_IMAGE_ENV, "value": "set from params.env"})
}

/// Kustomize Component: only the default server image, for a platform install
/// on a cluster with none of the OpenShift APIs [`component`] depends on.
pub fn related_image_component() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "kustomization.yaml",
            json!({
                "apiVersion": "kustomize.config.k8s.io/v1alpha1",
                "kind": "Component",
                "patches": [
                    {"path": "deployment-patch.yaml", "target": {"kind": "Deployment", "name": NAME}},
                ],
            }),
        ),
        (
            "deployment-patch.yaml",
            json!({
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": NAME},
                "spec": {"template": {"spec": {"containers": [{
                    "name": "operator",
                    "env": [server_image_env()],
                }]}}},
            }),
        ),
    ]
}

/// Sets the server image the operator defaults CRs to, from the base params
/// ConfigMap. It lives in the overlays because the env entry it fills in only
/// exists once a component has added it.
pub fn server_image_replacement() -> serde_json::Value {
    json!({
        "source": {"kind": "ConfigMap", "name": PARAMS_CONFIGMAP, "fieldPath": format!("data.{SERVER_IMAGE_PARAM}")},
        "targets": [{
            "select": {"kind": "Deployment", "name": NAME},
            "fieldPaths": [format!("spec.template.spec.containers.[name=operator].env.[name={SERVER_IMAGE_ENV}].value")],
        }],
    })
}

/// OpenShift overlay: the default install plus [`component`], in a namespace
/// cluster monitoring scrapes.
pub fn overlay() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "kustomization.yaml",
            json!({
                "apiVersion": "kustomize.config.k8s.io/v1beta1",
                "kind": "Kustomization",
                "namespace": DEFAULT_NAMESPACE,
                "resources": ["../default"],
                "components": [format!("../{COMPONENT_DIR}")],
                "replacements": [server_image_replacement()],
                "patches": [
                    {"path": "namespace-patch.yaml", "target": {"kind": "Namespace", "name": DEFAULT_NAMESPACE}},
                ],
            }),
        ),
        (
            "namespace-patch.yaml",
            json!({
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": DEFAULT_NAMESPACE,
                    "labels": {"openshift.io/cluster-monitoring": "true"},
                },
            }),
        ),
    ]
}

/// Kustomize Component: strategic-merge patches and extra objects that turn
/// the metrics endpoint into service-ca TLS, and the RBAC the operator needs
/// to read the cluster TLS profile. Namespace-agnostic, so every overlay that
/// targets OpenShift shares it.
pub fn component() -> Vec<(&'static str, serde_json::Value)> {
    vec![
        (
            "kustomization.yaml",
            json!({
                "apiVersion": "kustomize.config.k8s.io/v1alpha1",
                "kind": "Component",
                "resources": [
                    "openshift-clusterrole.yaml",
                    "openshift-clusterrolebinding.yaml",
                    "service-ca-configmap.yaml",
                ],
                "patches": [
                    {"path": "deployment-patch.yaml", "target": {"kind": "Deployment", "name": NAME}},
                    {"path": "service-patch.yaml", "target": {"kind": "Service", "name": METRICS_SERVICE_NAME}},
                ],
            }),
        ),
        (
            "deployment-patch.yaml",
            json!({
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": NAME},
                "spec": {"template": {
                    "metadata": {"annotations": {"prometheus.io/port": telemetry::METRICS_TLS_PORT.to_string(), "prometheus.io/scheme": "https"}},
                    "spec": {
                        "containers": [{
                            "name": "operator",
                            "env": [
                                {"name": telemetry::METRICS_TLS_DIR_ENV, "value": METRICS_TLS_MOUNT},
                                server_image_env(),
                                {
                                    "name": servicemonitor::NAMESPACE_ENV,
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
                                },
                            ],
                            "ports": [
                                {"name": telemetry::HEALTH_PORT_NAME, "containerPort": telemetry::HEALTH_PORT},
                                {"name": telemetry::METRICS_TLS_PORT_NAME, "containerPort": telemetry::METRICS_TLS_PORT},
                            ],
                            "volumeMounts": [{"name": "metrics-tls", "mountPath": METRICS_TLS_MOUNT, "readOnly": true}],
                        }],
                        "volumes": [{"name": "metrics-tls", "secret": {"secretName": METRICS_TLS_SECRET}}],
                    },
                }},
            }),
        ),
        (
            "service-patch.yaml",
            json!({
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": METRICS_SERVICE_NAME,
                    "annotations": {"service.beta.openshift.io/serving-cert-secret-name": METRICS_TLS_SECRET},
                },
                "spec": {"ports": [{
                    "name": telemetry::METRICS_TLS_PORT_NAME,
                    "port": telemetry::METRICS_TLS_PORT,
                    "targetPort": telemetry::METRICS_TLS_PORT_NAME,
                }]},
            }),
        ),
        (
            "service-ca-configmap.yaml",
            json!({
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": SERVICE_CA_CONFIGMAP,
                    "labels": labels(),
                    "annotations": {"service.beta.openshift.io/inject-cabundle": "true"},
                },
            }),
        ),
        (
            "openshift-clusterrole.yaml",
            json!({
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRole",
                "metadata": {"name": OPENSHIFT_ROLE, "labels": labels()},
                "rules": [
                    {
                        "apiGroups": ["config.openshift.io"],
                        "resources": ["apiservers"],
                        "verbs": ["get", "list", "watch"],
                    },
                    {
                        "apiGroups": [servicemonitor::API_GROUP],
                        "resources": [servicemonitor::PLURAL],
                        "verbs": ["get", "create", "patch", "update"],
                    },
                ],
            }),
        ),
        (
            "openshift-clusterrolebinding.yaml",
            json!({
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {"name": OPENSHIFT_ROLE, "labels": labels()},
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": OPENSHIFT_ROLE,
                },
                // Namespace unset so kustomize resolves the ServiceAccount by
                // nameReference, as the base ClusterRoleBinding does.
                "subjects": [{"kind": "ServiceAccount", "name": NAME}],
            }),
        ),
    ]
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use crate::openshift::{COMPONENT_DIR, component, overlay, related_image_component};
    use modelexpress_operator_openshift::images::SERVER_IMAGE_ENV;
    use std::collections::BTreeSet;

    /// Files a kustomization names through `resources` and `patches`, minus
    /// directories.
    fn referenced(kustomization: &serde_json::Value) -> BTreeSet<String> {
        let resources = kustomization["resources"].as_array().into_iter().flatten();
        let patches = kustomization["patches"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|patch| &patch["path"]);
        resources
            .chain(patches)
            .map(|path| path.as_str().expect("path is a string").to_string())
            .filter(|path| path.ends_with(".yaml"))
            .collect()
    }

    fn assert_emits_what_it_references(files: Vec<(&'static str, serde_json::Value)>) {
        let kustomization = files
            .iter()
            .find(|(file, _)| *file == "kustomization.yaml")
            .expect("has a kustomization")
            .1
            .clone();
        let emitted: BTreeSet<String> = files
            .iter()
            .map(|(file, _)| (*file).to_string())
            .filter(|file| file != "kustomization.yaml")
            .collect();
        assert_eq!(referenced(&kustomization), emitted);
    }

    #[test]
    fn component_emits_exactly_the_files_it_references() {
        assert_emits_what_it_references(component());
    }

    #[test]
    fn related_image_component_emits_exactly_the_files_it_references() {
        assert_emits_what_it_references(related_image_component());
    }

    /// server_image_replacement targets this env entry by name in whichever
    /// component the overlay pulled in; kustomize fails the build if it is
    /// missing from either.
    #[test]
    fn both_components_declare_the_server_image_env() {
        for files in [component(), related_image_component()] {
            let patch = &files
                .iter()
                .find(|(file, _)| *file == "deployment-patch.yaml")
                .expect("has a deployment patch")
                .1;
            let env = patch["spec"]["template"]["spec"]["containers"][0]["env"]
                .as_array()
                .expect("env list");
            assert_eq!(
                env.iter()
                    .filter(|entry| entry["name"] == SERVER_IMAGE_ENV)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn overlay_emits_exactly_the_files_it_references() {
        assert_emits_what_it_references(overlay());
    }

    #[test]
    fn component_is_a_namespace_agnostic_kustomize_component() {
        let files = component();
        let kustomization = &files[0].1;
        assert_eq!(kustomization["kind"], "Component");
        assert_eq!(
            kustomization["apiVersion"],
            "kustomize.config.k8s.io/v1alpha1"
        );
        assert!(kustomization.get("namespace").is_none());
        for (file, value) in &files {
            assert!(
                value["metadata"].get("namespace").is_none(),
                "{file} pins metadata.namespace"
            );
        }
    }

    #[test]
    fn overlay_pulls_in_the_component() {
        let files = overlay();
        assert_eq!(
            files[0].1["components"],
            serde_json::json!([format!("../{COMPONENT_DIR}")])
        );
    }
}
