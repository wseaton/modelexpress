// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! config/manifests/overlays/odh and odh-xks: the operator as a platform
//! operator deploys it on Open Data Hub and RHOAI, on OpenShift and on any
//! other Kubernetes.
//!
//! Neither overlay has a params.env. A platform operator that finds none in
//! an `overlays/<name>` directory rewrites `base/params.env`, and on xKS it
//! only ever rewrites the file the OpenShift overlay resolves to, so both
//! overlays have to read their images from the base.

use crate::openshift::{COMPONENT_DIR, RELATED_IMAGE_COMPONENT_DIR, server_image_replacement};
use serde_json::json;

/// Directory of the OpenShift platform overlay under config/manifests.
pub const OVERLAY_DIR: &str = "overlays/odh";
/// Directory of the xKS platform overlay under config/manifests.
pub const XKS_OVERLAY_DIR: &str = "overlays/odh-xks";

/// ODH overlay for OpenShift: the base plus the OpenShift component.
pub fn overlay() -> Vec<(&'static str, serde_json::Value)> {
    platform_overlay(COMPONENT_DIR)
}

/// ODH overlay for any other Kubernetes: the base plus the default server
/// image, and nothing that needs service-ca or the OpenShift config API.
pub fn xks_overlay() -> Vec<(&'static str, serde_json::Value)> {
    platform_overlay(RELATED_IMAGE_COMPONENT_DIR)
}

/// No namespace and no Namespace object, since the platform installs into
/// its own.
fn platform_overlay(component_dir: &str) -> Vec<(&'static str, serde_json::Value)> {
    vec![(
        "kustomization.yaml",
        json!({
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "resources": ["../../base"],
            "components": [format!("../../{component_dir}")],
            "replacements": [server_image_replacement()],
        }),
    )]
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use crate::objects::{
        DEFAULT_IMAGE, OPERATOR_IMAGE_PARAM, PARAMS_CONFIGMAP, SERVER_IMAGE_PARAM, params_env,
    };
    use crate::odh::{OVERLAY_DIR, XKS_OVERLAY_DIR, overlay, xks_overlay};
    use crate::openshift::{
        COMPONENT_DIR, RELATED_IMAGE_COMPONENT_DIR, component, related_image_component,
    };

    type Files = Vec<(&'static str, serde_json::Value)>;

    /// Each platform overlay with the component it builds on.
    fn platform_overlays() -> [(&'static str, Files, Files); 2] {
        [
            (OVERLAY_DIR, overlay(), component()),
            (XKS_OVERLAY_DIR, xks_overlay(), related_image_component()),
        ]
    }

    fn kustomization_of(files: &Files) -> serde_json::Value {
        files
            .iter()
            .find(|(file, _)| *file == "kustomization.yaml")
            .expect("overlay has a kustomization")
            .1
            .clone()
    }

    #[test]
    fn the_platform_owns_the_namespace() {
        for (name, files, component) in platform_overlays() {
            let k = kustomization_of(&files);
            assert!(
                k.get("namespace").is_none(),
                "{name} must not pin a namespace"
            );
            assert_eq!(
                k["resources"],
                serde_json::json!(["../../base"]),
                "{name}: ../../default would bring a Namespace object along"
            );
            for (file, value) in files.into_iter().chain(component) {
                assert_ne!(value["kind"], "Namespace", "{name}/{file} is a Namespace");
                assert!(
                    value["metadata"].get("namespace").is_none(),
                    "{name}/{file} pins metadata.namespace"
                );
            }
        }
    }

    /// A platform operator falls back to base/params.env only for an overlay
    /// directly under `overlays/` that has no params.env of its own.
    #[test]
    fn overlays_resolve_to_the_base_params() {
        for (name, files, _) in platform_overlays() {
            let (parent, leaf) = name.split_once('/').expect("two path segments");
            assert_eq!(
                parent, "overlays",
                "{name} must sit directly under overlays/"
            );
            assert!(!leaf.contains('/'));
            assert_eq!(
                files.iter().map(|(file, _)| *file).collect::<Vec<_>>(),
                ["kustomization.yaml"],
                "{name} must not ship a params.env"
            );
            let k = kustomization_of(&files);
            assert!(
                k.get("configMapGenerator").is_none(),
                "{name} must not generate its own params ConfigMap"
            );
            let replacements = k["replacements"].as_array().expect("replacements");
            assert_eq!(replacements.len(), 1);
            assert_eq!(replacements[0]["source"]["name"], PARAMS_CONFIGMAP);
            assert_eq!(
                replacements[0]["source"]["fieldPath"],
                format!("data.{SERVER_IMAGE_PARAM}")
            );
        }
    }

    #[test]
    fn overlays_differ_only_by_component() {
        let mut odh = kustomization_of(&overlay());
        let mut xks = kustomization_of(&xks_overlay());
        assert_eq!(
            odh["components"],
            serde_json::json!([format!("../../{COMPONENT_DIR}")])
        );
        assert_eq!(
            xks["components"],
            serde_json::json!([format!("../../{RELATED_IMAGE_COMPONENT_DIR}")])
        );
        odh["components"] = serde_json::Value::Null;
        xks["components"] = serde_json::Value::Null;
        assert_eq!(odh, xks);
    }

    /// The xKS install must run where no OpenShift API exists.
    #[test]
    fn xks_component_carries_nothing_openshift_specific() {
        let rendered = serde_json::to_string(&related_image_component()).expect("serializes");
        for needle in [
            "openshift",
            "service-ca",
            "monitoring.coreos.com",
            "metrics-tls",
        ] {
            assert!(
                !rendered.contains(needle),
                "xKS component mentions {needle}"
            );
        }
    }

    #[test]
    fn base_params_carry_every_image() {
        let image = "registry.example/operator@sha256:abc";
        let params = params_env(image);
        let keys: Vec<&str> = params
            .lines()
            .map(|line| line.split_once('=').expect("KEY=value").0)
            .collect();
        assert_eq!(keys, [OPERATOR_IMAGE_PARAM, SERVER_IMAGE_PARAM]);
        assert!(params.contains(&format!("{OPERATOR_IMAGE_PARAM}={image}\n")));
        assert!(params_env(DEFAULT_IMAGE).contains(DEFAULT_IMAGE));
    }

    /// The base kustomization is hand-written, and both replacements name
    /// this ConfigMap.
    #[test]
    fn base_generates_the_configmap_the_overlays_read() {
        let base = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../config/manifests/base/kustomization.yaml"),
        )
        .expect("base kustomization");
        assert!(base.contains(&format!("name: {PARAMS_CONFIGMAP}")));
        assert!(base.contains(&format!("data.{OPERATOR_IMAGE_PARAM}")));
    }
}
