// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The declared desired set: the models a node is supposed to hold. The
//! reconciler diffs this against what is actually on local NVMe and pulls the
//! difference. A [`DesiredSet`] is the only thing that varies between deployment
//! styles; [`StaticDesiredSet`] is the v1 implementation, built from the CLI /
//! config model list. A K8s-CRD/label-backed source is a later concern, so the
//! reconciler depends on the trait, not on where the list comes from.

use modelexpress_common::models::ModelProvider;

/// One model the node should hold. The cache daemon is HuggingFace-only in v1
/// (the [`crate::cached::locator::HfLocator`] resolves the on-disk layout), so
/// the provider is fixed; the type carries it explicitly so widening to other
/// providers later is a data change, not a control-flow change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// HuggingFace model id, e.g. `google-t5/t5-small`.
    pub model: String,
    /// Pin to a specific revision, or `None` to accept whatever revision a peer
    /// or origin serves (the v1 cache holds one revision per model).
    pub revision: Option<String>,
}

impl ModelSpec {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            revision: None,
        }
    }

    /// The provider this spec resolves through. Fixed to HuggingFace in v1.
    pub fn provider(&self) -> ModelProvider {
        ModelProvider::HuggingFace
    }

    /// The revision to publish in the source identity (`""` when unpinned).
    pub fn identity_revision(&self) -> &str {
        self.revision.as_deref().unwrap_or("")
    }
}

/// The set of models a node should converge its local cache toward.
pub trait DesiredSet {
    /// Snapshot the currently-desired models. Called once per reconcile pass, so
    /// a dynamic implementation can return a fresh view each time.
    fn desired(&self) -> Vec<ModelSpec>;
}

/// A fixed desired set, the v1 source: the model list handed in on the CLI (or
/// loaded from config). Immutable for the daemon's lifetime.
#[derive(Debug, Clone, Default)]
pub struct StaticDesiredSet {
    specs: Vec<ModelSpec>,
}

impl StaticDesiredSet {
    /// Build from model ids (e.g. the `--model` flags), de-duplicating while
    /// preserving order so the reconcile pass is deterministic.
    pub fn from_models<I, S>(models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut specs: Vec<ModelSpec> = Vec::new();
        for model in models {
            let spec = ModelSpec::new(model);
            if !specs.iter().any(|existing| existing.model == spec.model) {
                specs.push(spec);
            }
        }
        Self { specs }
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

impl DesiredSet for StaticDesiredSet {
    fn desired(&self) -> Vec<ModelSpec> {
        self.specs.clone()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn static_set_dedups_and_preserves_order() {
        let set = StaticDesiredSet::from_models([
            "google-t5/t5-small",
            "Qwen/Qwen2.5-7B",
            "google-t5/t5-small",
        ]);
        let desired = set.desired();
        let models: Vec<&str> = desired.iter().map(|s| s.model.as_str()).collect();
        assert_eq!(models, vec!["google-t5/t5-small", "Qwen/Qwen2.5-7B"]);
    }

    #[test]
    fn empty_set_is_empty() {
        assert!(StaticDesiredSet::default().is_empty());
        assert!(StaticDesiredSet::from_models(Vec::<String>::new()).is_empty());
    }

    #[test]
    fn spec_defaults_to_huggingface_and_unpinned_revision() {
        let spec = ModelSpec::new("google-t5/t5-small");
        assert_eq!(spec.provider(), ModelProvider::HuggingFace);
        assert_eq!(spec.identity_revision(), "");
        assert!(spec.revision.is_none());
    }
}
