// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dispatcher enum that ties each informer kind to its own config struct.
//!
//! Each informer module owns the schema of its config (e.g.
//! [`super::prometheus::PrometheusInformerConfig`]) — this file just
//! discriminates between them on the `kind` field at deserialize time.
//!
//! YAML example (in the server's config file):
//!
//! ```yaml
//! informers:
//!   - kind: prometheus
//!     name: gpu_idle
//!     weight: 1.0
//!     endpoint: "http://prometheus:9090"
//!     query: '1 - avg by (worker_id) (rate(vllm:gpu_cache_usage_perc{}[1m]))'
//!     worker_label: worker_id
//!     refresh_seconds: 30
//!     unknown_policy:
//!       kind: neutral
//!       default: 0.5
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{
    Informer, InformerContext, prometheus::PrometheusInformerConfig,
    vllm::VllmMetricsInformerConfig,
};

/// Top-level informer entry — tag-discriminated by `kind`. To add a new
/// informer kind: write the impl + its `*InformerConfig` in a sibling
/// module, then add a variant here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InformerConfig {
    Prometheus(PrometheusInformerConfig),
    VllmMetrics(VllmMetricsInformerConfig),
}

impl InformerConfig {
    /// Human-readable name for this informer (also its score namespace).
    pub fn name(&self) -> &str {
        match self {
            InformerConfig::Prometheus(c) => &c.name,
            InformerConfig::VllmMetrics(c) => &c.name,
        }
    }

    /// Discriminator string for diagnostics (e.g. "prometheus").
    pub fn kind(&self) -> &'static str {
        match self {
            InformerConfig::Prometheus(_) => "prometheus",
            InformerConfig::VllmMetrics(_) => "vllm_metrics",
        }
    }

    /// Weight multiplier on this informer's contribution in composite scoring.
    pub fn weight(&self) -> f64 {
        match self {
            InformerConfig::Prometheus(c) => c.weight,
            InformerConfig::VllmMetrics(c) => c.weight,
        }
    }

    /// Materialize this config into a live informer instance. `ctx`
    /// provides runtime dependencies (e.g. peer discovery) that some
    /// informer kinds need; others ignore it.
    pub fn build(&self, ctx: &InformerContext) -> Result<Arc<dyn Informer>> {
        match self {
            InformerConfig::Prometheus(c) => c.build(),
            InformerConfig::VllmMetrics(c) => c.build(ctx),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::p2p::informer::prometheus::UnknownPolicyConfig;
    use crate::p2p::informer::{DiscoveredPeer, PeerDiscovery};
    use anyhow::Result as AResult;
    use async_trait::async_trait;

    struct NoopDiscovery;
    #[async_trait]
    impl PeerDiscovery for NoopDiscovery {
        async fn discover(&self) -> AResult<Vec<DiscoveredPeer>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn deserializes_prometheus_neutral() {
        let yaml = r#"
          kind: prometheus
          name: gpu_idle
          weight: 2.0
          endpoint: http://prom:9090
          query: 'up{}'
          worker_label: worker_id
          refresh_seconds: 15
          unknown_policy:
            kind: neutral
            default: 0.5
        "#;
        let cfg: InformerConfig = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(cfg.name(), "gpu_idle");
        assert_eq!(cfg.weight(), 2.0);
        match cfg {
            InformerConfig::Prometheus(p) => {
                assert_eq!(p.endpoint, "http://prom:9090");
                assert_eq!(p.refresh_seconds, 15);
                assert!(matches!(
                    p.unknown_policy,
                    UnknownPolicyConfig::Neutral { .. }
                ));
            }
            other => panic!("expected Prometheus variant, got {other:?}"),
        }
    }

    #[test]
    fn deserializes_prometheus_exclude_with_defaults() {
        let yaml = r#"
          kind: prometheus
          name: gpu_idle
          endpoint: http://prom:9090
          query: 'up{}'
          worker_label: worker_id
          unknown_policy:
            kind: exclude
        "#;
        let cfg: InformerConfig = serde_yaml::from_str(yaml).expect("deserialize");
        match cfg {
            InformerConfig::Prometheus(p) => {
                assert_eq!(p.weight, 1.0);
                assert_eq!(p.refresh_seconds, 30);
                assert!(matches!(p.unknown_policy, UnknownPolicyConfig::Exclude));
            }
            other => panic!("expected Prometheus variant, got {other:?}"),
        }
    }

    #[test]
    fn builds_into_live_informer() {
        let yaml = r#"
          kind: prometheus
          name: test
          endpoint: http://unused
          query: 'up'
          worker_label: pod
          unknown_policy:
            kind: neutral
            default: 0.1
        "#;
        let cfg: InformerConfig = serde_yaml::from_str(yaml).expect("deserialize");
        // Prometheus build ignores ctx; supply a dummy discovery.
        let dummy_ctx = crate::p2p::informer::InformerContext {
            peer_discovery: std::sync::Arc::new(NoopDiscovery)
                as std::sync::Arc<dyn crate::p2p::informer::PeerDiscovery>,
        };
        let live = cfg.build(&dummy_ctx).expect("build");
        assert_eq!(live.name(), "test");
        // Peer with no labels at all → falls into unknown policy branch.
        let labels = std::collections::HashMap::new();
        let ctx = crate::p2p::informer::ScoreCtx {
            caller_worker_id: "caller",
            peer_worker_id: "never-seen",
            caller_labels: &labels,
            peer_labels: &labels,
        };
        assert_eq!(live.score_peer(&ctx), Some(0.1));
    }
}
