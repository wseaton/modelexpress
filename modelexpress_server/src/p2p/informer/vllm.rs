// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Informer that pulls each peer's vLLM `/metrics` endpoint directly,
//! parses the Prometheus text-exposition format, and exposes a single
//! configured metric per worker as a score.
//!
//! Self-contained — does not require a Prometheus stack in the cluster.
//! Useful when a) the cluster's Prometheus is unavailable, b) you want
//! lower scrape latency than a Prometheus poll-of-a-poll, or c) you're
//! running outside k8s.
//!
//! How it discovers peers:
//!   On each refresh tick, the informer calls [`super::PeerDiscovery`]
//!   (injected at construction by the server's main wiring) to get the
//!   current peer list with their published labels. For each peer, it
//!   builds the metrics URL from a configured label (typically
//!   `pod_ip`) and an explicit port. Peers without that label or whose
//!   fetch fails are dropped from the next snapshot and fall through to
//!   the unknown-policy at score time.
//!
//! Configurable score transforms. Convention: composite scoring is a
//! weighted sum where HIGHER = more-preferred peer (see
//! [`super::InformerRegistry::composite_score`]). A transform's job is
//! to flip the raw metric into that convention.
//!   * `Identity` — score = raw metric value. Correct ONLY when a higher
//!     raw value means a better peer. Most vLLM load gauges
//!     (`num_requests_running`, `gpu_cache_usage_perc`) are the opposite
//!     (higher = busier = worse), so `Identity` on those ranks backwards.
//!   * `OneMinus`  — score = 1.0 - metric value (turns "1.0 = busy"
//!     gauges like `vllm:gpu_cache_usage_perc` into "higher = better").
//!     Only stays in `[0, 1]` for metrics already bounded to `[0, 1]`.
//!   * `Negate`    — score = -metric, for any "lower is better" GAUGE.
//!     Do not point this at cumulative Prometheus counters (`*_total`):
//!     they only ever increase and never reset, so `Negate` would
//!     permanently penalize long-lived pods by lifetime volume rather
//!     than current load.
//!
//! CAVEAT — these transforms set DIRECTION, not SCALE. There is no
//! normalization: `OneMinus` emits `[0, 1]`, `Negate` emits unbounded
//! negatives, `Identity` unbounded positives. `composite_score` sums
//! these raw, so when more than one numeric informer runs together the
//! widest-range one dominates and the configured `weight` no longer
//! means what an operator expects. Until a normalization step exists,
//! keep informer outputs on a comparable scale (ideally clamp/scale each
//! into `[0, 1]`) if you combine them.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{debug, warn};

use super::prometheus::{UnknownPolicy, UnknownPolicyConfig};
use super::{Informer, InformerContext, PeerDiscovery, ScoreCtx};

/// What to do with the raw metric value before storing it as a score.
#[derive(Debug, Clone, Copy)]
pub enum ScoreTransform {
    Identity,
    OneMinus,
    Negate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreTransformConfig {
    Identity,
    OneMinus,
    Negate,
}

impl From<ScoreTransformConfig> for ScoreTransform {
    fn from(c: ScoreTransformConfig) -> Self {
        match c {
            ScoreTransformConfig::Identity => ScoreTransform::Identity,
            ScoreTransformConfig::OneMinus => ScoreTransform::OneMinus,
            ScoreTransformConfig::Negate => ScoreTransform::Negate,
        }
    }
}

impl ScoreTransform {
    fn apply(self, v: f64) -> f64 {
        match self {
            ScoreTransform::Identity => v,
            ScoreTransform::OneMinus => 1.0 - v,
            ScoreTransform::Negate => -v,
        }
    }
}

pub struct VllmMetricsInformer {
    name: &'static str,
    /// Prom metric name to extract, e.g. `"vllm:num_requests_running"`
    /// or `"vllm:gpu_cache_usage_perc"`. If the metric has multiple
    /// label sets per peer, we sum them.
    metric_name: String,
    /// Label key whose value is the host portion of the metrics URL
    /// (typically `"pod_ip"`).
    host_label: String,
    /// Port to hit on each peer's metrics endpoint (usually the vLLM
    /// OpenAI API port, since vLLM serves /metrics on the same port).
    port: u16,
    /// Path to the metrics endpoint. Defaults to `"/metrics"`.
    path: String,
    refresh: Duration,
    transform: ScoreTransform,
    unknown: UnknownPolicy,
    /// worker_id -> latest score. Score = transform(raw_metric_value).
    samples: RwLock<HashMap<String, f64>>,
    discovery: Arc<dyn PeerDiscovery>,
    client: reqwest::Client,
}

impl VllmMetricsInformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &'static str,
        metric_name: impl Into<String>,
        host_label: impl Into<String>,
        port: u16,
        path: impl Into<String>,
        refresh: Duration,
        transform: ScoreTransform,
        unknown: UnknownPolicy,
        discovery: Arc<dyn PeerDiscovery>,
    ) -> Result<Arc<Self>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .context("building reqwest client for VllmMetricsInformer")?;
        Ok(Arc::new(Self {
            name,
            metric_name: metric_name.into(),
            host_label: host_label.into(),
            port,
            path: path.into(),
            refresh,
            transform,
            unknown,
            samples: RwLock::new(HashMap::new()),
            discovery,
            client,
        }))
    }

    /// Snapshot for diagnostics + tests.
    pub fn snapshot(&self) -> HashMap<String, f64> {
        self.samples
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    async fn fetch_one(&self, peer: &super::DiscoveredPeer) -> Option<(String, f64)> {
        let host = peer.labels.get(&self.host_label)?;
        let url = format!("http://{}:{}{}", host, self.port, self.path);
        let body = match self.client.get(&url).send().await {
            Ok(r) => match r.error_for_status() {
                Ok(r) => match r.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        debug!(
                            "VllmMetricsInformer '{}': read body from {} failed: {:#}",
                            self.name, url, e
                        );
                        return None;
                    }
                },
                Err(e) => {
                    debug!(
                        "VllmMetricsInformer '{}': non-2xx from {}: {:#}",
                        self.name, url, e
                    );
                    return None;
                }
            },
            Err(e) => {
                debug!(
                    "VllmMetricsInformer '{}': fetch {} failed: {:#}",
                    self.name, url, e
                );
                return None;
            }
        };

        match extract_metric_sum(&body, &self.metric_name) {
            Ok(Some(raw)) => Some((peer.worker_id.clone(), self.transform.apply(raw))),
            Ok(None) => {
                debug!(
                    "VllmMetricsInformer '{}': metric '{}' not present at {}",
                    self.name, self.metric_name, url
                );
                None
            }
            Err(e) => {
                warn!(
                    "VllmMetricsInformer '{}': parse error at {}: {:#}",
                    self.name, url, e
                );
                None
            }
        }
    }
}

#[async_trait]
impl Informer for VllmMetricsInformer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn refresh_interval(&self) -> Option<Duration> {
        Some(self.refresh)
    }

    async fn refresh(&self) -> Result<()> {
        let peers = self.discovery.discover().await.context("peer discovery")?;
        let fetches = peers.iter().map(|p| self.fetch_one(p));
        let results = join_all(fetches).await;

        let mut next: HashMap<String, f64> = HashMap::new();
        for r in results.into_iter().flatten() {
            next.insert(r.0, r.1);
        }

        let count = next.len();
        let mut guard = self.samples.write().unwrap_or_else(|p| p.into_inner());
        *guard = next;
        drop(guard);

        debug!(
            "VllmMetricsInformer '{}': refreshed {} samples (peers={})",
            self.name,
            count,
            peers.len()
        );
        Ok(())
    }

    fn score_peer(&self, ctx: &ScoreCtx<'_>) -> Option<f64> {
        let guard = self.samples.read().unwrap_or_else(|p| p.into_inner());
        match guard.get(ctx.peer_worker_id) {
            Some(v) => Some(*v),
            None => match self.unknown {
                UnknownPolicy::Neutral(v) => Some(v),
                UnknownPolicy::Exclude => None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmMetricsInformerConfig {
    pub name: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Prom metric name to extract from each peer's /metrics page.
    pub metric_name: String,
    /// MX-side label key whose value is the host part of the metrics
    /// URL. Typically `"pod_ip"`. Set in the worker manifest via
    /// `MX_LABEL_pod_ip` from `status.podIP`.
    pub host_label: String,
    /// Port to hit on each peer (the vLLM OpenAI API port; vLLM serves
    /// /metrics on the same port).
    pub port: u16,
    /// Path to the metrics endpoint. Defaults to `/metrics`.
    #[serde(default = "default_path")]
    pub path: String,
    #[serde(default = "default_refresh_seconds")]
    pub refresh_seconds: u64,
    #[serde(default = "default_transform")]
    pub transform: ScoreTransformConfig,
    pub unknown_policy: UnknownPolicyConfig,
}

impl VllmMetricsInformerConfig {
    pub fn build(&self, ctx: &InformerContext) -> Result<Arc<dyn Informer>> {
        let leaked: &'static str = Box::leak(self.name.clone().into_boxed_str());
        let inf = VllmMetricsInformer::new(
            leaked,
            &self.metric_name,
            &self.host_label,
            self.port,
            &self.path,
            Duration::from_secs(self.refresh_seconds),
            self.transform.clone().into(),
            self.unknown_policy.clone().into(),
            Arc::clone(&ctx.peer_discovery),
        )
        .with_context(|| format!("building VllmMetricsInformer '{}'", self.name))?;
        Ok(inf as Arc<dyn Informer>)
    }
}

fn default_weight() -> f64 {
    1.0
}

fn default_refresh_seconds() -> u64 {
    15
}

fn default_path() -> String {
    "/metrics".to_string()
}

fn default_transform() -> ScoreTransformConfig {
    ScoreTransformConfig::Identity
}

/// Sum all sample lines whose metric name (the leading identifier
/// before either `{` or whitespace) matches `metric_name`. Returns
/// `Ok(None)` if the metric isn't present at all, `Ok(Some(sum))`
/// otherwise. Errors out on parse failures inside matching lines.
///
/// We sum rather than pick the first because vLLM exports per-model
/// labels (e.g. `vllm:num_requests_running{model_name="..."} 0`) and
/// the operator usually wants the aggregate across whatever the pod
/// is serving.
pub(crate) fn extract_metric_sum(body: &str, metric_name: &str) -> Result<Option<f64>> {
    let mut sum = 0.0;
    let mut matched = false;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Format: `<name>{labels} <value> [<timestamp>]` or `<name> <value>`.
        // Split into name-and-labels chunk vs value chunk.
        let (lhs, rhs) = match line.find(char::is_whitespace) {
            Some(i) => (&line[..i], line[i..].trim_start()),
            None => continue,
        };
        let name = match lhs.find('{') {
            Some(i) => &lhs[..i],
            None => lhs,
        };
        if name != metric_name {
            continue;
        }
        let value_str = rhs
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("metric '{}' line missing value: '{}'", metric_name, line))?;
        let v: f64 = value_str.parse().with_context(|| {
            format!("metric '{}' value '{}' not a float", metric_name, value_str)
        })?;
        if !v.is_finite() {
            continue;
        }
        sum += v;
        matched = true;
    }
    Ok(if matched { Some(sum) } else { None })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_gauge() {
        let body = "\
# HELP vllm:num_requests_running ...
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name=\"qwen\"} 2.0
";
        let v = extract_metric_sum(body, "vllm:num_requests_running").expect("ok");
        assert_eq!(v, Some(2.0));
    }

    #[test]
    fn sums_multiple_label_sets() {
        let body = "\
vllm:num_requests_running{model_name=\"a\"} 2
vllm:num_requests_running{model_name=\"b\"} 3
vllm:num_requests_running{model_name=\"c\"} 0.5
";
        let v = extract_metric_sum(body, "vllm:num_requests_running").expect("ok");
        assert_eq!(v, Some(5.5));
    }

    #[test]
    fn ignores_other_metrics_and_comments() {
        let body = "\
# HELP something_else
something_else 999
vllm:num_requests_running{model_name=\"qwen\"} 1
# vllm:num_requests_running is the one we want
other_metric{k=\"v\"} 50
";
        let v = extract_metric_sum(body, "vllm:num_requests_running").expect("ok");
        assert_eq!(v, Some(1.0));
    }

    #[test]
    fn missing_metric_returns_none() {
        let body = "other_metric 1\n";
        let v = extract_metric_sum(body, "vllm:num_requests_running").expect("ok");
        assert_eq!(v, None);
    }

    #[test]
    fn skips_non_finite_values() {
        let body = "vllm:num_requests_running{m=\"a\"} NaN\nvllm:num_requests_running{m=\"b\"} 4\n";
        let v = extract_metric_sum(body, "vllm:num_requests_running").expect("ok");
        assert_eq!(v, Some(4.0));
    }

    #[test]
    fn handles_no_labels() {
        let body = "vllm:gpu_cache_usage_perc 0.42\n";
        let v = extract_metric_sum(body, "vllm:gpu_cache_usage_perc").expect("ok");
        assert_eq!(v, Some(0.42));
    }

    #[test]
    fn score_transform_one_minus() {
        assert!((ScoreTransform::OneMinus.apply(0.25) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn score_transform_negate() {
        assert!((ScoreTransform::Negate.apply(7.0) - (-7.0)).abs() < 1e-9);
    }

    // Static PeerDiscovery for tests of score_peer behavior without HTTP.
    struct StaticDiscovery(Vec<super::super::DiscoveredPeer>);
    #[async_trait]
    impl PeerDiscovery for StaticDiscovery {
        async fn discover(&self) -> Result<Vec<super::super::DiscoveredPeer>> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn unknown_peer_with_neutral_policy() {
        let disc: Arc<dyn PeerDiscovery> = Arc::new(StaticDiscovery(vec![]));
        let inf = VllmMetricsInformer::new(
            "test",
            "vllm:num_requests_running",
            "pod_ip",
            8000,
            "/metrics",
            Duration::from_secs(30),
            ScoreTransform::Identity,
            UnknownPolicy::Neutral(0.5),
            disc,
        )
        .expect("build");
        let labels = HashMap::new();
        let ctx = ScoreCtx {
            caller_worker_id: "c",
            peer_worker_id: "unseen",
            caller_labels: &labels,
            peer_labels: &labels,
        };
        assert_eq!(inf.score_peer(&ctx), Some(0.5));
    }

    #[test]
    fn known_peer_returns_transformed_score() {
        let disc: Arc<dyn PeerDiscovery> = Arc::new(StaticDiscovery(vec![]));
        let inf = VllmMetricsInformer::new(
            "test",
            "vllm:num_requests_running",
            "pod_ip",
            8000,
            "/metrics",
            Duration::from_secs(30),
            ScoreTransform::Identity,
            UnknownPolicy::Exclude,
            disc,
        )
        .expect("build");
        inf.samples
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert("w-a".to_string(), 3.0);
        let labels = HashMap::new();
        let ctx = ScoreCtx {
            caller_worker_id: "c",
            peer_worker_id: "w-a",
            caller_labels: &labels,
            peer_labels: &labels,
        };
        assert_eq!(inf.score_peer(&ctx), Some(3.0));
    }
}
