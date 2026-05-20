// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical [`Informer`] implementation: polls a Prometheus instant
//! query and exposes the latest sample per worker as a score.
//!
//! Configuration:
//!
//! * `endpoint` — Prometheus base URL, e.g. `http://prometheus:9090`.
//! * `query` — A PromQL expression that returns an instant vector
//!   grouped by some label that uniquely identifies a worker. Example:
//!   `1 - avg by (worker_id) (rate(gpu_utilization{job="mx"}[1m]))`
//!   (closer to 1 = less utilized = better candidate).
//! * `worker_label` — Name of the label that holds the worker_id
//!   (e.g. `"worker_id"`, `"instance"`, or `"pod"`).
//! * `refresh` — How often to re-query Prometheus.
//! * `unknown_policy` — How to score peers absent from the latest
//!   query result. See [`UnknownPolicy`].
//!
//! Storage is a single [`RwLock<HashMap<String, f64>>`] holding the
//! latest value per worker. We don't keep history at this layer — the
//! framework only needs a current score. If you want smoothing or
//! windowed percentiles, layer them inside the PromQL itself (e.g.
//! `avg_over_time(...[5m])`).

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::debug;

use super::{Informer, ScoreCtx};

/// Policy for scoring peers that were not in the latest Prometheus result.
#[derive(Debug, Clone, Copy)]
pub enum UnknownPolicy {
    /// Return `Some(default)`: peer is eligible but gets the configured
    /// neutral score. Use this when missing data should NOT exclude
    /// peers (e.g., scraper just started, peer too new to have samples).
    Neutral(f64),
    /// Return `None`: hard-exclude unknown peers from candidacy. Use
    /// this when you only want peers with up-to-date health data.
    Exclude,
}

/// Serializable form of [`UnknownPolicy`], used by [`PrometheusInformerConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UnknownPolicyConfig {
    Neutral { default: f64 },
    Exclude,
}

impl From<UnknownPolicyConfig> for UnknownPolicy {
    fn from(c: UnknownPolicyConfig) -> Self {
        match c {
            UnknownPolicyConfig::Neutral { default } => UnknownPolicy::Neutral(default),
            UnknownPolicyConfig::Exclude => UnknownPolicy::Exclude,
        }
    }
}

/// Declarative configuration for a [`PrometheusInformer`]. Lives next to
/// the informer impl so the informer owns the schema of its own config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusInformerConfig {
    pub name: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    pub endpoint: String,
    pub query: String,
    /// Prometheus label name in the query result whose value identifies
    /// a worker (e.g. `"pod"` when the query is `... by (pod) (...)`).
    pub worker_label: String,
    /// MX-side label name to join on (e.g. `"pod"`). The informer
    /// looks up `peer.labels[join_label]` and uses that as the key
    /// into the latest Prometheus result. Defaults to `worker_label`
    /// when omitted — they're usually the same string.
    #[serde(default)]
    pub join_label: Option<String>,
    #[serde(default = "default_refresh_seconds")]
    pub refresh_seconds: u64,
    pub unknown_policy: UnknownPolicyConfig,
}

impl PrometheusInformerConfig {
    /// Materialize into a live informer. The name is intentionally
    /// leaked once (bounded by config size) to satisfy the trait's
    /// `&'static str` return without adding a `String` accessor that
    /// only this informer would need.
    pub fn build(&self) -> Result<Arc<dyn Informer>> {
        let leaked: &'static str = Box::leak(self.name.clone().into_boxed_str());
        let join = self
            .join_label
            .clone()
            .unwrap_or_else(|| self.worker_label.clone());
        let inf = PrometheusInformer::new(
            leaked,
            &self.endpoint,
            &self.query,
            &self.worker_label,
            &join,
            Duration::from_secs(self.refresh_seconds),
            self.unknown_policy.clone().into(),
        )
        .with_context(|| format!("building PrometheusInformer '{}'", self.name))?;
        Ok(inf as Arc<dyn Informer>)
    }
}

fn default_weight() -> f64 {
    1.0
}

fn default_refresh_seconds() -> u64 {
    30
}

pub struct PrometheusInformer {
    name: &'static str,
    endpoint: String,
    query: String,
    /// Name of the Prometheus label whose value identifies a worker
    /// (e.g. "pod"). Used to key the internal sample map from the
    /// query result.
    worker_label: String,
    /// Name of the MX-side worker label to join on. Often the same
    /// string as `worker_label`, but kept independent so the Prometheus
    /// label name can differ from the MX label name when needed.
    join_label: String,
    refresh: Duration,
    unknown: UnknownPolicy,
    samples: RwLock<HashMap<String, f64>>,
    client: reqwest::Client,
}

impl PrometheusInformer {
    /// Construct a Prometheus-backed informer. The HTTP client is
    /// owned by the informer and configured with a short request
    /// timeout — slow Prometheus queries should NOT stall the plan
    /// loop on a future call.
    pub fn new(
        name: &'static str,
        endpoint: impl Into<String>,
        query: impl Into<String>,
        worker_label: impl Into<String>,
        join_label: impl Into<String>,
        refresh: Duration,
        unknown: UnknownPolicy,
    ) -> Result<Arc<Self>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .context("building reqwest client for PrometheusInformer")?;
        Ok(Arc::new(Self {
            name,
            endpoint: endpoint.into(),
            query: query.into(),
            worker_label: worker_label.into(),
            join_label: join_label.into(),
            refresh,
            unknown,
            samples: RwLock::new(HashMap::new()),
            client,
        }))
    }

    /// Snapshot of the current sample map. Primarily for diagnostics
    /// and tests; the planner calls `score_peer` instead.
    pub fn snapshot(&self) -> HashMap<String, f64> {
        self.samples
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

#[async_trait]
impl Informer for PrometheusInformer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn refresh_interval(&self) -> Option<Duration> {
        Some(self.refresh)
    }

    async fn refresh(&self) -> Result<()> {
        let url = format!("{}/api/v1/query", self.endpoint.trim_end_matches('/'));
        let resp = self
            .client
            .get(&url)
            .query(&[("query", self.query.as_str())])
            .send()
            .await
            .with_context(|| format!("GET {url}"))?
            .error_for_status()
            .with_context(|| format!("prometheus query returned non-2xx ({url})"))?;
        let body = resp
            .text()
            .await
            .context("reading prometheus response body")?;

        let next = parse_prometheus_vector(&body, &self.worker_label)
            .context("parsing prometheus vector response")?;

        // Single short critical section: atomically replace the map.
        // Plan-time reads grab a clone or look up keys — they don't
        // hold the write lock.
        let count = next.len();
        let mut guard = self.samples.write().unwrap_or_else(|p| p.into_inner());
        *guard = next;
        drop(guard);

        debug!(
            "PrometheusInformer '{}' refreshed: {} samples",
            self.name, count
        );
        Ok(())
    }

    fn score_peer(&self, ctx: &ScoreCtx<'_>) -> Option<f64> {
        // Join on the peer's MX-side label value (e.g. ctx.peer_labels["pod"]).
        // Peers that didn't publish the required label can't be matched
        // against this informer's external data; treat them per the
        // unknown-policy (Neutral or Exclude).
        let key = match ctx.peer_labels.get(&self.join_label) {
            Some(k) => k.as_str(),
            None => {
                return match self.unknown {
                    UnknownPolicy::Neutral(v) => Some(v),
                    UnknownPolicy::Exclude => None,
                };
            }
        };
        let guard = self.samples.read().unwrap_or_else(|p| p.into_inner());
        match guard.get(key) {
            Some(v) => Some(*v),
            None => match self.unknown {
                UnknownPolicy::Neutral(v) => Some(v),
                UnknownPolicy::Exclude => None,
            },
        }
    }
}

/// Parse Prometheus's `/api/v1/query` JSON response (vector resultType)
/// into a `worker_label -> value` map.
///
/// Public (pub(crate)) so it can be unit-tested without spinning up an
/// HTTP server. The HTTP path is intentionally a thin shim around this.
pub(crate) fn parse_prometheus_vector(
    body: &str,
    worker_label: &str,
) -> Result<HashMap<String, f64>> {
    // Prom response shape (vector result):
    // {
    //   "status": "success",
    //   "data": {
    //     "resultType": "vector",
    //     "result": [
    //       { "metric": { "<label>": "value", ... }, "value": [ts, "0.85"] },
    //       ...
    //     ]
    //   }
    // }
    let v: serde_json::Value =
        serde_json::from_str(body).context("decoding prometheus response as JSON")?;

    let status = v
        .get("status")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("response missing 'status'"))?;
    if status != "success" {
        let err = v
            .get("error")
            .and_then(|e| e.as_str())
            .unwrap_or("(no error message)");
        return Err(anyhow!(
            "prometheus query failed: status={status} error={err}"
        ));
    }

    let result_type = v
        .pointer("/data/resultType")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("response missing data.resultType"))?;
    if result_type != "vector" {
        return Err(anyhow!(
            "expected resultType=vector, got '{result_type}' (use an instant query)"
        ));
    }

    let entries = v
        .pointer("/data/result")
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("response missing data.result array"))?;

    let mut out = HashMap::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let metric = entry
            .get("metric")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("result[{i}] missing metric object"))?;
        let Some(worker) = metric.get(worker_label).and_then(|v| v.as_str()) else {
            // Skip series that don't carry the worker label — they're
            // not addressable from the planner's POV. Log at the call
            // site if you care; we don't spam at this layer.
            continue;
        };
        let value_pair = entry
            .get("value")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("result[{i}] missing value array"))?;
        // value = [unix_ts, "stringified_float"]
        let raw = value_pair
            .get(1)
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("result[{i}].value[1] not a string"))?;
        let parsed: f64 = raw
            .parse()
            .with_context(|| format!("result[{i}].value[1] = '{raw}' is not a float"))?;
        if !parsed.is_finite() {
            // Prometheus emits NaN/+Inf as "NaN" / "+Inf" in JSON; skip
            // them rather than poisoning the score map.
            continue;
        }
        out.insert(worker.to_string(), parsed);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_vector_response() {
        let body = r#"{
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"worker_id": "w-a", "job": "mx"}, "value": [1700000000.0, "0.85"]},
                    {"metric": {"worker_id": "w-b", "job": "mx"}, "value": [1700000000.0, "0.42"]}
                ]
            }
        }"#;
        let out = parse_prometheus_vector(body, "worker_id").expect("parse");
        assert_eq!(out.len(), 2);
        assert!((out["w-a"] - 0.85).abs() < 1e-9);
        assert!((out["w-b"] - 0.42).abs() < 1e-9);
    }

    #[test]
    fn skips_series_missing_worker_label() {
        let body = r#"{
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"worker_id": "w-a"}, "value": [0, "1.0"]},
                    {"metric": {"job": "other"}, "value": [0, "2.0"]}
                ]
            }
        }"#;
        let out = parse_prometheus_vector(body, "worker_id").expect("parse");
        assert_eq!(out.len(), 1);
        assert!(out.contains_key("w-a"));
    }

    #[test]
    fn skips_non_finite_values() {
        let body = r#"{
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"worker_id": "w-a"}, "value": [0, "1.0"]},
                    {"metric": {"worker_id": "w-nan"}, "value": [0, "NaN"]},
                    {"metric": {"worker_id": "w-inf"}, "value": [0, "+Inf"]}
                ]
            }
        }"#;
        let out = parse_prometheus_vector(body, "worker_id").expect("parse");
        assert_eq!(out.len(), 1);
        assert!(out.contains_key("w-a"));
    }

    #[test]
    fn rejects_non_vector_result_type() {
        let body = r#"{
            "status": "success",
            "data": { "resultType": "matrix", "result": [] }
        }"#;
        let err = parse_prometheus_vector(body, "worker_id").expect_err("should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("vector"), "{msg}");
    }

    #[test]
    fn propagates_prometheus_status_error() {
        let body =
            r#"{ "status": "error", "errorType": "bad_data", "error": "parse error at char 5" }"#;
        let err = parse_prometheus_vector(body, "worker_id").expect_err("should fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("parse error"), "{msg}");
    }

    #[test]
    fn empty_vector_returns_empty_map() {
        let body = r#"{
            "status": "success",
            "data": { "resultType": "vector", "result": [] }
        }"#;
        let out = parse_prometheus_vector(body, "worker_id").expect("parse");
        assert!(out.is_empty());
    }

    fn ctx<'a>(peer: &'a str, labels: &'a HashMap<String, String>) -> ScoreCtx<'a> {
        ScoreCtx {
            caller_worker_id: "caller",
            peer_worker_id: peer,
            caller_labels: labels,
            peer_labels: labels,
        }
    }

    fn label_map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn unknown_policy_neutral_returns_default() {
        // Build an informer with no HTTP traffic — we just exercise the
        // unknown-peer fallback path against an empty samples map.
        let inf = PrometheusInformer::new(
            "test",
            "http://unused",
            "up",
            "pod",
            "pod",
            Duration::from_secs(30),
            UnknownPolicy::Neutral(0.5),
        )
        .expect("build");
        let labels = label_map(&[("pod", "never-seen")]);
        assert_eq!(inf.score_peer(&ctx("p", &labels)), Some(0.5));
    }

    #[test]
    fn unknown_policy_exclude_returns_none() {
        let inf = PrometheusInformer::new(
            "test",
            "http://unused",
            "up",
            "pod",
            "pod",
            Duration::from_secs(30),
            UnknownPolicy::Exclude,
        )
        .expect("build");
        let labels = label_map(&[("pod", "never-seen")]);
        assert_eq!(inf.score_peer(&ctx("p", &labels)), None);
    }

    #[test]
    fn known_peer_returns_stored_value_regardless_of_unknown_policy() {
        let inf = PrometheusInformer::new(
            "test",
            "http://unused",
            "up",
            "pod",
            "pod",
            Duration::from_secs(30),
            UnknownPolicy::Exclude,
        )
        .expect("build");
        // Simulate a successful refresh by writing directly.
        inf.samples
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert("w-a".to_string(), 0.9);
        let labels_known = label_map(&[("pod", "w-a")]);
        let labels_unknown = label_map(&[("pod", "w-b")]);
        assert_eq!(inf.score_peer(&ctx("p", &labels_known)), Some(0.9));
        assert_eq!(inf.score_peer(&ctx("p", &labels_unknown)), None);
    }

    #[test]
    fn peer_missing_join_label_applies_unknown_policy() {
        // Peer that didn't publish the join label key — Neutral returns default.
        let inf = PrometheusInformer::new(
            "test",
            "http://unused",
            "up",
            "pod",
            "pod",
            Duration::from_secs(30),
            UnknownPolicy::Neutral(0.25),
        )
        .expect("build");
        let no_labels = HashMap::new();
        assert_eq!(inf.score_peer(&ctx("p", &no_labels)), Some(0.25));

        // Exclude variant — same peer is hard-excluded.
        let inf_excl = PrometheusInformer::new(
            "test",
            "http://unused",
            "up",
            "pod",
            "pod",
            Duration::from_secs(30),
            UnknownPolicy::Exclude,
        )
        .expect("build");
        assert_eq!(inf_excl.score_peer(&ctx("p", &no_labels)), None);
    }

    #[test]
    fn join_label_can_differ_from_worker_label() {
        // Prometheus side calls the label `instance`, MX side calls
        // the same identity dimension `pod`. Both line up via the
        // resolved string values stored in samples vs. ctx.peer_labels.
        let inf = PrometheusInformer::new(
            "test",
            "http://unused",
            "up",
            "instance", // worker_label (prom side)
            "pod",      // join_label (MX side)
            Duration::from_secs(30),
            UnknownPolicy::Exclude,
        )
        .expect("build");
        inf.samples
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert("mx-source-0".to_string(), 0.7);
        let labels = label_map(&[("pod", "mx-source-0"), ("namespace", "mx-test")]);
        assert_eq!(inf.score_peer(&ctx("p", &labels)), Some(0.7));
    }
}
