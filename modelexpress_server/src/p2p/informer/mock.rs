// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry-level mock overlay for end-to-end scorer testing.
//!
//! This is NOT an [`super::Informer`]. It sits inside the
//! [`super::InformerRegistry`] and can force any informer's contribution
//! for specific peers, short-circuiting the real `score_peer` call. That
//! makes every informer mockable without touching its implementation:
//! you control the planner's inputs deterministically, no real metrics
//! or load required.
//!
//! Backed by a JSON file re-read on a refresh interval — edit the file
//! (e.g. a mounted ConfigMap) and overrides take effect within one
//! interval, no server restart.
//!
//! File format: `{ "<informer-name>": { "<join-value>": <score|null> } }`.
//! The special informer name `"*"` applies to every informer. A specific
//! name takes precedence over `"*"`. A `null` value hard-excludes the
//! peer from that informer's contribution (which, per composite
//! semantics, hard-excludes the peer entirely).
//!
//! ```json
//! {
//!   "*": { "mx-source-0": 0.1, "mx-source-1": 1.0 }
//! }
//! ```
//!
//! Joins on a configured label (default `pod`): the overlay looks up
//! `peer.labels[join_label]` and matches that value in the file.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::Duration;
use tracing::{debug, info};

use super::ScoreCtx;

/// Wildcard informer name that applies an override to every informer.
const WILDCARD: &str = "*";

/// `Some(score)` = forced soft score; `None` = forced hard-exclude.
type Forced = Option<f64>;
/// informer name (or `*`) -> (join-value -> forced contribution).
type OverrideTable = HashMap<String, HashMap<String, Forced>>;

pub struct MockOverlay {
    path: PathBuf,
    join_label: String,
    refresh: Duration,
    table: RwLock<OverrideTable>,
}

impl MockOverlay {
    pub fn new(
        path: impl Into<PathBuf>,
        join_label: impl Into<String>,
        refresh: Duration,
    ) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            path: path.into(),
            join_label: join_label.into(),
            refresh,
            table: RwLock::new(HashMap::new()),
        })
    }

    pub fn refresh_interval(&self) -> Duration {
        self.refresh
    }

    /// Re-read the override file. A missing file clears all overrides
    /// (so removing the file disables mocking without a restart).
    pub async fn refresh(&self) -> Result<()> {
        let body = match tokio::fs::read_to_string(&self.path).await {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let mut g = self.table.write().unwrap_or_else(|p| p.into_inner());
                if !g.is_empty() {
                    info!(
                        "MockOverlay: file {} gone, cleared overrides",
                        self.path.display()
                    );
                }
                g.clear();
                return Ok(());
            }
            Err(e) => return Err(e).with_context(|| format!("reading {}", self.path.display())),
        };
        let parsed = parse_overrides(&body)
            .with_context(|| format!("parsing override file {}", self.path.display()))?;
        let n: usize = parsed.values().map(HashMap::len).sum();
        let mut g = self.table.write().unwrap_or_else(|p| p.into_inner());
        *g = parsed;
        drop(g);
        info!(
            "MockOverlay: loaded {} override(s) from {}",
            n,
            self.path.display()
        );
        Ok(())
    }

    /// If an override applies to `informer_name` for the peer in `ctx`,
    /// return `Some(forced)`; otherwise `None` (defer to the real
    /// informer). A specific informer name wins over the `*` wildcard.
    pub fn lookup(&self, informer_name: &str, ctx: &ScoreCtx<'_>) -> Option<Forced> {
        let key = ctx.peer_labels.get(&self.join_label)?;
        let table = self.table.read().unwrap_or_else(|p| p.into_inner());
        if let Some(forced) = table.get(informer_name).and_then(|m| m.get(key)) {
            debug!(
                "MockOverlay: forcing '{}' for {}={} -> {:?}",
                informer_name, self.join_label, key, forced
            );
            return Some(*forced);
        }
        if let Some(forced) = table.get(WILDCARD).and_then(|m| m.get(key)) {
            debug!(
                "MockOverlay: forcing (*) for {}={} -> {:?}",
                self.join_label, key, forced
            );
            return Some(*forced);
        }
        None
    }
}

pub(crate) fn parse_overrides(body: &str) -> Result<OverrideTable> {
    let raw: HashMap<String, HashMap<String, serde_json::Value>> =
        serde_json::from_str(body).context("decoding override JSON")?;
    let mut out = OverrideTable::with_capacity(raw.len());
    for (informer, peers) in raw {
        let mut m = HashMap::with_capacity(peers.len());
        for (peer, v) in peers {
            let forced = match v {
                serde_json::Value::Null => None,
                serde_json::Value::Number(n) => Some(
                    n.as_f64()
                        .with_context(|| format!("override {informer}/{peer} not an f64"))?,
                ),
                other => {
                    anyhow::bail!("override {informer}/{peer} must be number or null, got {other}")
                }
            };
            m.insert(peer, forced);
        }
        out.insert(informer, m);
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockOverlayConfig {
    /// Path to the override JSON file (typically a mounted ConfigMap).
    pub path: String,
    /// MX-side label to join overrides on. Defaults to `pod`.
    #[serde(default = "default_join_label")]
    pub join_label: String,
    #[serde(default = "default_refresh_seconds")]
    pub refresh_seconds: u64,
}

impl MockOverlayConfig {
    pub fn build(&self) -> std::sync::Arc<MockOverlay> {
        MockOverlay::new(
            self.path.clone(),
            self.join_label.clone(),
            Duration::from_secs(self.refresh_seconds),
        )
    }
}

fn default_join_label() -> String {
    "pod".to_string()
}

fn default_refresh_seconds() -> u64 {
    10
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn ctx<'a>(peer: &'a str, labels: &'a HashMap<String, String>) -> ScoreCtx<'a> {
        ScoreCtx {
            caller_worker_id: "c",
            peer_worker_id: peer,
            caller_labels: labels,
            peer_labels: labels,
        }
    }

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn parses_named_and_wildcard_and_null() {
        let body = r#"{
            "vllm_kv_idle": { "mx-source-0": 0.1 },
            "*": { "mx-source-1": 1.0, "mx-source-2": null }
        }"#;
        let t = parse_overrides(body).expect("parse");
        assert_eq!(t["vllm_kv_idle"]["mx-source-0"], Some(0.1));
        assert_eq!(t["*"]["mx-source-1"], Some(1.0));
        assert_eq!(t["*"]["mx-source-2"], None);
    }

    #[test]
    fn rejects_bad_value() {
        assert!(parse_overrides(r#"{ "*": { "a": "nope" } }"#).is_err());
    }

    #[test]
    fn specific_name_beats_wildcard() {
        let ov = MockOverlay::new("/nonexistent", "role", Duration::from_secs(10));
        {
            let mut g = ov.table.write().expect("lock");
            let mut named = HashMap::new();
            named.insert("source-0".to_string(), Some(0.1));
            g.insert("vllm_kv_idle".to_string(), named);
            let mut wild = HashMap::new();
            wild.insert("source-0".to_string(), Some(0.9));
            g.insert("*".to_string(), wild);
        }
        let l = labels(&[("role", "source-0")]);
        // Specific informer name wins.
        assert_eq!(ov.lookup("vllm_kv_idle", &ctx("w", &l)), Some(Some(0.1)));
        // A different informer falls back to wildcard.
        assert_eq!(ov.lookup("other", &ctx("w", &l)), Some(Some(0.9)));
    }

    #[test]
    fn no_override_returns_none() {
        let ov = MockOverlay::new("/nonexistent", "role", Duration::from_secs(10));
        let l = labels(&[("role", "unknown")]);
        assert_eq!(ov.lookup("any", &ctx("w", &l)), None);
        // Peer missing the join label → also no override.
        let l2 = labels(&[("pod", "x")]);
        assert_eq!(ov.lookup("any", &ctx("w", &l2)), None);
    }

    #[test]
    fn null_override_is_hard_exclude() {
        let ov = MockOverlay::new("/nonexistent", "role", Duration::from_secs(10));
        {
            let mut g = ov.table.write().expect("lock");
            let mut wild = HashMap::new();
            wild.insert("source-9".to_string(), None);
            g.insert("*".to_string(), wild);
        }
        let l = labels(&[("role", "source-9")]);
        assert_eq!(ov.lookup("any", &ctx("w", &l)), Some(None));
    }

    #[tokio::test]
    async fn refresh_handles_missing_then_present() {
        let path = std::env::temp_dir().join(format!("mx-mock-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let ov = MockOverlay::new(path.clone(), "role", Duration::from_secs(10));

        ov.refresh().await.expect("missing ok");
        let l = labels(&[("role", "source-0")]);
        assert_eq!(ov.lookup("x", &ctx("w", &l)), None);

        std::fs::write(&path, r#"{ "*": { "source-0": 0.3 } }"#).expect("write");
        ov.refresh().await.expect("present ok");
        assert_eq!(ov.lookup("x", &ctx("w", &l)), Some(Some(0.3)));

        let _ = std::fs::remove_file(&path);
    }
}
