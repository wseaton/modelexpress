// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Informer framework for peer-scoring inputs.
//!
//! An `Informer` is an external-data oracle that contributes one numeric
//! signal to the planner's peer-ranking decision. Examples: a Prometheus
//! informer that scores by inverse GPU utilization, a topology informer
//! that scores by rack-distance-to-caller, a maintenance informer that
//! returns `None` to hard-exclude peers in a draining state.
//!
//! Each informer owns its storage. The framework only orchestrates
//! refresh cadences and composes contributions into a single score per
//! `(caller, peer)` pair via `InformerRegistry::composite_score`.
//!
//! Lifecycle: construct the registry with `(Arc<dyn Informer>, weight)`
//! pairs, call `start(shutdown)` to spawn one refresh task per informer,
//! and hand `Arc<InformerRegistry>` to the planner. Drop the shutdown
//! sender on exit to halt all tasks cleanly.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

pub mod config;
pub mod mock;
pub mod prometheus;
pub mod vllm;

pub use config::InformerConfig;
pub use mock::{MockOverlay, MockOverlayConfig};

/// One peer's identity from the informer's POV. The framework supplies
/// these to push-based informers via [`PeerDiscovery`].
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub worker_id: String,
    pub labels: std::collections::HashMap<String, String>,
}

/// Source of peer identities for informers that fetch per-peer external
/// data (e.g. [`vllm::VllmMetricsInformer`]). The framework provides one
/// implementation backed by `P2pStateManager`; tests use static fixtures.
#[async_trait]
pub trait PeerDiscovery: Send + Sync + 'static {
    /// Snapshot of currently-known peers. Called from inside
    /// [`Informer::refresh`] on each tick.
    async fn discover(&self) -> Result<Vec<DiscoveredPeer>>;
}

/// Dependencies injected into [`InformerConfig::build`] for informers
/// that need more than their own static config (e.g. access to the
/// live peer registry).
pub struct InformerContext {
    pub peer_discovery: Arc<dyn PeerDiscovery>,
}

/// Context handed to [`Informer::score_peer`]: identifies the requester
/// and the peer being scored, plus the peer's published external-identity
/// labels so the informer can join against external data sources keyed
/// by whatever dimension makes sense (`pod`, `node`, `rack`, etc.).
#[derive(Debug)]
pub struct ScoreCtx<'a> {
    pub caller_worker_id: &'a str,
    pub peer_worker_id: &'a str,
    /// Labels the caller advertised in PublishMetadata. Lets informers
    /// score on caller-relative comparisons (same rack, same tenant)
    /// without any external data source. May be empty if the caller
    /// published no labels.
    pub caller_labels: &'a HashMap<String, String>,
    /// Labels the peer advertised in PublishMetadata (e.g.
    /// `{"pod": "mx-source-0-abc", "node": "g12e022"}`). May be empty
    /// for peers that didn't publish any.
    pub peer_labels: &'a HashMap<String, String>,
}

/// One peer-scoring oracle.
///
/// Implementations own their storage shape. The framework calls
/// `refresh` on the informer's stated cadence and `score_peer` once
/// per peer per plan request.
#[async_trait]
pub trait Informer: Send + Sync + 'static {
    /// Stable identifier — used in logs and as the score namespace.
    /// Two informers with the same name in one registry is a config error.
    fn name(&self) -> &'static str;

    /// Background refresh cadence. `None` disables background refresh
    /// (e.g., for a one-shot informer initialized at construction).
    fn refresh_interval(&self) -> Option<Duration>;

    /// Refresh internal state from the external source. Implementations
    /// should be idempotent and tolerate transient failures by returning
    /// `Err(_)` — the registry logs and retries on the next tick rather
    /// than crashing.
    async fn refresh(&self) -> Result<()>;

    /// This informer's contribution to a peer's rank from a caller's
    /// point of view.
    ///
    /// * `Some(f64)` — soft signal. Composed via weighted sum.
    /// * `None` — hard exclude. The composite scorer returns `None`
    ///   for the whole `(caller, peer)` pair, signalling the planner
    ///   to drop the peer from consideration entirely.
    ///
    /// `ctx.peer_labels` carries the peer's external-identity labels
    /// (e.g. `{"pod": "mx-source-0-abc"}`). Informers that source data
    /// from external systems join on whichever label key they care
    /// about; informers that operate purely on `worker_id` can ignore
    /// labels.
    ///
    /// Must be cheap (no I/O). Use `refresh` for I/O; `score_peer` is
    /// called on the plan hot path.
    fn score_peer(&self, ctx: &ScoreCtx<'_>) -> Option<f64>;
}

/// Composite scoring: sum of per-informer contributions weighted by
/// configured weights.
///
/// Hard-exclude semantics: if any informer returns `None` for a
/// `(caller, peer)` pair, the composite returns `None`. The planner
/// treats this as "this peer is ineligible for this plan."
pub struct InformerRegistry {
    informers: Vec<WeightedInformer>,
    /// Optional test/debug overlay that can force any informer's
    /// contribution. `None` in normal operation.
    overlay: Option<Arc<MockOverlay>>,
    handles: std::sync::Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for InformerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InformerRegistry")
            .field(
                "informers",
                &self
                    .informers
                    .iter()
                    .map(|w| (w.inner.name(), w.weight))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

struct WeightedInformer {
    inner: Arc<dyn Informer>,
    weight: f64,
}

impl InformerRegistry {
    fn build_weighted(informers: Vec<(Arc<dyn Informer>, f64)>) -> Vec<WeightedInformer> {
        let mut names = std::collections::HashSet::new();
        for (i, _) in &informers {
            if !names.insert(i.name()) {
                // Duplicate names are a config bug, but we refuse to panic
                // at runtime — log and continue with the duplicate. Caller
                // can detect via the warn.
                warn!(
                    "InformerRegistry: duplicate informer name '{}' — \
                     contributions will both be summed, which is probably \
                     not intended",
                    i.name(),
                );
            }
        }
        informers
            .into_iter()
            .map(|(inner, weight)| WeightedInformer { inner, weight })
            .collect()
    }

    /// Construct a registry. Background tasks are NOT spawned until
    /// [`InformerRegistry::start`] is called.
    pub fn new(informers: Vec<(Arc<dyn Informer>, f64)>) -> Arc<Self> {
        Arc::new(Self {
            informers: Self::build_weighted(informers),
            overlay: None,
            handles: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Construct a registry with a [`MockOverlay`] that can force any
    /// informer's contribution for testing. The overlay is refreshed by
    /// its own task spawned in [`InformerRegistry::start`].
    pub fn with_overlay(
        informers: Vec<(Arc<dyn Informer>, f64)>,
        overlay: Arc<MockOverlay>,
    ) -> Arc<Self> {
        Arc::new(Self {
            informers: Self::build_weighted(informers),
            overlay: Some(overlay),
            handles: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Empty registry — no informers, composite score always `Some(0.0)`.
    /// Useful when the server is configured with no external signals.
    pub fn empty() -> Arc<Self> {
        Self::new(Vec::new())
    }

    /// Spawn one background refresh task per informer with a stated
    /// cadence. Tasks listen for `shutdown` and exit cleanly when the
    /// sender is dropped or sends `()`.
    ///
    /// Each task runs an initial refresh immediately (so the registry
    /// is warm by the time the first plan request arrives), then
    /// refreshes on the cadence.
    pub fn start(self: &Arc<Self>, shutdown: watch::Receiver<bool>) {
        // Mutex is held only briefly during start/join. Recover the inner
        // value if a previous holder panicked rather than propagating the
        // panic — losing the handles list is better than crashing the server.
        let mut handles = self.handles.lock().unwrap_or_else(|p| p.into_inner());
        for w in &self.informers {
            let informer = Arc::clone(&w.inner);
            let mut shutdown_rx = shutdown.clone();
            let interval = informer.refresh_interval();
            let name = informer.name();

            let handle = tokio::spawn(async move {
                // Initial refresh so scoring isn't a coin flip on first request.
                if let Err(e) = informer.refresh().await {
                    warn!("Informer '{}' initial refresh failed: {:#}", name, e);
                }

                let Some(period) = interval else {
                    info!(
                        "Informer '{}' has no refresh interval — exiting refresh task \
                         (data must be loaded statically)",
                        name
                    );
                    return;
                };

                let mut tick = tokio::time::interval(period);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                tick.tick().await; // consume the immediate tick we already did

                info!(
                    "Informer '{}' refresh task started (period={:?})",
                    name, period
                );

                loop {
                    tokio::select! {
                        _ = tick.tick() => {
                            if let Err(e) = informer.refresh().await {
                                warn!("Informer '{}' refresh failed: {:#}", name, e);
                            } else {
                                debug!("Informer '{}' refreshed", name);
                            }
                        }
                        changed = shutdown_rx.changed() => {
                            if changed.is_ok() && *shutdown_rx.borrow() {
                                info!("Informer '{}' refresh task shutting down", name);
                                break;
                            }
                            // sender dropped — also exit
                            if changed.is_err() {
                                info!("Informer '{}' shutdown sender dropped — exiting", name);
                                break;
                            }
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Spawn the mock overlay's own refresh task, if present.
        if let Some(overlay) = &self.overlay {
            let overlay = Arc::clone(overlay);
            let mut shutdown_rx = shutdown.clone();
            let handle = tokio::spawn(async move {
                if let Err(e) = overlay.refresh().await {
                    warn!("MockOverlay initial refresh failed: {:#}", e);
                }
                let mut tick = tokio::time::interval(overlay.refresh_interval());
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                tick.tick().await;
                info!(
                    "MockOverlay refresh task started (period={:?}) — scoring is being mocked",
                    overlay.refresh_interval()
                );
                loop {
                    tokio::select! {
                        _ = tick.tick() => {
                            if let Err(e) = overlay.refresh().await {
                                warn!("MockOverlay refresh failed: {:#}", e);
                            }
                        }
                        changed = shutdown_rx.changed() => {
                            if changed.is_err() || *shutdown_rx.borrow() {
                                info!("MockOverlay refresh task shutting down");
                                break;
                            }
                        }
                    }
                }
            });
            handles.push(handle);
        }
    }

    /// Wait for all background tasks to exit. Useful in tests after
    /// triggering shutdown.
    pub async fn join(self: &Arc<Self>) {
        let handles = {
            let mut guard = self.handles.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::take(&mut *guard)
        };
        for h in handles {
            let _ = h.await;
        }
    }

    /// Composite score for the `(caller, peer)` pair described by `ctx`:
    ///
    /// * Returns `None` if any informer hard-excludes the peer.
    /// * Returns `Some(sum_of_weighted_contributions)` otherwise.
    ///   Informers that have no opinion (return `None` for soft signals)
    ///   would conflict with the hard-exclude semantics, so by contract
    ///   `None` ALWAYS means hard-exclude — informers that mean
    ///   "neutral / no opinion" must return `Some(0.0)`.
    ///
    /// An empty registry returns `Some(0.0)` (no opinions = no signal).
    pub fn composite_score(&self, ctx: &ScoreCtx<'_>) -> Option<f64> {
        if self.informers.is_empty() {
            return Some(0.0);
        }
        let mut total = 0.0;
        for w in &self.informers {
            // A mock overlay (if active) can force this informer's
            // contribution; otherwise call the real informer.
            let contribution = match self
                .overlay
                .as_ref()
                .and_then(|o| o.lookup(w.inner.name(), ctx))
            {
                Some(forced) => forced,
                None => w.inner.score_peer(ctx),
            };
            match contribution {
                Some(s) => total += s * w.weight,
                None => return None,
            }
        }
        Some(total)
    }

    /// Diagnostic: list informer names and weights.
    pub fn describe(&self) -> Vec<(&'static str, f64)> {
        self.informers
            .iter()
            .map(|w| (w.inner.name(), w.weight))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
pub mod test_informers {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    /// Informer that returns `Some(0.0)` for everyone and counts refreshes.
    pub struct NullInformer {
        pub name: &'static str,
        pub interval: Option<Duration>,
        pub refresh_count: AtomicU64,
    }

    impl NullInformer {
        pub fn new(name: &'static str, interval: Option<Duration>) -> Arc<Self> {
            Arc::new(Self {
                name,
                interval,
                refresh_count: AtomicU64::new(0),
            })
        }
    }

    #[async_trait]
    impl Informer for NullInformer {
        fn name(&self) -> &'static str {
            self.name
        }
        fn refresh_interval(&self) -> Option<Duration> {
            self.interval
        }
        async fn refresh(&self) -> Result<()> {
            self.refresh_count.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(())
        }
        fn score_peer(&self, _ctx: &ScoreCtx<'_>) -> Option<f64> {
            Some(0.0)
        }
    }

    /// Informer with explicit per-peer scores set by tests.
    pub struct StaticScoreInformer {
        pub name: &'static str,
        pub scores: Mutex<std::collections::HashMap<String, Option<f64>>>,
    }

    impl StaticScoreInformer {
        pub fn new(name: &'static str) -> Arc<Self> {
            Arc::new(Self {
                name,
                scores: Mutex::new(std::collections::HashMap::new()),
            })
        }
        pub fn set(&self, peer: &str, score: Option<f64>) {
            self.scores
                .lock()
                .expect("test mutex poisoned")
                .insert(peer.to_string(), score);
        }
    }

    #[async_trait]
    impl Informer for StaticScoreInformer {
        fn name(&self) -> &'static str {
            self.name
        }
        fn refresh_interval(&self) -> Option<Duration> {
            None
        }
        async fn refresh(&self) -> Result<()> {
            Ok(())
        }
        fn score_peer(&self, ctx: &ScoreCtx<'_>) -> Option<f64> {
            self.scores
                .lock()
                .expect("test mutex poisoned")
                .get(ctx.peer_worker_id)
                .copied()
                .unwrap_or(Some(0.0))
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::test_informers::{NullInformer, StaticScoreInformer};
    use super::*;
    use std::sync::atomic::Ordering as AtomicOrdering;

    fn ctx<'a>(
        caller: &'a str,
        peer: &'a str,
        labels: &'a HashMap<String, String>,
    ) -> ScoreCtx<'a> {
        ScoreCtx {
            caller_worker_id: caller,
            peer_worker_id: peer,
            caller_labels: labels,
            peer_labels: labels,
        }
    }

    #[test]
    fn empty_registry_returns_zero_score() {
        let reg = InformerRegistry::empty();
        let labels = HashMap::new();
        assert_eq!(
            reg.composite_score(&ctx("caller", "peer", &labels)),
            Some(0.0)
        );
    }

    #[test]
    fn weighted_sum_composition() {
        let a = StaticScoreInformer::new("a");
        let b = StaticScoreInformer::new("b");
        a.set("p0", Some(2.0));
        b.set("p0", Some(3.0));

        let reg = InformerRegistry::new(vec![
            (Arc::clone(&a) as Arc<dyn Informer>, 1.0),
            (Arc::clone(&b) as Arc<dyn Informer>, 0.5),
        ]);

        let labels = HashMap::new();
        // 2.0 * 1.0 + 3.0 * 0.5 = 3.5
        assert_eq!(
            reg.composite_score(&ctx("caller", "p0", &labels)),
            Some(3.5)
        );
    }

    #[test]
    fn none_from_any_informer_hard_excludes() {
        let a = StaticScoreInformer::new("a");
        let b = StaticScoreInformer::new("b");
        a.set("p0", Some(5.0));
        b.set("p0", None); // hard exclude

        let reg = InformerRegistry::new(vec![
            (Arc::clone(&a) as Arc<dyn Informer>, 1.0),
            (Arc::clone(&b) as Arc<dyn Informer>, 1.0),
        ]);
        let labels = HashMap::new();
        assert_eq!(reg.composite_score(&ctx("caller", "p0", &labels)), None);

        // Other peer unaffected.
        a.set("p1", Some(7.0));
        b.set("p1", Some(0.0));
        assert_eq!(
            reg.composite_score(&ctx("caller", "p1", &labels)),
            Some(7.0)
        );
    }

    #[test]
    fn unknown_peer_treated_as_zero_by_static_informer() {
        // StaticScoreInformer chooses to return Some(0.0) by default; this
        // documents that contract.
        let a = StaticScoreInformer::new("a");
        let reg = InformerRegistry::new(vec![(Arc::clone(&a) as Arc<dyn Informer>, 1.0)]);
        let labels = HashMap::new();
        assert_eq!(
            reg.composite_score(&ctx("caller", "never-set", &labels)),
            Some(0.0)
        );
    }

    #[tokio::test]
    async fn background_refresh_fires_on_cadence_and_stops_on_shutdown() {
        let informer = NullInformer::new("counter", Some(Duration::from_millis(40)));
        let reg = InformerRegistry::new(vec![(Arc::clone(&informer) as Arc<dyn Informer>, 1.0)]);

        let (tx, rx) = watch::channel(false);
        reg.start(rx);

        // Let a few ticks land.
        tokio::time::sleep(Duration::from_millis(180)).await;
        let mid = informer.refresh_count.load(AtomicOrdering::SeqCst);
        // Initial refresh + several ticks; exact count depends on the
        // tokio runtime's scheduling, so we just assert "more than one."
        assert!(mid >= 2, "expected at least 2 refreshes, got {mid}");

        // Trigger shutdown and verify the task exits.
        tx.send(true).expect("shutdown send");
        reg.join().await;

        // No further refreshes after join returns.
        let after = informer.refresh_count.load(AtomicOrdering::SeqCst);
        tokio::time::sleep(Duration::from_millis(80)).await;
        let final_count = informer.refresh_count.load(AtomicOrdering::SeqCst);
        assert_eq!(after, final_count, "refresh continued after join");
    }

    #[tokio::test]
    async fn informer_with_no_interval_runs_only_initial_refresh() {
        let informer = NullInformer::new("one-shot", None);
        let reg = InformerRegistry::new(vec![(Arc::clone(&informer) as Arc<dyn Informer>, 1.0)]);

        let (_tx, rx) = watch::channel(false);
        reg.start(rx);
        reg.join().await;

        assert_eq!(informer.refresh_count.load(AtomicOrdering::SeqCst), 1);
    }
}
