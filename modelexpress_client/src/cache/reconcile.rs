// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reconciliation loop: drive the local cache toward the [`DesiredSet`].
//!
//! One pass observes what is on local NVMe, diffs it against the desired set,
//! and for every missing model either pulls it from a peer over RDMA (when the
//! P2P registry knows a holder) or fetches it from origin (when nobody does),
//! then advertises the node's own residency so other nodes can pull from it.
//!
//! The byte-moving itself is delegated to a [`Fetcher`] so the registry-facing
//! orchestration here stays free of the `!Send` NIXL agent (which must never
//! cross an `.await`): the production [`NixlFetcher`] confines the agent to a
//! `spawn_blocking` worker, and tests drive the same orchestration against the
//! honest in-process loopback transport plus a real registry server. The
//! `advertised` map is held by the caller, not the reconciler, and is locked
//! only momentarily, so a multi-minute pull never blocks the heartbeat task
//! that keeps published sources out of the reaper's reach.
//!
//! Presence is defined by the cache layout's completion sentinel: a model
//! counts as held only when [`cache_layout::is_complete`] is true for its
//! snapshot directory, so a half-written directory from a crashed pull is
//! re-fetched rather than served. Origin downloads don't write the sentinel, so
//! [`ensure_complete`] stamps it after a successful fetch, unifying the
//! completeness contract across both paths.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use anyhow::{Context, anyhow};
use modelexpress_common::grpc::p2p::SourceStatus;
use tracing::info;

use super::advertise::{cache_worker, file_cache_identity};
use super::desired::ModelSpec;
use super::discover::discover_blob;
use super::locator::{list_cached_models, locate_hf};
use super::registry::Registry;
use super::transfer::cache_layout;
use super::usage::{AtimeUsage, UsageSignal};

/// Where a model came from on a given reconcile pass. Returned so the daemon and
/// tests can assert the path taken without parsing logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchSource {
    /// Pulled from a peer found in the registry, over the transfer protocol.
    Peer,
    /// Fetched from origin because no peer advertised the model.
    Origin,
}

/// The byte-moving half of a reconcile, kept behind a trait so the registry
/// orchestration stays transport-agnostic and the `!Send` NIXL agent is confined
/// to the implementor. Both methods land the model under `dest_root` in the
/// standard cache layout; the reconciler stamps the completion sentinel after.
#[tonic::async_trait]
pub trait Fetcher: Send + Sync {
    /// Pull `spec` from the peer identified by `holder_md` into `dest_root`.
    async fn peer_pull(
        &self,
        spec: &ModelSpec,
        holder_md: Vec<u8>,
        dest_root: &Path,
    ) -> anyhow::Result<()>;

    /// Fetch `spec` from origin into `dest_root` (no peer holds it).
    async fn origin(&self, spec: &ModelSpec, dest_root: &Path) -> anyhow::Result<()>;
}

/// Whether `model` is held and complete in the cache at `cache_root`. A model
/// directory without the completion sentinel (a partial or in-flight pull) is
/// treated as absent.
pub fn is_present(cache_root: &Path, model: &str) -> bool {
    match locate_hf(cache_root, model) {
        Some((dir, _)) => cache_layout::is_complete(&dir),
        None => false,
    }
}

/// The desired models not currently held-and-complete at `cache_root`.
pub fn missing(desired: &[ModelSpec], cache_root: &Path) -> Vec<ModelSpec> {
    desired
        .iter()
        .filter(|spec| !is_present(cache_root, &spec.model))
        .cloned()
        .collect()
}

/// Stamp the completion sentinel on a freshly-fetched model if it isn't already
/// there (the peer-pull path writes it; the origin path does not). Errors if the
/// model isn't on disk after the fetch claimed success.
pub fn ensure_complete(cache_root: &Path, model: &str) -> anyhow::Result<()> {
    let (dir, _revision) =
        locate_hf(cache_root, model).with_context(|| format!("{model} not on disk after fetch"))?;
    if !cache_layout::is_complete(&dir) {
        cache_layout::mark_complete(&dir)?;
    }
    Ok(())
}

/// Reclaim a model's disk: drop the completion sentinel first so it is instantly
/// treated as absent and never served mid-delete, then remove the whole repo
/// directory (`models--<org>--<name>`), not just the snapshot.
fn evict_model(snapshot: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(snapshot.join(cache_layout::COMPLETE_SENTINEL));
    let repo = snapshot.parent().and_then(Path::parent).unwrap_or(snapshot);
    std::fs::remove_dir_all(repo)
}

/// Drives the local cache toward a desired set against one P2P registry. Holds
/// the static facts a pass needs (the cache root, the blob and identity to
/// advertise under); the mutable `advertised` map lives with the caller so the
/// heartbeat task can read it without contending on a long pull.
pub struct Reconciler {
    registry: Registry,
    cache_root: std::path::PathBuf,
    /// The NIXL metadata blob peers load to pull from this node (the serve
    /// agent's `local_md`). Advertised verbatim for every model.
    advertise_md: Vec<u8>,
    agent_name: String,
    metadata_endpoint: String,
    worker_id: String,
    /// Per-model on-disk fingerprint from the previous `observe_local` pass, so
    /// an externally-downloaded model is advertised only once it stops changing
    /// (see [`Reconciler::observe_local`]). Empty until auto-expand observes.
    observed: HashMap<String, (u64, u64)>,
    /// Bounded (auto-expand) mode: gate advertising on demand and run GC. When
    /// false the reconciler keeps Phase A behaviour (advertise everything
    /// desired, never evict).
    bounded: bool,
    /// Local use signal feeding the demand TTL (bounded mode).
    usage: Arc<dyn UsageSignal>,
    /// A model not pinned and unused for this long stops being advertised, so it
    /// decays out of the registry and other nodes stop wanting it.
    demand_ttl: Duration,
    /// A model touched within this window is never GC'd, a best-effort guard
    /// against deleting something just pulled or in use.
    gc_grace: Duration,
}

impl Reconciler {
    pub fn new(
        registry: Registry,
        cache_root: std::path::PathBuf,
        advertise_md: Vec<u8>,
        agent_name: impl Into<String>,
        metadata_endpoint: impl Into<String>,
        worker_id: impl Into<String>,
    ) -> Self {
        Self {
            registry,
            cache_root,
            advertise_md,
            agent_name: agent_name.into(),
            metadata_endpoint: metadata_endpoint.into(),
            worker_id: worker_id.into(),
            observed: HashMap::new(),
            bounded: false,
            usage: Arc::new(AtimeUsage),
            demand_ttl: Duration::MAX,
            gc_grace: Duration::ZERO,
        }
    }

    /// Turn on bounded (auto-expand) mode: advertise only pinned-or-recently-used
    /// models so unused ones decay out of the fleet, and enable [`Reconciler::gc`].
    pub fn with_bounding(
        mut self,
        usage: Arc<dyn UsageSignal>,
        demand_ttl: Duration,
        gc_grace: Duration,
    ) -> Self {
        self.bounded = true;
        self.usage = usage;
        self.demand_ttl = demand_ttl;
        self.gc_grace = gc_grace;
        self
    }

    /// Whether this node should keep `model` advertised: pinned in the base set,
    /// or used locally within `demand_ttl`. A model present but neither pinned nor
    /// recently used is held silently (no advertise) until it leaves the desired
    /// set, then [`Reconciler::gc`] reclaims it.
    fn is_wanted(&self, model: &str, pinned: &HashSet<String>) -> bool {
        if pinned.contains(model) {
            return true;
        }
        let Some((snapshot, _)) = locate_hf(&self.cache_root, model) else {
            return false;
        };
        match self.usage.last_used(&snapshot) {
            // A future timestamp (clock skew) reads as fresh, not stale.
            Some(t) => SystemTime::now()
                .duration_since(t)
                .map(|age| age <= self.demand_ttl)
                .unwrap_or(true),
            None => false,
        }
    }

    /// Mark one advertised source `STALE` so peers stop selecting it before the
    /// reaper's timeout. Used when a model's demand decays or it is evicted.
    async fn deregister_one(&mut self, source_id: &str) {
        if let Err(e) = self
            .registry
            .update_status(
                source_id.to_string(),
                0,
                SourceStatus::Stale,
                self.worker_id.clone(),
            )
            .await
        {
            tracing::warn!(source_id, error = %e, "deregister failed; reaper will reap");
        }
    }

    /// Run one reconcile pass. For each desired model: skip it if it is already
    /// held and advertised; otherwise fetch it if missing, stamp completion, and
    /// advertise residency, recording its `mx_source_id` in `advertised` so the
    /// heartbeat task keeps it alive. A failure on one model is logged and the
    /// pass continues, so one bad model can't stall convergence of the rest.
    pub async fn reconcile_once(
        &mut self,
        desired: &[ModelSpec],
        fetcher: &dyn Fetcher,
        advertised: &Mutex<HashMap<String, String>>,
        pinned: &HashSet<String>,
    ) -> anyhow::Result<()> {
        for spec in desired {
            if let Err(e) = self
                .reconcile_model(spec, fetcher, advertised, pinned)
                .await
            {
                tracing::warn!(model = %spec.model, error = %e, "reconcile failed; will retry next pass");
            }
        }
        Ok(())
    }

    async fn reconcile_model(
        &mut self,
        spec: &ModelSpec,
        fetcher: &dyn Fetcher,
        advertised: &Mutex<HashMap<String, String>>,
        pinned: &HashSet<String>,
    ) -> anyhow::Result<()> {
        let already_advertised = self.lock_advertised(advertised)?.contains_key(&spec.model);
        let present = is_present(&self.cache_root, &spec.model);

        if !present {
            let source = self.fetch(spec, fetcher).await?;
            ensure_complete(&self.cache_root, &spec.model)?;
            info!(model = %spec.model, source = ?source, "model converged into local cache");
        }

        // Advertise only models this node keeps alive fleet-wide: when bounded,
        // pinned-or-recently-used; otherwise everything desired (Phase A). A model
        // whose demand has decayed is deregistered but kept on disk until it
        // leaves the desired set, when `gc` reclaims it.
        let want = !self.bounded || self.is_wanted(&spec.model, pinned);
        match (want, already_advertised) {
            (true, false) => {
                let source_id = self.advertise(spec).await?;
                self.lock_advertised(advertised)?
                    .insert(spec.model.clone(), source_id);
            }
            (false, true) => {
                let source_id = self.lock_advertised(advertised)?.remove(&spec.model);
                if let Some(source_id) = source_id {
                    self.deregister_one(&source_id).await;
                    info!(model = %spec.model, "demand decayed; stopped advertising");
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Evict locally-held models the fleet no longer wants (bounded mode only). A
    /// model is reclaimed when it is complete, not pinned, absent from the desired
    /// set (so it has decayed out of the registry, no peer still advertises it),
    /// and untouched within the grace window. Stops advertising it, drops the
    /// completion sentinel so it is instantly unservable, then deletes the repo.
    pub async fn gc(
        &mut self,
        desired: &[ModelSpec],
        pinned: &HashSet<String>,
        advertised: &Mutex<HashMap<String, String>>,
    ) {
        if !self.bounded {
            return;
        }
        let desired_models: HashSet<&str> = desired.iter().map(|s| s.model.as_str()).collect();
        for model in list_cached_models(&self.cache_root) {
            if pinned.contains(&model) || desired_models.contains(model.as_str()) {
                continue;
            }
            let Some((snapshot, _)) = locate_hf(&self.cache_root, &model) else {
                continue;
            };
            // A mid-pull directory has no sentinel; leave it to the fetch path.
            if !cache_layout::is_complete(&snapshot) {
                continue;
            }
            // Grace: anything touched recently (just pulled or in use) is kept.
            if let Some(t) = self.usage.last_used(&snapshot)
                && SystemTime::now()
                    .duration_since(t)
                    .map(|age| age < self.gc_grace)
                    .unwrap_or(true)
            {
                continue;
            }
            let source_id = advertised
                .lock()
                .ok()
                .and_then(|mut map| map.remove(&model));
            if let Some(source_id) = source_id {
                self.deregister_one(&source_id).await;
            }
            match evict_model(&snapshot) {
                Ok(()) => info!(model = %model, "evicted (demand decayed, no longer desired)"),
                Err(e) => {
                    tracing::warn!(model = %model, error = %e, "evict failed; will retry next pass")
                }
            }
        }
    }

    /// Capture half of auto-expand: advertise models that appeared in the local
    /// cache without going through the reconcile fetch path, e.g. a co-located
    /// vLLM that downloaded straight into the shared cache dir. The serve agent
    /// already streams any locally-held model on request, so advertising is all
    /// it takes for peers to pull and replicate it fleet-wide.
    ///
    /// A model that already carries our completion sentinel (we pulled it, or it
    /// persisted across a restart) is advertised at once. An externally-written
    /// one is advertised only once it has *settled*: no in-progress
    /// `*.incomplete` blob, and an unchanged (count, bytes) fingerprint since the
    /// previous pass. That two-pass wait is what stops us advertising a
    /// half-downloaded model a peer would then pull as garbage; we can't know an
    /// external downloader's intended file list, so quiescence is the signal.
    pub async fn observe_local(
        &mut self,
        advertised: &Mutex<HashMap<String, String>>,
        pinned: &HashSet<String>,
    ) -> anyhow::Result<()> {
        for model in list_cached_models(&self.cache_root) {
            if self.lock_advertised(advertised)?.contains_key(&model) {
                continue;
            }
            let Some((snapshot, _revision)) = locate_hf(&self.cache_root, &model) else {
                continue;
            };
            let ready = cache_layout::is_complete(&snapshot)
                || match snapshot.parent().and_then(|p| p.parent()) {
                    Some(repo) => self.settled_and_stable(&model, repo, &snapshot),
                    None => false,
                };
            if !ready {
                continue;
            }
            if !cache_layout::is_complete(&snapshot) {
                cache_layout::mark_complete(&snapshot)?;
            }
            self.observed.remove(&model);
            // A captured model is advertised only if wanted; a freshly-settled
            // download is recently-touched, so under bounded mode it passes. A
            // stale leftover that happens to settle is held silently, not
            // advertised, and `gc` reclaims it once it leaves the desired set.
            if !self.bounded || self.is_wanted(&model, pinned) {
                let source_id = self.advertise(&ModelSpec::new(model.clone())).await?;
                self.lock_advertised(advertised)?.insert(model, source_id);
            }
        }
        Ok(())
    }

    /// Whether an externally-downloaded snapshot is settled (no in-progress blob)
    /// and unchanged since the previous pass. Records the current fingerprint for
    /// the next pass either way; a still-downloading model resets the clock so a
    /// later quiet pair of passes is needed before it advertises.
    fn settled_and_stable(&mut self, model: &str, repo: &Path, snapshot: &Path) -> bool {
        if cache_layout::has_incomplete_blobs(repo) {
            self.observed.remove(model);
            return false;
        }
        let fingerprint = cache_layout::snapshot_fingerprint(snapshot);
        if fingerprint.0 == 0 {
            self.observed.remove(model);
            return false;
        }
        // insert returns the previous fingerprint; stable iff it matches.
        self.observed.insert(model.to_string(), fingerprint) == Some(fingerprint)
    }

    /// Fetch a missing model: peer-pull when the registry knows a holder,
    /// otherwise origin. Returns which path was taken.
    async fn fetch(
        &mut self,
        spec: &ModelSpec,
        fetcher: &dyn Fetcher,
    ) -> anyhow::Result<FetchSource> {
        let identity = file_cache_identity(spec.model.clone(), spec.identity_revision());
        match discover_blob(&mut self.registry, identity, &self.worker_id).await? {
            Some(holder_md) => {
                info!(model = %spec.model, "peer holds model; pulling over RDMA");
                fetcher.peer_pull(spec, holder_md, &self.cache_root).await?;
                Ok(FetchSource::Peer)
            }
            None => {
                info!(model = %spec.model, "no peer holds model; fetching from origin");
                fetcher.origin(spec, &self.cache_root).await?;
                Ok(FetchSource::Origin)
            }
        }
    }

    /// Publish this node's residency for `spec`; returns the `mx_source_id`.
    async fn advertise(&mut self, spec: &ModelSpec) -> anyhow::Result<String> {
        let identity = file_cache_identity(spec.model.clone(), spec.identity_revision());
        let worker = cache_worker(
            self.advertise_md.clone(),
            self.agent_name.clone(),
            self.metadata_endpoint.clone(),
        );
        let source_id = self
            .registry
            .publish(identity, worker, self.worker_id.clone())
            .await?;
        info!(model = %spec.model, source_id = %source_id, "advertised");
        Ok(source_id)
    }

    /// Best-effort deregister on shutdown: mark every advertised source `STALE`
    /// so peers stop selecting this node immediately, rather than waiting out
    /// the reaper's heartbeat timeout. Failures are logged, not propagated, since
    /// the node is going away regardless and the reaper is the backstop.
    pub async fn deregister(&mut self, advertised: &Mutex<HashMap<String, String>>) {
        let source_ids: Vec<String> = match advertised.lock() {
            Ok(map) => map.values().cloned().collect(),
            Err(_) => return,
        };
        for source_id in source_ids {
            if let Err(e) = self
                .registry
                .update_status(
                    source_id.clone(),
                    0,
                    SourceStatus::Stale,
                    self.worker_id.clone(),
                )
                .await
            {
                tracing::warn!(source_id, error = %e, "deregister failed; reaper will reap");
            }
        }
    }

    fn lock_advertised<'a>(
        &self,
        advertised: &'a Mutex<HashMap<String, String>>,
    ) -> anyhow::Result<std::sync::MutexGuard<'a, HashMap<String, String>>> {
        advertised
            .lock()
            .map_err(|_| anyhow!("advertised map mutex poisoned"))
    }
}

/// Production [`Fetcher`]: peer-pulls over real NIXL/RDMA and falls back to the
/// origin download path. The `!Send` `NixlAgent` is created and dropped inside a
/// `spawn_blocking` worker, so it never crosses an `.await`. A fresh agent per
/// pull (co-resident with the serve agent the daemon also runs) matches the
/// validated one-shot pull path; pooling is a Phase 4 concern.
#[cfg(feature = "nixl")]
pub struct NixlFetcher {
    /// Base NIXL agent name; the pull agent uses `<name>-pull` to stay distinct
    /// from the co-resident serve agent.
    pub agent_name: String,
    /// Staging-buffer cap in GiB for the pull agent.
    pub buf_gib: u32,
    /// Receive pipeline depth (slots the staging buffer is carved into).
    pub pool_depth: usize,
    /// Write pulled files with `O_DIRECT` (bypass the page cache).
    pub direct: bool,
    /// Registry/server endpoint used for the origin fallback download.
    pub endpoint: String,
}

#[cfg(feature = "nixl")]
#[tonic::async_trait]
impl Fetcher for NixlFetcher {
    async fn peer_pull(
        &self,
        spec: &ModelSpec,
        holder_md: Vec<u8>,
        dest_root: &Path,
    ) -> anyhow::Result<()> {
        use crate::cache::transfer::{nixl::NixlAgent, puller::Puller};
        use modelexpress_common::cache::resolve_model_path;
        use modelexpress_common::models::ModelProvider;

        let name = format!("{}-pull", self.agent_name);
        let buf_gib = self.buf_gib;
        let pool_depth = self.pool_depth;
        let direct = self.direct;
        let model = spec.model.clone();
        let dest_root = dest_root.to_path_buf();
        // NixlAgent is !Send: confine it to the blocking worker, never across .await.
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mut agent = NixlAgent::new(&name, 0)?;
            let mut puller = Puller::new(&mut agent, buf_gib, pool_depth, direct)?;
            puller.pull(&holder_md, &model, |rev| {
                resolve_model_path(&dest_root, ModelProvider::HuggingFace, &model, Some(rev))
                    .unwrap_or_else(|_| {
                        dest_root
                            .join(format!("models--{}", model.replace('/', "--")))
                            .join("snapshots")
                            .join(rev)
                    })
            })?;
            Ok(())
        })
        .await?
    }

    async fn origin(&self, spec: &ModelSpec, dest_root: &Path) -> anyhow::Result<()> {
        // Stream into the node's own local NVMe (shared_storage = false), not a
        // shared FS: the daemon owns its cache root.
        let mut config = crate::ClientConfig::default();
        config.connection.endpoint = self.endpoint.clone();
        config.cache.local_path = dest_root.to_path_buf();
        config.cache.shared_storage = false;
        crate::Client::request_model_with_smart_fallback(
            spec.model.clone(),
            spec.provider(),
            config,
            false,
        )
        .await
        .map_err(|e| anyhow!("origin fetch of {} failed: {e}", spec.model))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Lay down an HF snapshot dir with one file; optionally mark it complete.
    fn write_model(
        cache_root: &Path,
        model: &str,
        revision: &str,
        complete: bool,
    ) -> std::path::PathBuf {
        let dir = cache_root
            .join(format!("models--{}", model.replace('/', "--")))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write");
        if complete {
            cache_layout::mark_complete(&dir).expect("sentinel");
        }
        dir
    }

    #[test]
    fn presence_requires_the_completion_sentinel() {
        let cache = tempfile::tempdir().expect("tempdir");
        assert!(!is_present(cache.path(), "google-t5/t5-small"));
        // Files present but no sentinel -> still absent (a partial pull).
        write_model(cache.path(), "google-t5/t5-small", "rev1", false);
        assert!(!is_present(cache.path(), "google-t5/t5-small"));
        // Sentinel present -> held.
        write_model(cache.path(), "google-t5/t5-small", "rev1", true);
        assert!(is_present(cache.path(), "google-t5/t5-small"));
    }

    #[test]
    fn missing_diffs_desired_against_complete_models() {
        let cache = tempfile::tempdir().expect("tempdir");
        let desired = vec![
            ModelSpec::new("google-t5/t5-small"),
            ModelSpec::new("Qwen/Qwen2.5-7B"),
        ];
        // Nothing on disk: everything is missing.
        assert_eq!(missing(&desired, cache.path()).len(), 2);
        // t5 lands complete; only qwen remains.
        write_model(cache.path(), "google-t5/t5-small", "rev1", true);
        let still = missing(&desired, cache.path());
        assert_eq!(still.len(), 1);
        assert_eq!(still[0].model, "Qwen/Qwen2.5-7B");
    }

    #[test]
    fn ensure_complete_stamps_origin_downloads() {
        let cache = tempfile::tempdir().expect("tempdir");
        // Simulate an origin download: files on disk, no sentinel.
        write_model(cache.path(), "google-t5/t5-small", "rev1", false);
        assert!(!is_present(cache.path(), "google-t5/t5-small"));
        ensure_complete(cache.path(), "google-t5/t5-small").expect("stamp");
        assert!(is_present(cache.path(), "google-t5/t5-small"));
        // Idempotent: stamping again is a no-op.
        ensure_complete(cache.path(), "google-t5/t5-small").expect("idempotent");
    }

    #[test]
    fn ensure_complete_errors_when_nothing_landed() {
        let cache = tempfile::tempdir().expect("tempdir");
        let err = ensure_complete(cache.path(), "ghost/model").expect_err("should error");
        assert!(err.to_string().contains("not on disk after fetch"));
    }
}
