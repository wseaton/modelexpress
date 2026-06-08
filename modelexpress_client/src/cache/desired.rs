// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The declared desired set: the models a node is supposed to hold. The
//! reconciler diffs this against what is actually on local NVMe and pulls the
//! difference. A [`DesiredSet`] is the only thing that varies between deployment
//! styles, so the reconciler depends on the trait, not on where the list comes
//! from. Two sources exist: [`StaticDesiredSet`] (the CLI `--model` list, fixed
//! for the process's life) and [`FileDesiredSet`] (a file - a mounted K8s
//! ConfigMap - re-read every pass, so editing the ConfigMap reconverges the
//! fleet with no restart). A CRD/label-backed source is a later concern.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use modelexpress_common::models::ModelProvider;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::Notify;

/// One model the node should hold. The cache daemon is HuggingFace-only in v1
/// (the [`crate::cache::locator::HfLocator`] resolves the on-disk layout), so
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

    /// Parse a `model` or `model@revision` spec (the form used on the CLI and in
    /// the desired-set file). A trailing `@` with no revision is treated as
    /// unpinned.
    pub fn parse(spec: &str) -> Self {
        match spec.split_once('@') {
            Some((model, revision)) if !revision.is_empty() => Self {
                model: model.to_string(),
                revision: Some(revision.to_string()),
            },
            _ => Self::new(spec.trim_end_matches('@')),
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
///
/// `Send + Sync` because the reconcile loop holds it as a `Box<dyn DesiredSet>`
/// across `.await` points. `desired()` is async because a dynamic source can do
/// I/O (the registry-backed set queries the P2P server); the static and file
/// sources just return ready.
#[tonic::async_trait]
pub trait DesiredSet: Send + Sync {
    /// Snapshot the currently-desired models. Called once per reconcile pass, so
    /// a dynamic implementation can return a fresh view each time.
    async fn desired(&self) -> Vec<ModelSpec>;
}

/// Append `spec` unless its model is already present, so a set stays
/// de-duplicated by model (one revision per model in v1) and order-stable.
fn push_unique(specs: &mut Vec<ModelSpec>, spec: ModelSpec) {
    if !specs.iter().any(|existing| existing.model == spec.model) {
        specs.push(spec);
    }
}

/// Parse a desired-set file body: one `model` or `model@revision` per line,
/// blank lines and `#` comments skipped, de-duplicated by model.
fn parse_lines(body: &str) -> Vec<ModelSpec> {
    let mut specs = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        push_unique(&mut specs, ModelSpec::parse(line));
    }
    specs
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
            push_unique(&mut specs, ModelSpec::parse(&model.into()));
        }
        Self { specs }
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

#[tonic::async_trait]
impl DesiredSet for StaticDesiredSet {
    async fn desired(&self) -> Vec<ModelSpec> {
        self.specs.clone()
    }
}

/// A desired set read from a file every pass. Backs a mounted K8s ConfigMap:
/// editing the ConfigMap (one `model` / `model@revision` per line) reconverges
/// the fleet with no restart. An unreadable file yields an empty set (logged),
/// not an error, so a not-yet-mounted or transiently-missing file just defers
/// reconciliation rather than crashing the daemon.
#[derive(Debug, Clone)]
pub struct FileDesiredSet {
    path: PathBuf,
}

impl FileDesiredSet {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[tonic::async_trait]
impl DesiredSet for FileDesiredSet {
    async fn desired(&self) -> Vec<ModelSpec> {
        match std::fs::read_to_string(&self.path) {
            Ok(body) => parse_lines(&body),
            Err(e) => {
                tracing::warn!(path = %self.path.display(), error = %e, "desired-set file unreadable; treating as empty this pass");
                Vec::new()
            }
        }
    }
}

/// A desired set that unions a base set with every model the P2P registry
/// reports any peer holding. This is the auto-expanding cache: a model used on
/// any node gets advertised, lands in the registry, and so enters every node's
/// desired set, which makes the whole fleet converge to hold it. The base set
/// (a ConfigMap or `--model` list) stays pinned as the always-present floor;
/// the registry union is the part that grows with usage.
///
/// A registry query failure degrades to the base set for that pass (logged),
/// not an error, so a transient server blip can't wipe the desired set.
pub struct RegistryDesiredSet {
    base: Box<dyn DesiredSet>,
    registry: crate::cache::registry::Registry,
}

impl RegistryDesiredSet {
    pub fn new(base: Box<dyn DesiredSet>, registry: crate::cache::registry::Registry) -> Self {
        Self { base, registry }
    }
}

#[tonic::async_trait]
impl DesiredSet for RegistryDesiredSet {
    async fn desired(&self) -> Vec<ModelSpec> {
        // Base first so a pinned-revision base entry wins over the registry's
        // unpinned one (push_unique keeps the first per model).
        let mut specs = self.base.desired().await;
        // Registry handle is cheap to clone (shares the channel); cloning keeps
        // desired() on &self despite the gRPC call needing &mut.
        let mut registry = self.registry.clone();
        match registry.list_ready_models().await {
            Ok(models) => {
                for model in models {
                    push_unique(&mut specs, ModelSpec::new(model));
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "registry model list failed; using base desired set this pass");
            }
        }
        specs
    }
}

/// Watch the desired-set file for changes and signal `on_change` whenever it is
/// touched, so the reconcile loop can converge immediately instead of waiting
/// for its next interval tick. Returns the watcher guard; the caller must keep
/// it alive (dropping it stops the watch).
///
/// We watch the file's *parent directory*, not the file itself. A mounted K8s
/// ConfigMap is not a plain file: the kubelet stages new content in a timestamped
/// directory and atomically swaps a `..data` symlink
/// (`models -> ..data/models -> ..2026_.../models`). An inotify watch on the file
/// inode would go deaf after that swap (the old inode is now orphaned), whereas a
/// directory watch keeps firing across swaps. Reads through the symlink are always
/// atomic, so a reconcile triggered by an event sees complete content.
pub fn watch_desired_file(
    path: &Path,
    on_change: Arc<Notify>,
) -> notify::Result<RecommendedWatcher> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        match res {
            // Any event in the directory may be the ConfigMap swap; let the
            // reconcile loop re-read and diff. `notify_one` coalesces a burst of
            // events into a single pass.
            Ok(_) => on_change.notify_one(),
            Err(e) => tracing::warn!(error = %e, "desired-set watcher error"),
        }
    })?;
    watcher.watch(dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

/// Watch a cache root *recursively* and signal `on_change` on any change beneath
/// it, so auto-expand capture can react within seconds of a co-located process
/// (e.g. a vLLM) downloading a model into the shared cache, instead of waiting
/// for the next interval tick. Recursive because downloads land files deep in
/// the `models--*/blobs` and `snapshots/` tree, not at the root. A multi-file
/// download fires a burst of events; `notify_one` coalesces them into a single
/// pending wake, and the loop debounces and re-checks settledness before
/// advertising, so an in-progress download is never published early. Returns the
/// watcher guard; the caller keeps it alive (dropping it stops the watch).
pub fn watch_cache_dir(dir: &Path, on_change: Arc<Notify>) -> notify::Result<RecommendedWatcher> {
    let mut watcher =
        notify::recommended_watcher(move |res: notify::Result<notify::Event>| match res {
            Ok(_) => on_change.notify_one(),
            Err(e) => tracing::warn!(error = %e, "cache-dir watcher error"),
        })?;
    watcher.watch(dir, RecursiveMode::Recursive)?;
    Ok(watcher)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_set_dedups_and_preserves_order() {
        let set = StaticDesiredSet::from_models([
            "google-t5/t5-small",
            "Qwen/Qwen2.5-7B",
            "google-t5/t5-small",
        ]);
        let desired = set.desired().await;
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

    #[test]
    fn parse_handles_pinned_and_unpinned_revisions() {
        let pinned = ModelSpec::parse("google-t5/t5-small@abc123");
        assert_eq!(pinned.model, "google-t5/t5-small");
        assert_eq!(pinned.revision.as_deref(), Some("abc123"));
        assert_eq!(pinned.identity_revision(), "abc123");

        let unpinned = ModelSpec::parse("Qwen/Qwen2.5-7B");
        assert_eq!(unpinned.model, "Qwen/Qwen2.5-7B");
        assert!(unpinned.revision.is_none());

        // A trailing `@` with no revision is unpinned, not a revision named "".
        let trailing = ModelSpec::parse("org/m@");
        assert_eq!(trailing.model, "org/m");
        assert!(trailing.revision.is_none());
    }

    #[test]
    fn parse_lines_skips_comments_blanks_and_dedups() {
        let body = "\
            # desired models\n\
            google-t5/t5-small\n\
            \n\
            Qwen/Qwen2.5-7B@rev9\n\
            google-t5/t5-small\n\
            # trailing comment\n";
        let specs = parse_lines(body);
        let models: Vec<&str> = specs.iter().map(|s| s.model.as_str()).collect();
        assert_eq!(models, vec!["google-t5/t5-small", "Qwen/Qwen2.5-7B"]);
        assert_eq!(specs[1].revision.as_deref(), Some("rev9"));
    }

    #[tokio::test]
    async fn file_desired_set_reads_live_and_tolerates_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("models.txt");
        // Missing file -> empty, no panic.
        let set = FileDesiredSet::new(&path);
        assert!(set.desired().await.is_empty());

        std::fs::write(&path, "google-t5/t5-small\n").expect("write");
        assert_eq!(set.desired().await.len(), 1);

        // Re-reads each call: a ConfigMap edit is picked up without restart.
        std::fs::write(&path, "google-t5/t5-small\nQwen/Qwen2.5-7B\n").expect("rewrite");
        assert_eq!(set.desired().await.len(), 2);
    }

    #[tokio::test]
    async fn watch_signals_on_file_change() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("models");
        std::fs::write(&path, "google-t5/t5-small\n").expect("seed");

        let changed = Arc::new(Notify::new());
        // Keep the guard alive for the duration of the test.
        let _watcher = watch_desired_file(&path, changed.clone()).expect("watch");

        // Editing the file must wake the waiter. A generous timeout absorbs
        // filesystem-event latency without flaking on a loaded CI box.
        std::fs::write(&path, "google-t5/t5-small\nQwen/Qwen2.5-7B\n").expect("edit");
        tokio::time::timeout(std::time::Duration::from_secs(5), changed.notified())
            .await
            .expect("watcher did not fire on file change");
    }

    #[tokio::test]
    async fn cache_watch_signals_on_nested_write() {
        let root = tempfile::tempdir().expect("tempdir");
        let changed = Arc::new(Notify::new());
        let _watcher = watch_cache_dir(root.path(), changed.clone()).expect("watch");

        // A download writes deep in the tree (models--*/blobs/...); the recursive
        // watch must fire, not just on top-level changes.
        let nested = root.path().join("models--org--m").join("blobs");
        std::fs::create_dir_all(&nested).expect("mkdir");
        std::fs::write(nested.join("abc123"), b"weights").expect("write");
        tokio::time::timeout(std::time::Duration::from_secs(5), changed.notified())
            .await
            .expect("cache watcher did not fire on nested write");
    }
}
