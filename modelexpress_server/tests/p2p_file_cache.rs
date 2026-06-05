// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end checks of the cache daemon's registry integration against a real
//! server (in-memory backend), no mocks; the server is the real `run_server`.
//!
//! Two layers are covered:
//! - the registry round trip: a node advertises a FILE_CACHE model and a peer
//!   discovers its NIXL blob through publish -> list -> get and `discover_blob`.
//! - the full reconcile self-heal: node B's [`Reconciler`] observes its empty
//!   cache, discovers node A's residency in the registry, pulls the whole model
//!   over the honest in-process loopback transport (real bytes, real SHA), and
//!   re-advertises so both nodes appear in `ListSources`. The origin-fallback
//!   routing (no peer holds the model) is covered the same way, with a real
//!   local fixture standing in for the origin download.
//!
//! Boots a server, so gated behind `integration-tests`:
//! `cargo test -p modelexpress-server --features integration-tests`.

#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::num::NonZeroU16;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;

use modelexpress_client::cached::desired::ModelSpec;
use modelexpress_client::cached::locator::HfLocator;
use modelexpress_client::cached::reconcile::{Fetcher, Reconciler};
use modelexpress_client::cached::registry::Registry;
use modelexpress_client::cached::transfer::loopback::{Fabric, Loopback};
use modelexpress_client::cached::transfer::puller::Puller;
use modelexpress_client::cached::transfer::stager::CacheServer;
use modelexpress_client::cached::{advertise, discover};
use modelexpress_common::cache::resolve_model_path;
use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
use modelexpress_common::models::ModelProvider;
use modelexpress_server::config::ServerConfig;
use modelexpress_server::run_server;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

type ServerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn ensure_memory_backend() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: set once under `Once`, before any server reads the env.
        unsafe { std::env::set_var("MX_METADATA_BACKEND", "memory") };
    });
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

fn start_server(port: u16) -> (oneshot::Sender<()>, JoinHandle<ServerResult>) {
    ensure_memory_backend();

    let mut config = ServerConfig::default();
    config.server.host = "127.0.0.1".to_string();
    config.server.port = NonZeroU16::new(port).expect("port is non-zero");
    config.cache.eviction.enabled = false;

    let (tx, rx) = oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };
    let handle = tokio::spawn(run_server(config, shutdown));
    (tx, handle)
}

async fn registry_at(port: u16) -> Registry {
    let endpoint = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if let Ok(reg) = Registry::connect(endpoint.clone()).await {
            return reg;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("registry on port {port} never became reachable");
}

async fn stop_and_join(shutdown: oneshot::Sender<()>, handle: JoinHandle<ServerResult>) {
    let _ = shutdown.send(());
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("server task did not exit in time")
        .expect("server task panicked")
        .expect("run_server returned an error");
}

#[tokio::test]
async fn file_cache_publish_list_get_discover_roundtrip() {
    let port = free_port();
    let (shutdown, handle) = start_server(port);
    let mut reg = registry_at(port).await;

    let blob = vec![9u8, 8, 7, 6, 5];
    let identity = advertise::file_cache_identity("google-t5/t5-small", "");
    let worker = advertise::cache_worker(blob.clone(), "agent-node-a", "10.0.0.1:7000");
    let source_id = reg
        .publish(identity.clone(), worker, "node-a")
        .await
        .expect("publish");
    assert_eq!(source_id.len(), 16, "server returns a 16-char source id");

    // ListSources(READY) returns exactly our node, keyed on the same id.
    let instances = reg.list_ready(identity.clone()).await.expect("list");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].mx_source_id, source_id);
    assert_eq!(instances[0].worker_id, "node-a");

    // GetMetadata round-trips the NIXL blob and agent name.
    let got = reg
        .get_worker(source_id.clone(), "node-a")
        .await
        .expect("get")
        .expect("worker found");
    assert_eq!(got.agent_name, "agent-node-a");
    match got.backend_metadata {
        Some(BackendMetadata::NixlMetadata(b)) => assert_eq!(b, blob),
        other => panic!("expected NixlMetadata, got {other:?}"),
    }

    // discover_blob resolves identity -> the same blob a puller would load.
    let discovered = discover::discover_blob(&mut reg, identity, "probe")
        .await
        .expect("discover");
    assert_eq!(discovered.as_deref(), Some(blob.as_slice()));

    // A model nobody holds yields no peer (caller falls back to origin).
    let none = discover::discover_blob(
        &mut reg,
        advertise::file_cache_identity("nobody/holds-this", ""),
        "probe",
    )
    .await
    .expect("discover none");
    assert!(none.is_none());

    stop_and_join(shutdown, handle).await;
}

const MODEL: &str = "google-t5/t5-small";
const REVISION: &str = "rev1";

/// The files a fixture model holds; the puller must reproduce them byte-for-byte.
const FILES: &[(&str, &[u8])] = &[
    ("config.json", b"{\"hidden\": 8}"),
    (
        "model.safetensors",
        b"the quick brown fox jumped over the lazy weights",
    ),
    ("nested/tokenizer.json", b"tok-bytes-here"),
];

/// Lay a model's snapshot down under `cache_root` in the standard HF layout.
fn write_fixture_model(cache_root: &Path) -> PathBuf {
    let dir = cache_root
        .join(format!("models--{}", MODEL.replace('/', "--")))
        .join("snapshots")
        .join(REVISION);
    for (rel, bytes) in FILES {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&path, bytes).expect("write fixture");
    }
    dir
}

/// Where a pulled/fetched revision lands, matching the daemon's resolver.
fn dest_for(cache_root: &Path, model: &str, revision: &str) -> PathBuf {
    resolve_model_path(
        cache_root,
        ModelProvider::HuggingFace,
        model,
        Some(revision),
    )
    .unwrap_or_else(|_| {
        cache_root
            .join(format!("models--{}", model.replace('/', "--")))
            .join("snapshots")
            .join(revision)
    })
}

/// A real [`Fetcher`] for the self-heal tests: peer-pull runs the honest
/// in-process loopback transport against a holder thread (real file IO, real
/// SHA), and origin lays a local fixture down (standing in for the network
/// download, which is HW-validated separately). Counters record the path taken
/// so a test can assert routing without parsing logs. No mocked behavior.
struct LoopbackFetcher {
    fabric: Fabric,
    puller_name: String,
    peer_calls: Arc<AtomicUsize>,
    origin_calls: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl Fetcher for LoopbackFetcher {
    async fn peer_pull(
        &self,
        spec: &ModelSpec,
        holder_md: Vec<u8>,
        dest_root: &Path,
    ) -> anyhow::Result<()> {
        self.peer_calls.fetch_add(1, Ordering::Relaxed);
        let fabric = self.fabric.clone();
        let puller_name = self.puller_name.clone();
        let model = spec.model.clone();
        let dest_root = dest_root.to_path_buf();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mut agent = Loopback::new(&puller_name, &fabric);
            let mut puller = Puller::new(&mut agent, 0, 2, false)?;
            puller.pull(&holder_md, &model, |rev| dest_for(&dest_root, &model, rev))?;
            Ok(())
        })
        .await?
    }

    async fn origin(&self, spec: &ModelSpec, dest_root: &Path) -> anyhow::Result<()> {
        self.origin_calls.fetch_add(1, Ordering::Relaxed);
        // Real local population of the cache layout: the origin download path
        // itself is exercised on hardware, so here we only assert that reconcile
        // routes to origin and converges the cache.
        let dir = dest_for(dest_root, &spec.model, REVISION);
        for (rel, bytes) in FILES {
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)?;
        }
        Ok(())
    }
}

/// Spawn node A's stager thread serving `holder_root` over the loopback fabric.
/// Its loopback metadata blob is just its agent name (what a puller loads), so
/// we return it for the registry advertise without touching the thread.
fn spawn_holder(
    fabric: &Fabric,
    agent_name: &str,
    holder_root: PathBuf,
    stop: Arc<AtomicBool>,
) -> JoinHandle<()> {
    let fabric = fabric.clone();
    let agent_name = agent_name.to_string();
    // A blocking serve loop on a Tokio blocking thread so the async test can run
    // alongside it; the loopback agent never crosses an .await.
    tokio::task::spawn_blocking(move || {
        let mut agent = Loopback::new(&agent_name, &fabric);
        let mut server = CacheServer::new(&mut agent, HfLocator::new(holder_root), 0, false)
            .expect("build cache server");
        server.serve(&stop).expect("serve");
    })
}

/// Read every fixture file back from `snapshot_dir` and assert it round-tripped.
fn assert_model_intact(snapshot_dir: &Path) {
    for (rel, bytes) in FILES {
        let got = std::fs::read(snapshot_dir.join(rel)).expect("read pulled file");
        assert_eq!(got.as_slice(), *bytes, "{rel} round-tripped byte-for-byte");
    }
}

/// Two-node self-heal: node A holds the model and advertises it; node B starts
/// with an empty cache, reconciles, pulls the whole snapshot from A over the
/// loopback transport, and both nodes then appear in `ListSources`.
#[tokio::test]
async fn reconcile_peer_pull_self_heals_and_re_advertises() {
    let port = free_port();
    let (shutdown, handle) = start_server(port);
    let mut reg = registry_at(port).await;

    let holder_cache = tempfile::tempdir().expect("holder cache");
    let puller_cache = tempfile::tempdir().expect("puller cache");
    write_fixture_model(holder_cache.path());

    // Node A: serve thread + registry advertise (its blob is its loopback name).
    let fabric = Fabric::default();
    let stop = Arc::new(AtomicBool::new(false));
    let holder_root = holder_cache.path().to_path_buf();
    let holder = spawn_holder(&fabric, "node-a-agent", holder_root, stop.clone());

    let identity = advertise::file_cache_identity(MODEL, "");
    let worker_a = advertise::cache_worker(b"node-a-agent".to_vec(), "node-a-agent", "");
    reg.publish(identity.clone(), worker_a, "node-a")
        .await
        .expect("publish A");

    // Node B: empty cache, reconcile once.
    let peer_calls = Arc::new(AtomicUsize::new(0));
    let origin_calls = Arc::new(AtomicUsize::new(0));
    let fetcher = LoopbackFetcher {
        fabric: fabric.clone(),
        puller_name: "node-b-agent".to_string(),
        peer_calls: peer_calls.clone(),
        origin_calls: origin_calls.clone(),
    };
    let reg_b = registry_at(port).await;
    let mut reconciler = Reconciler::new(
        reg_b,
        puller_cache.path().to_path_buf(),
        b"node-b-agent".to_vec(),
        "node-b-agent",
        "",
        "node-b",
    );
    let advertised = Mutex::new(HashMap::new());
    let desired = vec![ModelSpec::new(MODEL)];
    reconciler
        .reconcile_once(&desired, &fetcher, &advertised)
        .await
        .expect("reconcile");

    // B pulled from the peer, not origin.
    assert_eq!(peer_calls.load(Ordering::Relaxed), 1, "pulled from peer");
    assert_eq!(
        origin_calls.load(Ordering::Relaxed),
        0,
        "origin not touched"
    );

    // The whole snapshot landed byte-for-byte in B's cache.
    assert_model_intact(&dest_for(puller_cache.path(), MODEL, REVISION));

    // Both nodes are now READY for the model.
    let instances = reg.list_ready(identity).await.expect("list");
    let workers: Vec<&str> = instances.iter().map(|i| i.worker_id.as_str()).collect();
    assert_eq!(
        instances.len(),
        2,
        "both A and B advertise the model: {workers:?}"
    );
    assert!(workers.contains(&"node-a"));
    assert!(workers.contains(&"node-b"));

    stop.store(true, Ordering::Relaxed);
    holder.await.expect("holder thread");
    stop_and_join(shutdown, handle).await;
}

/// No peer holds the model, so reconcile routes to origin, converges the cache,
/// and advertises the node. A single instance ends up in `ListSources`.
#[tokio::test]
async fn reconcile_origin_fallback_converges_and_advertises() {
    let port = free_port();
    let (shutdown, handle) = start_server(port);
    let mut reg = registry_at(port).await;

    let cache = tempfile::tempdir().expect("cache");
    let peer_calls = Arc::new(AtomicUsize::new(0));
    let origin_calls = Arc::new(AtomicUsize::new(0));
    let fetcher = LoopbackFetcher {
        fabric: Fabric::default(),
        puller_name: "node-c-agent".to_string(),
        peer_calls: peer_calls.clone(),
        origin_calls: origin_calls.clone(),
    };
    let reg_c = registry_at(port).await;
    let mut reconciler = Reconciler::new(
        reg_c,
        cache.path().to_path_buf(),
        b"node-c-agent".to_vec(),
        "node-c-agent",
        "",
        "node-c",
    );
    let advertised = Mutex::new(HashMap::new());
    let desired = vec![ModelSpec::new(MODEL)];
    reconciler
        .reconcile_once(&desired, &fetcher, &advertised)
        .await
        .expect("reconcile");

    // Nobody advertised it, so the origin path ran.
    assert_eq!(
        origin_calls.load(Ordering::Relaxed),
        1,
        "fetched from origin"
    );
    assert_eq!(
        peer_calls.load(Ordering::Relaxed),
        0,
        "no peer to pull from"
    );

    // The model converged and the node advertises it (exactly one instance).
    assert_model_intact(&dest_for(cache.path(), MODEL, REVISION));
    let identity = advertise::file_cache_identity(MODEL, "");
    let instances = reg.list_ready(identity).await.expect("list");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].worker_id, "node-c");

    stop_and_join(shutdown, handle).await;
}

/// On shutdown the daemon marks its advertised sources STALE so peers stop
/// selecting it at once, instead of waiting out the reaper. After deregister,
/// `ListSources(READY)` no longer returns the node.
#[tokio::test]
async fn reconcile_deregister_marks_sources_stale() {
    let port = free_port();
    let (shutdown, handle) = start_server(port);
    let mut reg = registry_at(port).await;

    // The model is already held-and-complete, so reconcile only advertises it
    // (no fetch); the fetcher is present but never called.
    let cache = tempfile::tempdir().expect("cache");
    write_fixture_model(cache.path());
    modelexpress_client::cached::reconcile::ensure_complete(cache.path(), MODEL).expect("complete");

    let fetcher = LoopbackFetcher {
        fabric: Fabric::default(),
        puller_name: "node-d-agent".to_string(),
        peer_calls: Arc::new(AtomicUsize::new(0)),
        origin_calls: Arc::new(AtomicUsize::new(0)),
    };
    let reg_d = registry_at(port).await;
    let mut reconciler = Reconciler::new(
        reg_d,
        cache.path().to_path_buf(),
        b"node-d-agent".to_vec(),
        "node-d-agent",
        "",
        "node-d",
    );
    let advertised = Mutex::new(HashMap::new());
    let desired = vec![ModelSpec::new(MODEL)];
    reconciler
        .reconcile_once(&desired, &fetcher, &advertised)
        .await
        .expect("reconcile");

    // Advertised and READY.
    let identity = advertise::file_cache_identity(MODEL, "");
    assert_eq!(
        reg.list_ready(identity.clone()).await.expect("list").len(),
        1
    );

    // Deregister marks it STALE; it drops out of the READY listing.
    reconciler.deregister(&advertised).await;
    assert!(
        reg.list_ready(identity).await.expect("list").is_empty(),
        "stale source must not appear as a READY holder"
    );

    stop_and_join(shutdown, handle).await;
}
