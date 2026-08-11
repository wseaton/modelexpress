// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `modelexpress-cache`: the self-healing cache daemon.
//!
//! Three modes:
//! - `--serve --model <m>...`: advertise the held models to the P2P registry and
//!   serve their snapshots to pullers over RDMA.
//! - `--pull --model <m>`: discover a peer holding the model through the registry
//!   and pull it into the local cache (one-shot).
//! - `--reconcile --model <m>...`: the full self-healing loop. Serve held models
//!   AND, on an interval, converge the local cache toward the desired `--model`
//!   set: pull each missing model from a peer (or origin if none holds it) and
//!   advertise residency so other nodes can pull from this one.
//!
//! All require the `nixl` build. `NixlAgent` is `!Send`, so it is confined to a
//! dedicated thread (serve) or a `spawn_blocking` worker (pull) and never held
//! across an `.await`; only its metadata blob crosses to the async side that
//! talks to the registry. In `--reconcile` the serve agent and the per-pull
//! agent are co-resident in one process.

#[cfg(feature = "nixl")]
use std::ops::ControlFlow;
use std::path::PathBuf;

use clap::Parser;

/// Self-healing cache daemon: reconciles the local cache toward a desired set by
/// pulling missing models from peers over RDMA, and serves its own cache out.
#[derive(Debug, Parser)]
#[command(name = "modelexpress-cache", version, about)]
struct Cli {
    /// Advertise and serve the named models to peers.
    #[arg(long)]
    serve: bool,

    /// One-shot: pull the named model from a peer found via the registry.
    #[arg(long)]
    pull: bool,

    /// Run the self-healing loop: serve held models and converge the local
    /// cache toward the `--model` set (peer-pull, else origin), advertising what
    /// this node holds.
    #[arg(long)]
    reconcile: bool,

    /// Model to serve, pull, or converge toward (repeat for several; `--pull`
    /// uses the first).
    #[arg(long = "model")]
    models: Vec<String>,

    /// `--reconcile` only: path to a desired-set file (one `model` or
    /// `model@revision` per line, `#` comments allowed), re-read every pass so a
    /// mounted ConfigMap can be edited to reconverge the fleet without a restart.
    /// Takes precedence over `--model` when set.
    #[arg(long, env = "MODEL_EXPRESS_CACHE_MODELS_FILE")]
    models_file: Option<PathBuf>,

    /// Seconds between reconcile passes in `--reconcile` mode.
    #[arg(long, env = "MODEL_EXPRESS_CACHE_RECONCILE_SECS", default_value_t = 60)]
    reconcile_secs: u64,

    /// `--reconcile` only: grow the desired set to include every model any peer
    /// advertises to the registry, on top of the `--model`/`--models-file` base.
    /// A model used on any node then replicates fleet-wide automatically. Off by
    /// default; the cache only grows while it is set (no eviction yet).
    #[arg(long, env = "MODEL_EXPRESS_CACHE_AUTO_EXPAND")]
    auto_expand: bool,

    /// `--auto-expand` only: a model that is not pinned (in the base set) and has
    /// not been used locally for this many seconds stops being advertised, decays
    /// out of the registry, and is then evicted. Default 7 days.
    #[arg(
        long,
        env = "MODEL_EXPRESS_CACHE_DEMAND_TTL_SECS",
        default_value_t = 7 * 24 * 3600
    )]
    demand_ttl_secs: u64,

    /// `--auto-expand` only: a model touched within this many seconds is never
    /// evicted, a guard against deleting one just pulled or in use. Default 6h.
    #[arg(
        long,
        env = "MODEL_EXPRESS_CACHE_GC_GRACE_SECS",
        default_value_t = 6 * 3600
    )]
    gc_grace_secs: u64,

    /// `--auto-expand` only: when capturing a model downloaded into the cache by
    /// another process, wait until its newest file is this many seconds old (a
    /// backstop behind the weights-index check) so a paused-but-unfinished
    /// download is not advertised. Default 15s.
    #[arg(
        long,
        env = "MODEL_EXPRESS_CACHE_CAPTURE_QUIESCENCE_SECS",
        default_value_t = 15
    )]
    capture_quiescence_secs: u64,

    /// NIXL agent name for this process.
    #[arg(long, default_value = "mx-cache")]
    name: String,

    /// NIXL listen port for the serving agent.
    #[arg(long, env = "MODEL_EXPRESS_CACHE_NIXL_PORT", default_value_t = 7000)]
    nixl_port: u16,

    /// P2P registry endpoint (the ModelExpress server).
    #[arg(
        long,
        env = "MODEL_EXPRESS_ENDPOINT",
        default_value = "http://localhost:8001"
    )]
    endpoint: String,

    /// Bounded staging-buffer size in GiB (the DRAM cap for a transfer).
    #[arg(long, env = "MODEL_EXPRESS_CACHE_BUF_GIB", default_value_t = 4)]
    buf_gib: u32,

    /// Receive pipeline depth: the staging buffer is carved into this many slots
    /// so a shard's NVMe write overlaps the next shard's RDMA receive. Each slot
    /// must hold the largest shard, so raising depth needs proportionally more
    /// `--buf-gib`. 2 is the validated double-buffer.
    #[arg(long, env = "MODEL_EXPRESS_CACHE_POOL_DEPTH", default_value_t = 2)]
    pool_depth: usize,

    /// Concurrent NVMe write streams per shard: each posted write is striped
    /// across this many transfer requests so the RAID absorbs parallel writers.
    /// Measured on the target array (uring queue): 1 stream 2.85 GB/s, 4
    /// streams 3.0 GB/s (the disk's aggregate ceiling); more buys nothing.
    #[arg(
        long,
        env = modelexpress_common::envs::MODEL_EXPRESS_CACHE_WRITE_STREAMS,
        default_value_t = 4
    )]
    write_streams: usize,

    /// Write pulled files with `O_DIRECT`, bypassing the page cache. Much faster
    /// for the sustained large writes the puller does (block-aligned, with the
    /// partial tail truncated back). Requires a filesystem that supports
    /// `O_DIRECT` (real NVMe does; tmpfs does not).
    #[arg(long, env = "MODEL_EXPRESS_CACHE_O_DIRECT")]
    o_direct: bool,

    /// Local model cache root. Defaults to the standard ModelExpress cache
    /// discovery (`MODEL_EXPRESS_CACHE_DIRECTORY`, config file, or `~`).
    #[arg(long, env = "MODEL_EXPRESS_CACHE_DIRECTORY")]
    cache_dir: Option<PathBuf>,
}

/// Span-aware subscriber: `RUST_LOG` controls levels (default `info`), and span
/// close events print each span's busy/idle time. The puller wraps its transfer
/// legs (recv/hash/write/sync) in `debug` spans, so `RUST_LOG=...puller=debug`
/// surfaces a per-leg timing breakdown without any hand-rolled timers.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::format::FmtSpan;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(FmtSpan::CLOSE)
        .init();
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    init_tracing();
    let cli = Cli::parse();
    match run(&cli).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run(cli: &Cli) -> anyhow::Result<()> {
    match (cli.serve, cli.pull, cli.reconcile) {
        (true, false, false) => run_serve(cli).await,
        (false, true, false) => run_pull(cli).await,
        (false, false, true) => run_reconcile(cli).await,
        (false, false, false) => anyhow::bail!("specify one of --serve, --pull, or --reconcile"),
        _ => anyhow::bail!("--serve, --pull, and --reconcile are mutually exclusive"),
    }
}

/// Resolve the cache root from the flag, falling back to standard discovery.
#[cfg(feature = "nixl")]
fn cache_root(cli: &Cli) -> anyhow::Result<PathBuf> {
    use modelexpress_common::cache::CacheConfig;
    match &cli.cache_dir {
        Some(path) => Ok(path.clone()),
        None => Ok(CacheConfig::discover()?.local_path),
    }
}

/// `POD_IP:port` for the NIXL listen thread, or empty if `POD_IP` is unset. The
/// metadata blob carries the connection info, so this is advisory.
#[cfg(feature = "nixl")]
fn metadata_endpoint(port: u16) -> String {
    match std::env::var("POD_IP") {
        Ok(ip) if !ip.is_empty() => format!("{ip}:{port}"),
        _ => String::new(),
    }
}

#[cfg(feature = "nixl")]
async fn wait_for_shutdown() {
    use tokio::signal::unix::{SignalKind, signal};
    match (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
    ) {
        (Ok(mut term), Ok(mut int)) => {
            tokio::select! {
                _ = term.recv() => {}
                _ = int.recv() => {}
            }
        }
        _ => {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

/// Spawn the serve agent on its own thread (the `!Send` `NixlAgent` lives only
/// there) and block until it hands back its metadata blob. Shared by `--serve`
/// and `--reconcile`, which both serve held models out while doing their own
/// async work against the registry. Flipping `stop` ends the serve loop.
#[cfg(feature = "nixl")]
fn start_serve_thread(
    name: String,
    port: u16,
    buf_gib: u32,
    cache_root: PathBuf,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<(std::thread::JoinHandle<anyhow::Result<()>>, Vec<u8>)> {
    use modelexpress_client::cache::locator::HfLocator;
    use modelexpress_client::cache::transfer::{nixl::NixlAgent, stager::CacheServer};

    let (md_tx, md_rx) = std::sync::mpsc::channel::<anyhow::Result<Vec<u8>>>();
    let serve_thread = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut agent = NixlAgent::new(&name, port)?;
        let locator = HfLocator::new(cache_root);
        let mut server = match CacheServer::new(&mut agent, locator, buf_gib, false) {
            Ok(server) => server,
            Err(e) => {
                let _ = md_tx.send(Err(e));
                return Ok(());
            }
        };
        match server.local_md() {
            Ok(md) => {
                let _ = md_tx.send(Ok(md));
            }
            Err(e) => {
                let _ = md_tx.send(Err(e));
                return Ok(());
            }
        }
        server.serve(&stop)
    });

    match md_rx.recv() {
        Ok(Ok(md)) => Ok((serve_thread, md)),
        Ok(Err(e)) => {
            let _ = serve_thread.join();
            Err(e)
        }
        Err(_) => {
            serve_thread
                .join()
                .map_err(|_| anyhow::anyhow!("serve thread panicked"))??;
            Err(anyhow::anyhow!(
                "serve thread exited before sending metadata"
            ))
        }
    }
}

#[cfg(feature = "nixl")]
async fn run_serve(cli: &Cli) -> anyhow::Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use modelexpress_client::cache::advertise;
    use modelexpress_client::cache::locator::locate_hf;
    use modelexpress_client::cache::registry::Registry;
    use modelexpress_common::grpc::p2p::SourceStatus;

    if cli.models.is_empty() {
        anyhow::bail!("--serve needs at least one --model");
    }
    let cache_root = cache_root(cli)?;
    let stop = Arc::new(AtomicBool::new(false));
    let (serve_thread, md) = start_serve_thread(
        cli.name.clone(),
        cli.nixl_port,
        cli.buf_gib,
        cache_root.clone(),
        stop.clone(),
    )?;

    let mut registry = Registry::connect(cli.endpoint.clone()).await?;
    let worker_id = advertise::worker_id();
    let endpoint = metadata_endpoint(cli.nixl_port);
    let mut published: Vec<(String, String)> = Vec::new();
    for model in &cli.models {
        if locate_hf(&cache_root, model).is_none() {
            tracing::warn!(model, "not held locally; skipping advertise");
            continue;
        }
        let identity = advertise::file_cache_identity(model.clone(), "");
        let worker = advertise::cache_worker(md.clone(), cli.name.clone(), endpoint.clone());
        let source_id = registry
            .publish(identity, worker, worker_id.clone())
            .await?;
        tracing::info!(model, source_id, "advertised");
        published.push((source_id, model.clone()));
    }

    // Heartbeat each advertised source within the reaper's window.
    let mut heartbeat = registry.clone();
    let hb_published = published.clone();
    let hb_id = worker_id.clone();
    let hb_stop = stop.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            tick.tick().await;
            if hb_stop.load(Ordering::Relaxed) {
                break;
            }
            for (source_id, _) in &hb_published {
                if let Err(e) = heartbeat
                    .update_status(source_id.clone(), 0, SourceStatus::Ready, hb_id.clone())
                    .await
                {
                    tracing::warn!(source_id, error = %e, "heartbeat failed");
                }
            }
        }
    });

    tracing::info!(models = published.len(), "serving; waiting for shutdown");
    wait_for_shutdown().await;
    stop.store(true, Ordering::Relaxed);
    serve_thread
        .join()
        .map_err(|_| anyhow::anyhow!("serve thread panicked"))??;
    Ok(())
}

#[cfg(feature = "nixl")]
async fn run_pull(cli: &Cli) -> anyhow::Result<()> {
    use anyhow::Context;

    use modelexpress_client::cache::registry::Registry;
    use modelexpress_client::cache::transfer::{nixl::NixlAgent, puller::Puller};
    use modelexpress_client::cache::{advertise, discover};
    use modelexpress_common::cache::resolve_model_path;
    use modelexpress_common::models::ModelProvider;

    let cache_root = cache_root(cli)?;
    let model = cli
        .models
        .first()
        .context("--pull needs a --model")?
        .clone();
    if cli.models.len() > 1 {
        tracing::warn!("only the first --model is pulled in this phase");
    }

    let mut registry = Registry::connect(cli.endpoint.clone()).await?;
    let identity = advertise::file_cache_identity(model.clone(), "");
    let blob = discover::discover_blob(&mut registry, identity, &cli.name)
        .await?
        .with_context(|| format!("no peer advertises {model}"))?;

    let name = cli.name.clone();
    let buf_gib = cli.buf_gib;
    let pool_depth = cli.pool_depth;
    let o_direct = cli.o_direct;
    let write_streams = cli.write_streams;
    let pull_model = model.clone();
    // NixlAgent lives only on the blocking thread, never across an .await.
    let dest = tokio::task::spawn_blocking(move || -> anyhow::Result<PathBuf> {
        let mut agent = NixlAgent::new(&name, 0)?.with_write_streams(write_streams);
        let mut puller = Puller::new(&mut agent, buf_gib, pool_depth, o_direct)?;
        let summary = puller.pull(&blob, &pull_model, |rev| {
            resolve_model_path(
                &cache_root,
                ModelProvider::HuggingFace,
                &pull_model,
                Some(rev),
            )
            .unwrap_or_else(|_| {
                cache_root
                    .join(format!("models--{}", pull_model.replace('/', "--")))
                    .join("snapshots")
                    .join(rev)
            })
        })?;
        Ok(summary.dest)
    })
    .await??;

    tracing::info!(model, dest = %dest.display(), "pull complete");
    Ok(())
}

#[cfg(feature = "nixl")]
async fn run_reconcile(cli: &Cli) -> anyhow::Result<()> {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use modelexpress_client::cache::advertise;
    use modelexpress_client::cache::desired::{
        DesiredSet, FileDesiredSet, RegistryDesiredSet, StaticDesiredSet, desired_file_fingerprint,
        watch_cache_dir, watch_desired_file,
    };
    use modelexpress_client::cache::reconcile::{NixlFetcher, Reconciler};
    use modelexpress_client::cache::registry::Registry;
    use modelexpress_common::grpc::p2p::SourceStatus;
    use tokio::sync::Notify;

    if cli.models.is_empty() && cli.models_file.is_none() {
        anyhow::bail!("--reconcile needs --model or --models-file");
    }
    // The base desired set is re-read every pass, so a mounted ConfigMap (via
    // --models-file) can be edited to reconverge the fleet without a restart.
    let base_desired: Arc<dyn DesiredSet> = match &cli.models_file {
        Some(path) => Arc::new(FileDesiredSet::new(path.clone())),
        None => Arc::new(StaticDesiredSet::from_models(cli.models.clone())),
    };

    // Watch the desired-set file so a ConfigMap edit reconverges in ~1s instead
    // of waiting up to a full interval. The interval tick stays as the backstop
    // (peers going STALE, transient pull failures). A failed watch setup is
    // non-fatal: we just fall back to interval-only reconciliation.
    let changed = Arc::new(Notify::new());
    let _watcher = match &cli.models_file {
        Some(path) => match watch_desired_file(path, changed.clone()) {
            Ok(w) => {
                tracing::info!(path = %path.display(), "watching desired-set file for changes");
                Some(w)
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to watch desired-set file; interval-only");
                None
            }
        },
        None => None,
    };
    let cache_root = cache_root(cli)?;
    let stop = Arc::new(AtomicBool::new(false));
    let (serve_thread, md) = start_serve_thread(
        cli.name.clone(),
        cli.nixl_port,
        cli.buf_gib,
        cache_root.clone(),
        stop.clone(),
    )?;

    // With --auto-expand, watch the cache root so a model a co-located process
    // (e.g. vLLM) downloads into it is captured within seconds rather than at the
    // next interval tick. The recursive watch needs the root to exist, so create
    // it first; a failed setup degrades to interval-only capture, not an error.
    let cache_changed = Arc::new(Notify::new());
    let _cache_watcher = if cli.auto_expand {
        if let Err(e) = std::fs::create_dir_all(&cache_root) {
            tracing::warn!(path = %cache_root.display(), error = %e, "could not create cache root to watch");
        }
        match watch_cache_dir(&cache_root, cache_changed.clone()) {
            Ok(w) => {
                tracing::info!(path = %cache_root.display(), "watching cache root for new models");
                Some(w)
            }
            Err(e) => {
                tracing::warn!(path = %cache_root.display(), error = %e, "failed to watch cache root; capture is interval-only");
                None
            }
        }
    } else {
        None
    };

    let registry = Registry::connect(cli.endpoint.clone()).await?;
    let worker_id = advertise::worker_id();
    let endpoint = metadata_endpoint(cli.nixl_port);

    // With --auto-expand, union the base set with every model the registry
    // reports a peer holding, so usage anywhere grows the whole fleet's cache.
    let desired_set: Arc<dyn DesiredSet> = if cli.auto_expand {
        tracing::info!(
            demand_ttl_secs = cli.demand_ttl_secs,
            "auto-expand on: desired set = base + every model any peer advertises; unused models decay + evict"
        );
        Arc::new(RegistryDesiredSet::new(
            base_desired.clone(),
            registry.clone(),
        ))
    } else {
        base_desired.clone()
    };

    // The advertised map is shared with the heartbeat task. The reconcile loop
    // is its only writer and locks it only momentarily, so a long pull never
    // blocks heartbeats: model name -> mx_source_id.
    let advertised: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

    // Heartbeat advertised sources within the reaper's window. A background task
    // with nothing to drain, so it is stopped by aborting its handle on shutdown
    // rather than polling a flag (which would lag shutdown by a whole tick).
    let mut heartbeat = registry.clone();
    let hb_advertised = advertised.clone();
    let hb_id = worker_id.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(30));
        loop {
            tick.tick().await;
            let snapshot: Vec<String> = match hb_advertised.lock() {
                Ok(map) => map.values().cloned().collect(),
                Err(_) => {
                    tracing::error!("advertised map poisoned; stopping heartbeat");
                    break;
                }
            };
            for source_id in snapshot {
                if let Err(e) = heartbeat
                    .update_status(source_id.clone(), 0, SourceStatus::Ready, hb_id.clone())
                    .await
                {
                    tracing::warn!(source_id, error = %e, "heartbeat failed");
                }
            }
        }
    });

    let mut reconciler = Reconciler::new(
        registry,
        cache_root,
        md,
        cli.name.clone(),
        endpoint,
        worker_id,
    );
    if cli.auto_expand {
        use modelexpress_client::cache::usage::AtimeUsage;
        reconciler = reconciler
            .with_bounding(
                std::sync::Arc::new(AtimeUsage),
                Duration::from_secs(cli.demand_ttl_secs),
                Duration::from_secs(cli.gc_grace_secs),
            )
            .with_capture_quiescence(Duration::from_secs(cli.capture_quiescence_secs));
    }
    let fetcher = NixlFetcher {
        agent_name: cli.name.clone(),
        buf_gib: cli.buf_gib,
        pool_depth: cli.pool_depth,
        direct: cli.o_direct,
        write_streams: cli.write_streams,
        endpoint: cli.endpoint.clone(),
    };

    tracing::info!(
        interval_secs = cli.reconcile_secs,
        "reconcile loop started; serving and converging"
    );
    // A co-located download settles before its on-disk fingerprint stops
    // changing; observe twice spaced by this window so the stability check can
    // confirm completion before advertising.
    let cache_settle = Duration::from_secs(2);
    // Fingerprint of the desired-set file as last reconciled, so a kubelet
    // ConfigMap resync that re-fires the watch without changing the body is a
    // no-op instead of a full pass. Seeded with the current content: startup
    // convergence is driven by the immediate interval tick below, so the watch
    // only acts on genuine post-startup edits.
    let mut desired_fp = cli
        .models_file
        .as_deref()
        .and_then(desired_file_fingerprint);
    let shutdown = wait_for_shutdown();
    tokio::pin!(shutdown);
    let mut tick = tokio::time::interval(Duration::from_secs(cli.reconcile_secs));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let flow = tokio::select! {
            _ = tick.tick() => {
                reconcile_pass(&mut reconciler, desired_set.as_ref(), base_desired.as_ref(), &fetcher, &advertised, cli.auto_expand).await;
                ControlFlow::Continue(())
            }
            _ = changed.notified() => {
                // Coalesce the kubelet's symlink-swap burst (and any rapid edits)
                // into a single pass before re-reading.
                tokio::time::sleep(Duration::from_millis(500)).await;
                let fp = cli.models_file.as_deref().and_then(desired_file_fingerprint);
                if fp == desired_fp {
                    // A resync that swapped the symlink without touching the body;
                    // the interval tick already covers periodic reconciliation.
                    ControlFlow::Continue(())
                } else {
                    desired_fp = fp;
                    tracing::info!("desired-set file changed; reconciling now");
                    reconcile_pass(&mut reconciler, desired_set.as_ref(), base_desired.as_ref(), &fetcher, &advertised, cli.auto_expand).await;
                    ControlFlow::Continue(())
                }
            }
            _ = cache_changed.notified(), if cli.auto_expand => {
                // A co-located process touched the cache. Debounce the event
                // burst, then observe twice across the settle window so a model
                // still downloading isn't advertised until its fingerprint holds.
                tokio::time::sleep(cache_settle).await;
                observe_pass(&mut reconciler, base_desired.as_ref(), &advertised).await;
                tokio::time::sleep(cache_settle).await;
                observe_pass(&mut reconciler, base_desired.as_ref(), &advertised).await;
                ControlFlow::Continue(())
            }
            _ = &mut shutdown => ControlFlow::Break(()),
        };
        if flow.is_break() {
            break;
        }
    }

    // Stop heartbeating before deregistering so the heartbeat can't re-mark a
    // source READY after we mark it STALE; the serve loop stops last.
    heartbeat_task.abort();
    reconciler.deregister(&advertised).await;
    stop.store(true, Ordering::Relaxed);
    serve_thread
        .join()
        .map_err(|_| anyhow::anyhow!("serve thread panicked"))??;
    Ok(())
}

/// One reconcile pass: snapshot the (re-read) desired set and converge toward it.
/// Shared by the interval tick and the file-change trigger so both paths behave
/// identically. A pass failure is logged, not propagated: the loop retries.
#[cfg(feature = "nixl")]
async fn reconcile_pass(
    reconciler: &mut modelexpress_client::cache::reconcile::Reconciler,
    desired_set: &dyn modelexpress_client::cache::desired::DesiredSet,
    base: &dyn modelexpress_client::cache::desired::DesiredSet,
    fetcher: &modelexpress_client::cache::reconcile::NixlFetcher,
    advertised: &std::sync::Mutex<std::collections::HashMap<String, String>>,
    auto_expand: bool,
) {
    // Pinned = the base set (always advertised, never evicted). Only needed in
    // bounded mode; skip the extra read otherwise.
    let pinned = if auto_expand {
        pinned_models(base).await
    } else {
        std::collections::HashSet::new()
    };
    let desired = desired_set.desired().await;
    if let Err(e) = reconciler
        .reconcile_once(&desired, fetcher, advertised, &pinned)
        .await
    {
        tracing::warn!(error = %e, "reconcile pass failed; retrying next interval");
    }
    // With auto-expand, also advertise models that appeared locally outside the
    // fetch path (e.g. a co-located vLLM downloaded into the shared cache), then
    // reclaim models the fleet no longer wants.
    if auto_expand {
        if let Err(e) = reconciler.observe_local(advertised, &pinned).await {
            tracing::warn!(error = %e, "local cache observe failed; retrying next interval");
        }
        reconciler.gc(&desired, &pinned, advertised).await;
    }
}

/// One capture-only pass: advertise any newly-settled local models. Used by the
/// cache-watch trigger, which needs the cheap observe without re-running the full
/// desired-set convergence the interval/file-change passes do.
#[cfg(feature = "nixl")]
async fn observe_pass(
    reconciler: &mut modelexpress_client::cache::reconcile::Reconciler,
    base: &dyn modelexpress_client::cache::desired::DesiredSet,
    advertised: &std::sync::Mutex<std::collections::HashMap<String, String>>,
) {
    let pinned = pinned_models(base).await;
    if let Err(e) = reconciler.observe_local(advertised, &pinned).await {
        tracing::warn!(error = %e, "local cache observe failed; retrying next interval");
    }
}

/// The base (pinned) models, the floor that is always advertised and never
/// evicted, regardless of use.
#[cfg(feature = "nixl")]
async fn pinned_models(
    base: &dyn modelexpress_client::cache::desired::DesiredSet,
) -> std::collections::HashSet<String> {
    base.desired()
        .await
        .into_iter()
        .map(|spec| spec.model)
        .collect()
}

#[cfg(not(feature = "nixl"))]
async fn run_serve(_cli: &Cli) -> anyhow::Result<()> {
    anyhow::bail!("the cache daemon requires the `nixl` build feature")
}

#[cfg(not(feature = "nixl"))]
async fn run_pull(_cli: &Cli) -> anyhow::Result<()> {
    anyhow::bail!("the cache daemon requires the `nixl` build feature")
}

#[cfg(not(feature = "nixl"))]
async fn run_reconcile(_cli: &Cli) -> anyhow::Result<()> {
    anyhow::bail!("the cache daemon requires the `nixl` build feature")
}
