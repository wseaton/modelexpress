// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `modelexpress-cached`: the self-healing cache daemon.
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

use std::path::PathBuf;

use clap::Parser;

/// Self-healing cache daemon: reconciles the local cache toward a desired set by
/// pulling missing models from peers over RDMA, and serves its own cache out.
#[derive(Debug, Parser)]
#[command(name = "modelexpress-cached", version, about)]
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

    /// Seconds between reconcile passes in `--reconcile` mode.
    #[arg(
        long,
        env = "MODEL_EXPRESS_CACHED_RECONCILE_SECS",
        default_value_t = 60
    )]
    reconcile_secs: u64,

    /// NIXL agent name for this process.
    #[arg(long, default_value = "mx-cached")]
    name: String,

    /// NIXL listen port for the serving agent.
    #[arg(long, env = "MODEL_EXPRESS_CACHED_NIXL_PORT", default_value_t = 7000)]
    nixl_port: u16,

    /// P2P registry endpoint (the ModelExpress server).
    #[arg(
        long,
        env = "MODEL_EXPRESS_ENDPOINT",
        default_value = "http://localhost:8001"
    )]
    endpoint: String,

    /// Bounded staging-buffer size in GiB (the DRAM cap for a transfer).
    #[arg(long, env = "MODEL_EXPRESS_CACHED_BUF_GIB", default_value_t = 4)]
    buf_gib: u32,

    /// Local model cache root. Defaults to the standard ModelExpress cache
    /// discovery (`MODEL_EXPRESS_CACHE_DIRECTORY`, config file, or `~`).
    #[arg(long, env = "MODEL_EXPRESS_CACHE_DIRECTORY")]
    cache_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt::init();
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
    use modelexpress_client::cached::locator::HfLocator;
    use modelexpress_client::cached::transfer::{nixl::NixlAgent, stager::CacheServer};

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

    use modelexpress_client::cached::advertise;
    use modelexpress_client::cached::locator::locate_hf;
    use modelexpress_client::cached::registry::Registry;
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

    use modelexpress_client::cached::registry::Registry;
    use modelexpress_client::cached::transfer::{nixl::NixlAgent, puller::Puller};
    use modelexpress_client::cached::{advertise, discover};
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
    let pull_model = model.clone();
    // NixlAgent lives only on the blocking thread, never across an .await.
    let dest = tokio::task::spawn_blocking(move || -> anyhow::Result<PathBuf> {
        let mut agent = NixlAgent::new(&name, 0)?;
        let mut puller = Puller::new(&mut agent, buf_gib, false)?;
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

    use modelexpress_client::cached::advertise;
    use modelexpress_client::cached::desired::{DesiredSet, StaticDesiredSet};
    use modelexpress_client::cached::reconcile::{NixlFetcher, Reconciler};
    use modelexpress_client::cached::registry::Registry;
    use modelexpress_common::grpc::p2p::SourceStatus;

    if cli.models.is_empty() {
        anyhow::bail!("--reconcile needs at least one --model");
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

    let registry = Registry::connect(cli.endpoint.clone()).await?;
    let worker_id = advertise::worker_id();
    let endpoint = metadata_endpoint(cli.nixl_port);
    let desired = StaticDesiredSet::from_models(cli.models.clone()).desired();

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
    let fetcher = NixlFetcher {
        agent_name: cli.name.clone(),
        buf_gib: cli.buf_gib,
        endpoint: cli.endpoint.clone(),
    };

    tracing::info!(
        models = desired.len(),
        interval_secs = cli.reconcile_secs,
        "reconcile loop started; serving and converging"
    );
    let shutdown = wait_for_shutdown();
    tokio::pin!(shutdown);
    let mut tick = tokio::time::interval(Duration::from_secs(cli.reconcile_secs));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tick.tick() => {
                if let Err(e) = reconciler
                    .reconcile_once(&desired, &fetcher, &advertised)
                    .await
                {
                    tracing::warn!(error = %e, "reconcile pass failed; retrying next interval");
                }
            }
            _ = &mut shutdown => break,
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
