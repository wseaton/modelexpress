// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check of the cache daemon's registry integration against a real
//! server (in-memory backend) over loopback: a node advertises a FILE_CACHE
//! model and a peer discovers its NIXL blob through publish -> list -> get and
//! the `discover_blob` helper. No mocks; the server is the real `run_server`.
//!
//! Boots a server, so gated behind `integration-tests`:
//! `cargo test -p modelexpress-server --features integration-tests`.

#![allow(clippy::expect_used)]

use std::num::NonZeroU16;
use std::sync::Once;
use std::time::Duration;

use modelexpress_client::cached::registry::Registry;
use modelexpress_client::cached::{advertise, discover};
use modelexpress_common::grpc::p2p::worker_metadata::BackendMetadata;
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
    let discovered = discover::discover_blob(&mut reg, identity)
        .await
        .expect("discover");
    assert_eq!(discovered.as_deref(), Some(blob.as_slice()));

    // A model nobody holds yields no peer (caller falls back to origin).
    let none = discover::discover_blob(
        &mut reg,
        advertise::file_cache_identity("nobody/holds-this", ""),
    )
    .await
    .expect("discover none");
    assert!(none.is_none());

    stop_and_join(shutdown, handle).await;
}
