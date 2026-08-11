// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Environment-variable *names* shared across the workspace and external
//! consumers (operators, tooling). Typed getters live in
//! modelexpress_common::envs, which re-exports these constants.

// ── Config-loader prefix ────────────────────────────────────────────────────
/// Prefix consumed by the `config` crate's `Environment` source in
/// [`crate::config::load_layered_config`] (env vars like `MODEL_EXPRESS_*`
/// override matching config-file fields).
pub const MODEL_EXPRESS_PREFIX: &str = "MODEL_EXPRESS";

// ── ModelExpress-owned variables ────────────────────────────────────────────
/// Client server endpoint (`ClientArgs::endpoint`).
pub const MODEL_EXPRESS_ENDPOINT: &str = "MODEL_EXPRESS_ENDPOINT";
/// Client request timeout in seconds (`ClientArgs::timeout`).
pub const MODEL_EXPRESS_TIMEOUT: &str = "MODEL_EXPRESS_TIMEOUT";
/// Local model cache directory (client, server, and both providers).
pub const MODEL_EXPRESS_CACHE_DIRECTORY: &str = "MODEL_EXPRESS_CACHE_DIRECTORY";
/// Log level (client and server).
pub const MODEL_EXPRESS_LOG_LEVEL: &str = "MODEL_EXPRESS_LOG_LEVEL";
/// Log output format (client and server).
pub const MODEL_EXPRESS_LOG_FORMAT: &str = "MODEL_EXPRESS_LOG_FORMAT";
/// Maximum connection/request retries (`ClientArgs::max_retries`).
pub const MODEL_EXPRESS_MAX_RETRIES: &str = "MODEL_EXPRESS_MAX_RETRIES";
/// Delay between retries in seconds (`ClientArgs::retry_delay`).
pub const MODEL_EXPRESS_RETRY_DELAY: &str = "MODEL_EXPRESS_RETRY_DELAY";
/// Disable shared-storage mode (`ClientArgs::no_shared_storage`).
pub const MODEL_EXPRESS_NO_SHARED_STORAGE: &str = "MODEL_EXPRESS_NO_SHARED_STORAGE";
/// File-transfer chunk size in bytes (`ClientArgs::transfer_chunk_size`).
pub const MODEL_EXPRESS_TRANSFER_CHUNK_SIZE: &str = "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE";
/// gRPC server listen port (`ServerArgs::port`).
pub const MODEL_EXPRESS_SERVER_PORT: &str = "MODEL_EXPRESS_SERVER_PORT";
/// Server host/bind address (`ServerArgs::host`).
pub const MODEL_EXPRESS_SERVER_HOST: &str = "MODEL_EXPRESS_SERVER_HOST";
/// Toggle the background cache-eviction sweeper (`ServerArgs::cache_eviction_enabled`).
pub const MODEL_EXPRESS_CACHE_EVICTION_ENABLED: &str = "MODEL_EXPRESS_CACHE_EVICTION_ENABLED";
/// Server endpoint used by the cache module's default-endpoint helper.
pub const MODEL_EXPRESS_SERVER_ENDPOINT: &str = "MODEL_EXPRESS_SERVER_ENDPOINT";

// ── HuggingFace ─────────────────────────────────────────────────────────────
/// HuggingFace Hub auth token.
pub const HF_TOKEN: &str = "HF_TOKEN";
/// HuggingFace Hub cache directory.
pub const HF_HUB_CACHE: &str = "HF_HUB_CACHE";
/// Enables HuggingFace offline mode.
pub const HF_HUB_OFFLINE: &str = "HF_HUB_OFFLINE";
/// HuggingFace Hub endpoint override. Read directly by the `hf_hub` crate;
/// registered here for reference (ModelExpress only sets it in tests).
pub const HF_ENDPOINT: &str = "HF_ENDPOINT";

// ── NGC ─────────────────────────────────────────────────────────────────────
/// Base URL for the NGC artifact/download API.
pub const NGC_API_ENDPOINT: &str = "NGC_API_ENDPOINT";
/// Base URL for the NGC authentication endpoint.
pub const NGC_AUTH_ENDPOINT: &str = "NGC_AUTH_ENDPOINT";
/// NGC API key.
pub const NGC_API_KEY: &str = "NGC_API_KEY";
/// Alternate NGC CLI API key.
pub const NGC_CLI_API_KEY: &str = "NGC_CLI_API_KEY";
/// Root directory used to locate the NGC CLI config file (`~/.ngc/config`).
pub const NGC_CLI_HOME: &str = "NGC_CLI_HOME";

/// Default NGC API base URL when [`NGC_API_ENDPOINT`] is unset.
pub const DEFAULT_NGC_API_BASE: &str = "https://api.ngc.nvidia.com";
/// Default NGC auth base URL when [`NGC_AUTH_ENDPOINT`] is unset.
pub const DEFAULT_NGC_AUTHN_BASE: &str = "https://authn.nvidia.com";

// ── Redis / metadata backend (server) ───────────────────────────────────────
/// Selects the metadata backend implementation (`redis`, `kubernetes`, `memory`).
pub const MX_METADATA_BACKEND: &str = "MX_METADATA_BACKEND";
/// Full Redis connection URL for the redis metadata backend.
pub const REDIS_URL: &str = "REDIS_URL";
/// Redis host (preferred) when building the URL from host + port.
pub const MX_REDIS_HOST: &str = "MX_REDIS_HOST";
/// Redis host alias for charts predating the `MX_` prefix.
pub const REDIS_HOST: &str = "REDIS_HOST";
/// Redis port (preferred) when building the URL from host + port.
pub const MX_REDIS_PORT: &str = "MX_REDIS_PORT";
/// Redis port alias for charts predating the `MX_` prefix.
pub const REDIS_PORT: &str = "REDIS_PORT";
/// Kubernetes namespace for ModelCacheEntry CRs (overrides [`POD_NAMESPACE`]).
pub const MX_METADATA_NAMESPACE: &str = "MX_METADATA_NAMESPACE";
/// Kubernetes namespace injected via the downward API for in-cluster pods.
pub const POD_NAMESPACE: &str = "POD_NAMESPACE";
/// Kubernetes pod name injected via the downward API (used by clients).
pub const POD_NAME: &str = "POD_NAME";
/// Kubernetes pod UID injected via the downward API (used by clients).
pub const POD_UID: &str = "POD_UID";

// ── Reaper (server) ─────────────────────────────────────────────────────────
/// Interval (seconds) between reaper scans for stale/GC worker sweeps.
pub const MX_REAPER_SCAN_INTERVAL_SECS: &str = "MX_REAPER_SCAN_INTERVAL_SECS";
/// Age (seconds) after which an active worker's heartbeat is considered stale.
pub const MX_HEARTBEAT_TIMEOUT_SECS: &str = "MX_HEARTBEAT_TIMEOUT_SECS";
/// Age (seconds) after which a STALE worker is garbage-collected.
pub const MX_GC_TIMEOUT_SECS: &str = "MX_GC_TIMEOUT_SECS";

// ── Security / auth (server) ────────────────────────────────────────────────
/// ServiceAccount auth mode (`off`, `enforce`). Off by default.
pub const MODEL_EXPRESS_SECURITY_MODE: &str = "MODEL_EXPRESS_SECURITY_MODE";
/// Comma-separated SA token audiences the caller's token must carry.
pub const MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES: &str = "MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES";
/// Comma-separated allowed callers as `<namespace>:<serviceaccount>`.
pub const MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS: &str =
    "MODEL_EXPRESS_SECURITY_ALLOWED_SERVICE_ACCOUNTS";
/// TTL for the verified-token and rejection caches, in seconds.
pub const MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS: &str = "MODEL_EXPRESS_SECURITY_CACHE_TTL_SECS";

// ── Auth (client) ───────────────────────────────────────────────────────────
/// Path to the Kubernetes projected ServiceAccount token file.
pub const MX_AUTH_TOKEN_PATH: &str = "MX_AUTH_TOKEN_PATH";
/// TTL in seconds for the cached token.
pub const MX_AUTH_TOKEN_TTL_SECONDS: &str = "MX_AUTH_TOKEN_TTL_SECONDS";

// ── System ──────────────────────────────────────────────────────────────────
/// Primary source for the user's home directory.
pub const HOME: &str = "HOME";
/// Windows fallback for the home directory when [`HOME`] is unset.
pub const USERPROFILE: &str = "USERPROFILE";
/// Path to a kubeconfig file (consumed by the k8s integration tests).
pub const KUBECONFIG: &str = "KUBECONFIG";
