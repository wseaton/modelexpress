// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `modelexpress-cached`: the self-healing NVMe model-weight cache daemon.
//!
//! The reconciliation loop is built out across the implementation phases. Today
//! this binary also exposes the Phase-0 two-node transfer spike via `--role`
//! (built only with the `nixl` feature), a Rust port of the validated benchmark
//! used to prove the FFI transport end to end on real RDMA.

use std::path::PathBuf;

use clap::Parser;

/// Self-healing NVMe model-weight cache daemon: reconciles the local cache
/// toward a desired set by pulling missing models from peers over RDMA, and
/// stages its own cache out to peers.
#[derive(Debug, Parser)]
#[command(name = "modelexpress-cached", version, about)]
struct Cli {
    /// Two-node transfer-spike role: `holder` or `puller`. Omit to run the
    /// daemon. Requires the `nixl` build.
    #[arg(long)]
    role: Option<String>,

    /// NIXL agent name for this process (spike).
    #[arg(long, default_value = "mx-cached")]
    name: String,

    /// Shard directory: the holder's source, the puller's destination (spike).
    #[arg(long)]
    dir: Option<PathBuf>,

    /// Holder NIXL listen port (spike).
    #[arg(long, env = "MODEL_EXPRESS_CACHED_NIXL_PORT", default_value_t = 7000)]
    nixl_port: u16,

    /// Puller: the holder's agent name (from the holder's `HOLDER_NAME` line).
    #[arg(long)]
    holder_name: Option<String>,

    /// Puller: the holder's NIXL metadata blob as hex (`HOLDER_MD` line).
    #[arg(long, env = "HOLDER_MD")]
    holder_md: Option<String>,

    /// Bounded staging-buffer size, in GiB. The whole DRAM cap for a transfer.
    #[arg(long, env = "MODEL_EXPRESS_CACHED_BUF_GIB", default_value_t = 4)]
    buf_gib: u32,

    /// Local model cache root (daemon). Defaults to the standard ModelExpress
    /// cache discovery (`MODEL_EXPRESS_CACHE_DIRECTORY`, config file, or `~`).
    #[arg(long, env = "MODEL_EXPRESS_CACHE_DIRECTORY")]
    cache_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.role.as_deref() {
        Some("holder") | Some("puller") => run_spike(&cli),
        Some(other) => {
            eprintln!("unknown --role {other:?} (expected holder|puller)");
            std::process::ExitCode::FAILURE
        }
        None => {
            tracing::info!(cache_dir = ?cli.cache_dir, "modelexpress-cached starting");
            tracing::error!("daemon loop not yet implemented");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "nixl")]
fn run_spike(cli: &Cli) -> std::process::ExitCode {
    use modelexpress_client::cached::transfer::spike;

    let Some(dir) = cli.dir.as_deref() else {
        eprintln!("--dir is required for a spike role");
        return std::process::ExitCode::FAILURE;
    };
    let result = match cli.role.as_deref() {
        Some("holder") => spike::run_holder(dir, &cli.name, cli.nixl_port, cli.buf_gib),
        Some("puller") => match (cli.holder_name.as_deref(), cli.holder_md.as_deref()) {
            (Some(holder_name), Some(holder_md)) => {
                spike::run_puller(dir, &cli.name, holder_name, holder_md, cli.buf_gib)
            }
            _ => {
                eprintln!("puller requires --holder-name and --holder-md (or HOLDER_MD env)");
                return std::process::ExitCode::FAILURE;
            }
        },
        _ => unreachable!("role validated by caller"),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("spike failed: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "nixl"))]
fn run_spike(_cli: &Cli) -> std::process::ExitCode {
    eprintln!("spike roles require the `nixl` build feature");
    std::process::ExitCode::FAILURE
}
