// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-healing NVMe model-weight cache daemon.
//!
//! A long-lived per-node agent that reconciles the local NVMe cache toward a
//! declared desired set. For each missing model it discovers a peer through the
//! P2P registry and pulls the weights over RDMA using the active-stager NIXL
//! pattern (`FILE -> DRAM -> RDMA -> DRAM -> FILE`), falling back to origin when
//! no peer holds it, then advertises its own residency so other nodes can pull
//! from it. The node is simultaneously a puller (reconciling itself) and a
//! stager (serving peers).
//!
//! Module map (built out across phases; see the design plan):
//! - [`transfer`] - the wire protocol, the on-disk cache layout, and the
//!   stager/puller halves driving a [`transfer::Transport`]. The NIXL FFI
//!   implementor is the only piece behind the `nixl` feature; the protocol is
//!   transport-generic and unit-tested with an in-process loopback.
//! - [`registry`] - thin P2P metadata gRPC client (publish / heartbeat / list /
//!   get); no NIXL, exercised against a real in-process server in CI.
//! - [`advertise`] - builds the `FILE_CACHE` identity + worker metadata a node
//!   publishes for the models it holds, plus its stable per-node worker id.
//! - [`discover`] - resolves a model identity to a peer's NIXL blob to pull.
//! - [`locator`] - maps a model name to the on-disk snapshot the node holds,
//!   so the server can serve it.
//! - [`desired`] - the declared desired set of models a node should hold.
//! - [`reconcile`] - the reconcile loop: diff desired vs local, pull the
//!   difference (peer or origin), advertise residency.

// Scaffolding is introduced ahead of its callers across the implementation
// phases; allow until the daemon loop wires everything together.
#![allow(dead_code)]

pub mod advertise;
pub mod desired;
pub mod discover;
pub mod locator;
pub mod reconcile;
pub mod registry;
pub mod transfer;
