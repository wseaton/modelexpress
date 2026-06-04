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
//! - [`transfer`] - the wire protocol and the NIXL transfer legs.
//!
//! The transfer's NIXL/FFI surface is feature-gated (`nixl`) so the default
//! build and unit tests need no RDMA hardware or `libnixl`.

// Scaffolding is introduced ahead of its callers across the implementation
// phases; allow until the daemon loop wires everything together.
#![allow(dead_code)]

pub mod transfer;
