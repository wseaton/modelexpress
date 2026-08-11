// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dependency-light shared types: env-var names and the Kubernetes CRD types
//! used by the metadata backends. Exists so external consumers (e.g. an
//! operator) can link these without pulling in the gRPC/server stack.

pub mod envs;
pub mod p2p;
pub mod registry;
