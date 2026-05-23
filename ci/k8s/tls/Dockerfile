# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Debug, server-only image for ci/k8s/tls/cluster-e2e.sh.
FROM rust:1.95 AS builder
WORKDIR /app
RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --bin modelexpress-server

# trixie to match the rust builder's glibc.
FROM debian:trixie-slim AS runtime
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/debug/modelexpress-server /app/modelexpress-server
EXPOSE 8001
ENV MODEL_EXPRESS_SERVER_PORT=8001
ENV MODEL_EXPRESS_LOG_LEVEL=info
ENV MODEL_EXPRESS_CACHE_DIRECTORY=/app/cache
CMD ["/app/modelexpress-server"]
