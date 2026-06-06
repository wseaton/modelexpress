<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CoreWeave cache-daemon

A DaemonSet that runs `modelexpress-cache --reconcile` on each selected node.
Every node converges its node-local NVMe cache toward a ConfigMap-declared model
set: it pulls each missing model from a peer over RDMA (or from origin when no
peer holds it), then advertises what it holds so other nodes pull from it. The
data path is pure InfiniBand; only the registry needs TCP.

## Prerequisites

- A reachable ModelExpress server (the P2P registry); set `registryEndpoint`.
- RDMA exposed as a device-plugin resource (`rdmaResource`, default `rdma/ib`).
- The daemon image built and pushed; set `image.tag` to a date-stamped tag
  (the floating `:dev` tag is unsafe on containerd, which caches by tag).
- Pods run out-of-mesh (`istioInject: false`): RDMA cannot traverse the istio
  sidecar.

## Configuration

`values.yaml` holds generic, cluster-agnostic defaults. Fabric- and site-specific
knobs (RDMA device-plugin resource, out-of-mesh, node sizing, the node-local NVMe
volume) live in a cluster overlay: `values-coreweave.yaml` for CoreWeave/Waldorf.
Add an overlay for another cluster rather than editing the chart.

## Install

```bash
helm install mx-cache examples/coreweave_cache_daemon \
  --namespace mx-cache --create-namespace \
  -f examples/coreweave_cache_daemon/values-coreweave.yaml \
  --set image.tag=dev-20260605-165937 \
  --set registryEndpoint=http://modelexpress-server:8001 \
  --set-json 'nodeSelector={"kubernetes.io/hostname":"gd91fda"}' \
  --set 'models={google-t5/t5-small}'
```

## Demonstrating the cascade

Starting every node cold at once makes them all fetch from origin in parallel
(no peer holds anything yet). To exercise peer-pull, warm one node first, then
widen the fleet:

1. Install pinned to a single node; wait for `model converged ... source=Origin`
   then `advertised`.
2. `helm upgrade` with a broader `nodeSelector` (or remove it); the new nodes log
   `source=Peer` as they pull from the warmed node, which then cascades.

## Editing the desired set live

The model list is a mounted ConfigMap the daemon watches (inotify on the mount
directory, which survives the kubelet's atomic symlink swap), so editing it
reconverges the fleet within ~1s with no restart. The `reconcile.intervalSeconds`
pass remains as a backstop:

```bash
helm upgrade mx-cache examples/coreweave_cache_daemon --reuse-values \
  --set 'models={google-t5/t5-small,Qwen/Qwen2.5-14B-Instruct}'
# or: kubectl -n mx-cache edit configmap mx-cache-modelexpress-cache-daemon-models
```

## Persistence

`cache.hostPath` (a node NVMe path) persists the warm cache across daemon
restarts and exposes it to co-located consumers. The default `emptyDir` is wiped
on pod restart, so the node re-pulls everything; fine for a convergence test.

## Tuning

`reconcile.bufGiB` must hold the largest shard per slot, so at `poolDepth` 2 each
slot is `bufGiB/2` GiB. `oDirect: true` plus the daemon's BLAKE3 hashing reach
the single-stream write ceiling (~2 GB/s on node-local NVMe RAID).
