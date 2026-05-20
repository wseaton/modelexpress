# Multi-Peer Transfer Plan E2E Test Debug Log

## Setup

- **Cluster**: CoreWeave waldorf (`coreweave-waldorf` context, `us-east-04a`)
- **Namespace**: `mx-test`
- **Images**:
  - `quay.io/wseaton/mx-server:dev` (built from `ci/k8s/server/Dockerfile.server`)
  - `quay.io/wseaton/mx-worker-vllm:dev` (built from `examples/p2p_transfer_k8s/client/vllm/Dockerfile`, base `vllm/vllm-openai:v0.17.1`)
- **Model**: `Qwen/Qwen2.5-14B-Instruct` (ungated, ~14B, multiple safetensor shards)
- **Topology**: 2 source jobs (`mx-source-0`, `mx-source-1`) + 1 target job (`mx-target`), pod anti-affinity spreading across nodes
- **Target env**: `MX_USE_TRANSFER_PLAN=1` opts into server-side planning + parallel receive

## What worked

- MX server (`quay.io/wseaton/mx-server:dev`) deployed and running, K8s CRD backend healthy.
- Both source pods scheduled on separate GPU nodes (`gd91fda`, `g12e022`), images pulled.
- CRDs, RBAC, ServiceAccount all created without issue.

## Current blocker

Both source pods crash on startup:

```
ValueError: Load format `mx` is not supported
```

Full traceback path:
```
vllm.entrypoints.openai.api_server
  -> build_async_engine_client_from_engine_args
  -> AsyncLLM.from_vllm_config
  -> EngineCoreClient.make_async_mp_client
  -> gpu_worker.load_model
  -> gpu_model_runner.load_model
  -> get_model_loader(self.load_config)
  -> ValueError: Load format `mx` is not supported
```

The `VLLM_PLUGINS=modelexpress` env var is set and the modelexpress package is installed (`pip install .` in the Dockerfile), but vLLM's plugin system isn't registering the `mx` load format.

## Hypotheses

1. **vLLM v0.17.1 plugin API mismatch**: The base image `vllm/vllm-openai:v0.17.1` may have changed how `vllm.general_plugins` entrypoints are loaded, or the `LoadFormat` enum registration differs from what the MX client expects. The plugin entrypoint is declared in `pyproject.toml` as:
   ```toml
   [project.entry-points."vllm.general_plugins"]
   modelexpress = "modelexpress:register_modelexpress_loaders"
   ```
   This calls `modelexpress.engines.vllm.register_modelexpress_loaders()` which imports the `loader` module. If the loader module fails to import silently (e.g., missing dependency), the load format never gets registered.

2. **Silent import failure**: The `register_modelexpress_loaders` function imports `modelexpress.engines.vllm.loader`. If that import raises (e.g., nixl not available, proto stubs incompatible with installed grpcio), the plugin fails silently and `mx` never appears in vLLM's format registry.

3. **vllm v0.17.1 might not exist yet**: The version tag could be ahead of what's published on Docker Hub. Need to verify `docker.io/vllm/vllm-openai:v0.17.1` actually exists and what vLLM commit it corresponds to.

## Next steps

- [ ] Check if `vllm/vllm-openai:v0.17.1` actually exists: `podman pull --platform linux/amd64 vllm/vllm-openai:v0.17.1`
- [ ] Exec into a running source pod and run `python3 -c "import modelexpress; modelexpress.register_modelexpress_loaders()"` to see if registration fails with a traceback
- [ ] Check what vLLM version the MX CI tests use successfully (see `.github/workflows/modelexpress-ci-tests.yml` for `VLLM_VERSION` or equivalent)
- [ ] Try the vLLM version from CI or from the latest passing P2P test
- [ ] If the plugin registration works but the load format enum is missing, check if vLLM v0.17.1 changed `LoadFormat` to a different extensibility mechanism
