# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata building and publishing for MxModelLoader."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import grpc
import torch

from .heartbeat import HeartbeatThread
from ..client import MxClient
from .. import p2p_pb2

if TYPE_CHECKING:
    from ..nixl_transfer import NixlTransferManager
    from .worker_server import WorkerGrpcServer

logger = logging.getLogger("modelexpress.metadata.publish")

PUBLISH_METADATA_MAX_ATTEMPTS = 3
PUBLISH_METADATA_INITIAL_BACKOFF_SECONDS = 1.0
PUBLISH_METADATA_RETRYABLE_STATUS_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
}

# Global storage for heartbeat threads and worker servers, keyed by device_id.
_heartbeat_threads: dict[int, HeartbeatThread] = {}
_worker_servers: dict[int, "WorkerGrpcServer"] = {}  # P2P mode only
_catalog_generation_lock = threading.Lock()
_catalog_generations: dict[tuple[str, int], int] = {}


def build_source_identity(
    vllm_config, model_config,
) -> p2p_pb2.SourceIdentity:
    """Build a SourceIdentity from vLLM config objects."""
    from importlib.metadata import version as pkg_version

    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    parallel = vllm_config.parallel_config
    tp_size = getattr(parallel, "tensor_parallel_size", 1)
    pp_size = getattr(parallel, "pipeline_parallel_size", 1)
    ep_size = getattr(parallel, "expert_parallel_size", 0)

    # torch.dtype.__str__ returns e.g. "torch.bfloat16"; strip the prefix
    dtype = str(model_config.dtype).replace("torch.", "")
    quantization = model_config.quantization or ""

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_config.model,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        expert_parallel_size=ep_size,
        dtype=dtype,
        quantization=quantization,
        revision=_resolve_model_revision(model_config),
    )


def _resolve_model_revision(model_config) -> str:
    """Resolve the model revision for content-addressed identity.

    Priority:
    1. MX_MODEL_REVISION env var (explicit deployer override, useful
       for local checkpoints or non-HF sources).
    2. model_config.revision (from vLLM's ModelConfig; typically the
       HuggingFace commit SHA or branch/tag that was loaded).
    3. Empty string (unknown revision; handshake relies on the other
       identity fields only, and decentralized deployments lose the
       bit-identical guarantee).
    """
    override = os.environ.get("MX_MODEL_REVISION", "")
    if override:
        return override
    revision = getattr(model_config, "revision", None)
    return revision or ""


def collect_worker_labels() -> dict[str, str]:
    """Collect external-identity labels from MX_LABEL_<key>=<value> env vars.

    Each env var named ``MX_LABEL_<key>`` becomes label ``key`` (lowercased)
    with the env var value. Used by the planner's external-data informers
    (Prometheus, topology, maintenance, etc.) to join MX workers against
    signals tagged by deployer-meaningful identity dimensions like
    ``pod``, ``namespace``, ``node``, ``rack``.

    Empty values are dropped (a fieldRef pointing at an unset field yields
    an empty string; we don't want to advertise empty labels).
    """
    labels: dict[str, str] = {}
    prefix = "MX_LABEL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix) or not value:
            continue
        labels[key[len(prefix):].lower()] = value
    return labels


def build_tensor_protos(
    tensors: dict[str, torch.Tensor],
    device_id: int,
    global_rank: int,
) -> list["p2p_pb2.TensorDescriptor"]:
    """Build per-tensor descriptor protos from registered tensors."""
    del global_rank  # unused, kept for caller-symmetry with publish_metadata_and_ready
    return [
        p2p_pb2.TensorDescriptor(
            name=name,
            addr=t.data_ptr(),
            size=t.numel() * t.element_size(),
            device_id=device_id,
            dtype=str(t.dtype),
        )
        for name, t in tensors.items()
    ]


def build_tensor_catalog_entries(
    tensors: dict[str, torch.Tensor],
) -> list["p2p_pb2.TensorCatalogEntry"]:
    """Build lightweight tensor catalog entries without GPU addresses."""
    entries: list["p2p_pb2.TensorCatalogEntry"] = []
    for name, tensor in tensors.items():
        shape = getattr(tensor, "shape", ())
        try:
            shape_values = [int(dim) for dim in shape]
        except (TypeError, ValueError):
            shape_values = []

        entries.append(
            p2p_pb2.TensorCatalogEntry(
                name=name,
                byte_len=int(tensor.numel()) * int(tensor.element_size()),
                dtype=str(tensor.dtype),
                shape=shape_values,
            )
        )
    return entries


def publish_metadata_and_ready(
    mx_client: MxClient,
    nixl_manager: "NixlTransferManager",
    tensors: dict[str, torch.Tensor],
    worker_rank: int,
    device_id: int,
    identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
) -> None:
    """Publish tensor metadata and ready flag to the ModelExpress server."""
    logger.info(
        f"[Worker {worker_rank}] Publishing {len(tensors)} tensors for model '{identity.model_name}'"
    )

    tensor_protos = build_tensor_protos(tensors, device_id, worker_rank)
    catalog_entries = build_tensor_catalog_entries(tensors)
    worker_labels = collect_worker_labels()
    if worker_labels:
        logger.info(
            "[Worker %s] Publishing labels: %s",
            worker_rank,
            ", ".join(f"{k}={v}" for k, v in sorted(worker_labels.items())),
        )

    if _is_p2p_metadata_enabled(mx_client):
        from .worker_server import WorkerGrpcServer

        host = _get_worker_host()

        grpc_base = int(os.environ.get("MX_WORKER_GRPC_PORT", "6555"))
        worker_grpc_port = grpc_base + device_id

        # P2P PublishMetadata stays lightweight: it carries endpoint
        # pointers only. The planner learns what each peer owns from its
        # AdvertiseTensorCatalog call (below), and receivers fetch the
        # actual GPU addresses from this worker's GetTensorManifest RPC at
        # transfer time. Descriptors used to be duplicated here as well;
        # now they live solely on the worker gRPC server + the catalog.
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            metadata_endpoint=f"{host}:{nixl_manager._listen_port}",
            agent_name=nixl_manager.agent_name,
            worker_grpc_endpoint="",
            labels=worker_labels,
        )
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            worker_rank=worker_rank,
        )

        grpc_server = WorkerGrpcServer(
            tensor_protos=tensor_protos,
            mx_source_id=mx_source_id,
            port=worker_grpc_port,
            metadata_endpoint=f"{host}:{nixl_manager._listen_port}",
            agent_name=nixl_manager.agent_name,
            worker_rank=worker_rank,
        )
        actual_port = grpc_server.start()
        _worker_servers[device_id] = grpc_server

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            metadata_endpoint=f"{host}:{nixl_manager._listen_port}",
            agent_name=nixl_manager.agent_name,
            worker_grpc_endpoint=f"{host}:{actual_port}",
            labels=worker_labels,
        )
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            worker_rank=worker_rank,
        )
        logger.info(
            f"[Worker {worker_rank}] Published P2P metadata to MX server "
            f"(mx_source_id={mx_source_id}, worker_grpc={host}:{actual_port})"
        )
        advertise_tensor_catalog(
            mx_client=mx_client,
            identity=identity,
            worker_id=worker_id,
            worker_rank=worker_rank,
            entries=catalog_entries,
        )
    else:
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            nixl_metadata=nixl_manager.nixl_metadata,
            tensors=tensor_protos,
            labels=worker_labels,
        )
        mx_source_id = _publish_metadata_to_server(
            mx_client=mx_client,
            identity=identity,
            worker=worker,
            worker_id=worker_id,
            worker_rank=worker_rank,
        )
        logger.info(
            f"[Worker {worker_rank}] Published metadata to MX server "
            f"(mx_source_id={mx_source_id}, worker_id={worker_id})"
        )
        advertise_tensor_catalog(
            mx_client=mx_client,
            identity=identity,
            worker_id=worker_id,
            worker_rank=worker_rank,
            entries=catalog_entries,
        )

    heartbeat = HeartbeatThread(
        mx_client=mx_client,
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        worker_rank=worker_rank,
        nixl_manager=nixl_manager,
    )
    heartbeat.start()
    _heartbeat_threads[worker_rank] = heartbeat


def _next_catalog_generation(worker_id: str, worker_rank: int) -> int:
    """Return a process-local monotonic catalog generation for this worker."""
    key = (worker_id, worker_rank)
    with _catalog_generation_lock:
        generation = _catalog_generations.get(key, 0) + 1
        _catalog_generations[key] = generation
        return generation


def advertise_tensor_catalog(
    mx_client: MxClient,
    identity: "p2p_pb2.SourceIdentity",
    worker_id: str,
    worker_rank: int,
    entries: list["p2p_pb2.TensorCatalogEntry"],
) -> None:
    """Best-effort catalog advertisement for planner-owned tensor sets."""
    generation = _next_catalog_generation(worker_id, worker_rank)
    try:
        response = mx_client.advertise_tensor_catalog(
            identity=identity,
            worker_id=worker_id,
            worker_rank=worker_rank,
            entries=entries,
            generation=generation,
        )
    except Exception as exc:
        logger.warning(
            "[Worker %s] AdvertiseTensorCatalog failed; transfer planning "
            "will fall back to PublishMetadata tensors: %s",
            worker_rank,
            exc,
        )
        return

    logger.info(
        "[Worker %s] Advertised tensor catalog generation %d "
        "(%d entries, %d bytes)",
        worker_rank,
        response.generation,
        response.entries_accepted,
        response.total_bytes,
    )


def _publish_metadata_to_server(
    mx_client: MxClient,
    identity: "p2p_pb2.SourceIdentity",
    worker: "p2p_pb2.WorkerMetadata",
    worker_id: str,
    worker_rank: int,
) -> str:
    """Publish metadata with bounded retries and exponential backoff."""
    last_error: grpc.RpcError | None = None

    for attempt in range(1, PUBLISH_METADATA_MAX_ATTEMPTS + 1):
        try:
            return mx_client.publish_metadata(identity, worker, worker_id)
        except grpc.RpcError as exc:
            if exc.code() not in PUBLISH_METADATA_RETRYABLE_STATUS_CODES:
                raise

            last_error = exc
            if attempt == PUBLISH_METADATA_MAX_ATTEMPTS:
                break

            backoff_seconds = PUBLISH_METADATA_INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"[Worker {worker_rank}] Publish metadata attempt {attempt}/"
                f"{PUBLISH_METADATA_MAX_ATTEMPTS} failed with retryable gRPC status "
                f"{exc.code().name}: {exc}. Retrying in {backoff_seconds:.1f}s"
            )
            time.sleep(backoff_seconds)

    message = (
        f"[Worker {worker_rank}] Failed to publish metadata after "
        f"{PUBLISH_METADATA_MAX_ATTEMPTS} attempts"
    )
    logger.error("%s: %s", message, last_error)
    raise RuntimeError(f"{message}: {last_error}") from last_error


def _is_p2p_metadata_enabled(mx_client) -> bool:
    """Whether to take the P2P metadata exchange path.

    Some metadata backends (e.g. ``k8s-service``) have no central
    store and REQUIRE this path regardless of the env var: they
    expose a class-level ``REQUIRES_P2P_METADATA = True`` and this
    function returns True for them unconditionally.

    For backends that DON'T force it (``MxClient`` backed by the
    central server), the ``MX_P2P_METADATA`` env var controls
    whether the source publishes lightweight pointers (and serves
    the full metadata itself) or full metadata to the server.
    """
    # Strict identity check against True so MagicMock's auto-attribute
    # (and any other non-literal truthy value) doesn't accidentally
    # force the P2P path in tests or misconfigured clients.
    if getattr(mx_client, "REQUIRES_P2P_METADATA", False) is True:
        env_value = os.environ.get("MX_P2P_METADATA", "")
        if env_value not in ("", "1"):
            logger.warning(
                "MX_P2P_METADATA=%r is ignored for backend %s which "
                "always uses the P2P metadata path",
                env_value, type(mx_client).__name__,
            )
        return True
    return os.environ.get("MX_P2P_METADATA", "0") == "1"


def _get_worker_host() -> str:
    """Get the routable hostname/IP for this worker.

    Priority: MX_WORKER_HOST env var, then pod IP via socket.
    Falls back to FQDN. Rejects localhost variants.
    """
    import socket
    explicit = os.environ.get("MX_WORKER_HOST", "")
    if explicit:
        return explicit
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    fqdn = socket.getfqdn()
    if fqdn in ("localhost", "localhost.localdomain"):
        raise RuntimeError(
            "Cannot determine routable address for P2P metadata exchange. "
            "Set MX_WORKER_HOST or configure DNS."
        )
    return fqdn
