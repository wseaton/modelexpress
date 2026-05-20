# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RDMA P2P loading strategy: receive weights from an existing source via NIXL."""

from __future__ import annotations

import logging
import os
import random
import time

import torch

from ..adapter import EngineAdapter, StrategyFailed
from .base import (
    LoadContext,
    LoadStrategy,
    SourceTransferError,
    _as_load_result,
    register_tensors,
)
from .context import LoadResult
from ..nixl_transfer import is_nixl_available
from ..transfer_safety import check_transfer_allowed
from ..types import TensorDescriptor
from .. import p2p_pb2

logger = logging.getLogger("modelexpress.strategy_rdma")

MAX_SOURCE_RETRIES = 3


class RdmaStrategy(LoadStrategy):
    """Load weights via RDMA P2P transfer from an existing source.

    Overrides load() entirely since RDMA has a fundamentally different flow:
    prepare target storage -> RDMA receive -> register + publish.
    """

    name = "rdma"
    requires = (EngineAdapter.discover_tensors,)

    def rollback(self, ctx: LoadContext) -> None:
        """Clean up NIXL state from a failed RDMA target attempt."""
        if ctx.nixl_manager is not None:
            try:
                ctx.nixl_manager.shutdown()
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to shut down NIXL manager: {e}"
                )
        ctx.tensors = {}
        ctx.nixl_manager = None

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not is_nixl_available():
            return False

        # Decentralized backends (k8s-service) serve their own
        # metadata; skip the central-server precondition for them.
        # Strict `is True` check so MagicMock's auto-attribute doesn't
        # masquerade as the flag in tests.
        server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
        requires_p2p = getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True
        if not server_addr and not requires_p2p:
            logger.info(f"[Worker {ctx.global_rank}] No MX server configured, skipping RDMA")
            return False

        allowed, reason = check_transfer_allowed(ctx.model_config)
        if not allowed:
            logger.info(
                f"[Worker {ctx.global_rank}] RDMA transfer disabled: {reason}"
            )
            return False

        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Load from a READY source or raise StrategyFailed for fallback.

        Source discovery and metadata misses do not mutate the target model and
        therefore raise clean StrategyFailed errors. Once _load_as_target()
        prepares target storage, failures are treated as mutated because the
        engine may have initialized or transformed model tensors, and those
        failures are raised immediately instead of trying another source.
        """
        result = _as_load_result(result)

        if os.environ.get("MX_USE_TRANSFER_PLAN", "").strip() == "1":
            return self._load_via_transfer_plan(result, ctx)

        return self._load_via_discovery(result, ctx)

    def _load_via_transfer_plan(
        self, result: LoadResult, ctx: LoadContext,
    ) -> LoadResult:
        """Server-side planning: single RPC, parallel RDMA from all peers."""
        max_peers_str = os.environ.get("MX_MAX_PEERS", "")
        max_peers = int(max_peers_str) if max_peers_str.strip() else None

        try:
            plan = ctx.mx_client.compute_transfer_plan(
                identity=ctx.identity,
                requester_worker_rank=ctx.worker_rank,
                requester_worker_id=ctx.worker_id,
                max_peers=max_peers,
            )
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] ComputeTransferPlan failed: {e}"
            )
            raise StrategyFailed(
                f"ComputeTransferPlan failed: {e}", mutated=False,
            ) from e

        if not plan.peers:
            diag = plan.diagnostics
            note = diag.note if diag else "unknown"
            logger.info(
                f"[Worker {ctx.global_rank}] Transfer plan returned no peers: {note}"
            )
            raise StrategyFailed("No peers in transfer plan", mutated=False)

        diag = plan.diagnostics
        logger.info(
            f"[Worker {ctx.global_rank}] Transfer plan: {len(plan.peers)} peer(s), "
            f"{diag.candidates_eligible}/{diag.candidates_total} eligible"
            + (f" ({diag.note})" if diag and diag.note else "")
        )

        try:
            result = ctx.adapter.prepare_rdma_target(result)
            result = ctx.adapter.before_rdma_receive(result)
            self._receive_from_plan(result, ctx, plan)
            return ctx.adapter.after_rdma_receive(result)
        except StrategyFailed:
            raise
        except Exception as e:
            raise StrategyFailed(str(e), mutated=True) from e

    def _receive_from_plan(
        self,
        result: LoadResult,
        ctx: LoadContext,
        plan: "p2p_pb2.ComputeTransferPlanResponse",
    ) -> None:
        """Fetch NIXL metadata from all peers, then issue parallel RDMA reads."""
        from ..metadata.worker_server import fetch_tensor_manifest

        receive_start = time.perf_counter()
        register_tensors(result, ctx)

        peers_for_transfer: list[tuple[str, list[TensorDescriptor], set[str]]] = []

        for peer in plan.peers:
            assigned_names = set(peer.assigned_tensor_names)
            if not assigned_names:
                continue

            if peer.worker_grpc_endpoint:
                tensor_protos, _ = fetch_tensor_manifest(
                    endpoint=peer.worker_grpc_endpoint,
                    mx_source_id=peer.mx_source_id,
                )
                source_tensors = [
                    TensorDescriptor(
                        name=t.name, addr=t.addr, size=t.size,
                        device_id=t.device_id, dtype=t.dtype,
                    )
                    for t in tensor_protos
                ]

                ep = peer.metadata_endpoint
                host, port_str = ep.rsplit(":", 1)
                ctx.nixl_manager.fetch_remote_and_wait(
                    remote_agent_name=peer.agent_name,
                    ip=host,
                    port=int(port_str),
                )
            else:
                source_tensors = [
                    TensorDescriptor(
                        name=t.name, addr=t.addr, size=t.size,
                        device_id=t.device_id, dtype=t.dtype,
                    )
                    for t in peer.tensors
                ]
                ctx.nixl_manager._agent.add_remote_agent(peer.nixl_metadata)

            logger.info(
                f"[Worker {ctx.global_rank}] Peer {peer.agent_name}: "
                f"{len(assigned_names)} assigned tensors of "
                f"{len(source_tensors)} total"
            )
            peers_for_transfer.append(
                (peer.agent_name, source_tensors, assigned_names)
            )

        if not peers_for_transfer:
            raise SourceTransferError("No peers had assigned tensors")

        try:
            bytes_transferred, tensor_count, duration = (
                ctx.nixl_manager.receive_from_peers(
                    peers=peers_for_transfer,
                    timeout_seconds=300.0,
                )
            )
        except Exception as e:
            raise SourceTransferError(f"Multi-peer RDMA receive failed: {e}") from e

        bandwidth_gbps = (
            (bytes_transferred * 8) / (duration * 1e9) if duration > 0 else 0
        )
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] RDMA transfer complete: "
            f"{len(peers_for_transfer)} peers, {tensor_count} tensors, "
            f"{bytes_transferred / 1e9:.2f} GB, {duration:.3f}s, "
            f"{bandwidth_gbps:.1f} Gbps"
        )

        torch.cuda.synchronize()

        total_time = time.perf_counter() - receive_start
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] Total receive time: "
            f"{total_time:.2f}s"
        )

    def _load_via_discovery(
        self, result: LoadResult, ctx: LoadContext,
    ) -> LoadResult:
        """Legacy client-side discovery: ListSources + GetMetadata loop."""
        candidates = self._find_source_instances(ctx)
        if not candidates:
            logger.info(f"[Worker {ctx.global_rank}] No RDMA source available, skipping")
            raise StrategyFailed("No RDMA source available", mutated=False)

        for instance in candidates[:MAX_SOURCE_RETRIES]:
            mx_source_id = instance.mx_source_id
            worker_id = instance.worker_id

            try:
                source_worker = self._fetch_worker_metadata(
                    ctx, mx_source_id, worker_id,
                )
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to fetch metadata for worker {worker_id}: {e}. "
                    f"Trying next candidate."
                )
                continue

            if source_worker is None:
                continue

            logger.info(
                f"[Worker {ctx.global_rank}] Trying source worker {worker_id} "
                f"({len(source_worker.tensors)} tensors)"
            )

            # Do not try another source after target preparation starts. The
            # adapter may have initialized or transformed model tensors, and a
            # failed receive may have partially written weights. The chain will
            # re-initialize the model before trying the next loading strategy.
            return self._load_as_target(
                result, ctx, source_worker, mx_source_id, worker_id,
            )

        tried = min(len(candidates), MAX_SOURCE_RETRIES)
        logger.warning(
            f"[Worker {ctx.global_rank}] Tried {tried} of {len(candidates)} source workers "
            f"(max retries={MAX_SOURCE_RETRIES}), falling through"
        )
        # Only pre-target metadata/discovery misses reach here. Failures after
        # target preparation are raised from _load_as_target() as mutated=True.
        raise StrategyFailed("No RDMA source succeeded", mutated=False)

    def _find_source_instances(
        self, ctx: LoadContext,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """Return all READY source instances (shuffled for load balancing)."""
        try:
            list_resp = ctx.mx_client.list_sources(
                identity=ctx.identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            if not list_resp.instances:
                logger.debug(f"[Worker {ctx.global_rank}] No ready source instances found")
                return []

            candidates = [
                inst for inst in list_resp.instances
                if inst.worker_rank == ctx.worker_rank
            ]
            random.shuffle(candidates)
            logger.info(
                f"[Worker {ctx.global_rank}] Found {len(candidates)} ready source worker(s)"
            )
            return candidates

        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Error listing sources, falling through: {e}"
            )
            return []

    def _fetch_worker_metadata(
        self,
        ctx: LoadContext,
        mx_source_id: str,
        worker_id: str,
    ) -> p2p_pb2.WorkerMetadata | None:
        """Fetch tensor metadata for one worker."""
        fetch_start = time.perf_counter()
        metadata_resp = ctx.mx_client.get_metadata(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )
        if not metadata_resp.found:
            logger.debug(
                f"[Worker {ctx.global_rank}] Metadata not found for worker {worker_id}, skipping"
            )
            return None
        worker = metadata_resp.worker
        if not worker.tensors and not worker.worker_grpc_endpoint:
            logger.debug(
                f"[Worker {ctx.global_rank}] Worker {worker_id} has no tensors "
                f"and no P2P endpoint, skipping"
            )
            return None
        fetch_time = time.perf_counter() - fetch_start
        mode = "P2P (lightweight)" if worker.worker_grpc_endpoint else "centralized"
        tensor_count = len(worker.tensors)
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] GetMetadata ({mode}): "
            f"{fetch_time:.3f}s, {tensor_count} tensors"
        )
        return worker

    def _load_as_target(
        self,
        result: LoadResult,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
        source_worker_id: str,
    ) -> LoadResult:
        """Receive fully-processed weights via RDMA from an existing source."""
        try:
            result = ctx.adapter.prepare_rdma_target(result)
            result = ctx.adapter.before_rdma_receive(result)
            self._receive_from_peer(result, ctx, source_worker, mx_source_id)
            return ctx.adapter.after_rdma_receive(result)
        except StrategyFailed:
            raise
        except Exception as e:
            raise StrategyFailed(str(e), mutated=True) from e

    def _receive_from_peer(
        self,
        result: LoadResult,
        ctx: LoadContext,
        source_worker,
        mx_source_id: str,
    ) -> None:
        """Receive fully-processed tensors via RDMA from the detected source."""
        receive_start = time.perf_counter()
        register_tensors(result, ctx)

        is_p2p = bool(source_worker.worker_grpc_endpoint)
        remote_agent_name_override = None

        if is_p2p:
            from ..metadata.worker_server import fetch_tensor_manifest

            manifest_start = time.perf_counter()
            logger.info(
                f"[Worker {ctx.global_rank}] P2P mode: fetching tensor manifest from "
                f"{source_worker.worker_grpc_endpoint}"
            )
            tensor_protos, manifest_bytes = fetch_tensor_manifest(
                endpoint=source_worker.worker_grpc_endpoint,
                mx_source_id=mx_source_id,
            )
            manifest_time = time.perf_counter() - manifest_start
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in tensor_protos
            ]
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P tensor manifest: "
                f"{manifest_time:.3f}s ({len(source_tensors)} tensors, "
                f"{manifest_bytes} bytes)"
            )

            nixl_fetch_start = time.perf_counter()
            ep = source_worker.metadata_endpoint
            host, port_str = ep.rsplit(":", 1)
            ctx.nixl_manager.fetch_remote_and_wait(
                remote_agent_name=source_worker.agent_name,
                ip=host,
                port=int(port_str),
            )
            nixl_fetch_time = time.perf_counter() - nixl_fetch_start
            logger.info(
                f"[Worker {ctx.global_rank}] [TIMING] P2P NIXL metadata fetch: "
                f"{nixl_fetch_time:.3f}s"
            )
            remote_agent_name_override = source_worker.agent_name
        else:
            source_tensors = [
                TensorDescriptor(
                    name=t.name, addr=t.addr, size=t.size,
                    device_id=t.device_id, dtype=t.dtype,
                )
                for t in source_worker.tensors
            ]

        logger.info(
            f"[Worker {ctx.global_rank}] Receiving {len(source_tensors)} tensors from source"
            f"{' (P2P)' if is_p2p else ''}"
        )

        transfer_start = time.perf_counter()
        try:
            bytes_transferred, tensor_count, _ = ctx.nixl_manager.receive_from_source(
                source_metadata=source_worker.nixl_metadata,
                source_tensors=source_tensors,
                timeout_seconds=300.0,
                remote_agent_name=remote_agent_name_override,
            )
        except Exception as e:
            raise SourceTransferError(f"RDMA receive failed: {e}") from e
        transfer_time = time.perf_counter() - transfer_start

        bandwidth_gbps = (bytes_transferred * 8) / (transfer_time * 1e9) if transfer_time > 0 else 0
        logger.info(
            f"[Worker {ctx.global_rank}] [TIMING] RDMA transfer complete: "
            f"{tensor_count} tensors, {bytes_transferred / 1e9:.2f} GB, "
            f"{transfer_time:.3f}s, {bandwidth_gbps:.1f} Gbps"
        )

        torch.cuda.synchronize()

        total_time = time.perf_counter() - receive_start
        logger.info(f"[Worker {ctx.global_rank}] [TIMING] Total receive time: {total_time:.2f}s")
