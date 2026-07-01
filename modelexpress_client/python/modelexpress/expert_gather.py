# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert-wise transfer planning for cross-topology RDMA weight loading.

When a source and target run the same MoE model under different
expert-parallel layouts, the experts a target rank needs are spread across
one or more source ranks and sit at different physical slots. Instead of
copying a whole expert tensor rank-to-rank, the plan addresses each expert
individually: target physical slot s_t receives global expert g from the
source slot holding g, computed from vLLM's own expert placement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalTensor:
    """A target tensor to be filled, addressed by byte offset for experts."""

    addr: int
    size: int
    device_id: int
    num_local_experts: int  # 0 for non-expert (whole-tensor) copy


@dataclass(frozen=True)
class SourceTensor:
    """A source worker's advertised tensor of the same name."""

    addr: int
    size: int
    device_id: int
    ep_rank: int  # source worker rank within the expert-parallel group


@dataclass(frozen=True)
class SubTransfer:
    """One RDMA read: remote (addr,size,device) -> local (addr,size,device)."""

    src_ep_rank: int
    remote: tuple[int, int, int]
    local: tuple[int, int, int]


@dataclass(frozen=True)
class GatherCtx:
    """Topology needed to plan an expert gather from a source worker."""

    target_ep_size: int
    target_ep_rank: int
    source_ep_rank: int
    global_num_experts: int
    placement: str


def build_expert_map(ep_size: int, ep_rank: int, global_num_experts: int,
                     placement: str) -> dict[int, int]:
    """Return {global_expert_id: local_slot} for one rank (mirrors vLLM).

    Prefers vLLM's determine_expert_map; falls back to the same linear /
    round_robin math so the planner is unit-testable without a live group.
    """
    try:
        from vllm.model_executor.layers.fused_moe.layer import determine_expert_map

        _, emap, _ = determine_expert_map(ep_size, ep_rank, global_num_experts, placement)
        if emap is None:
            return {g: g for g in range(global_num_experts)}
        return {g: int(s) for g, s in enumerate(emap.tolist()) if int(s) >= 0}
    except Exception:
        return _expert_map_fallback(ep_size, ep_rank, global_num_experts, placement)


def _expert_map_fallback(ep_size: int, ep_rank: int, global_num_experts: int,
                         placement: str) -> dict[int, int]:
    if ep_size <= 1:
        return {g: g for g in range(global_num_experts)}
    base = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local = base + 1 if ep_rank < remainder else base
    if placement == "round_robin":
        globals_here = list(range(ep_rank, global_num_experts, ep_size))
    else:  # linear
        start = ep_rank * base + min(ep_rank, remainder)
        globals_here = list(range(start, start + local))
    return {g: slot for slot, g in enumerate(globals_here)}


def plan_gather(
    name: str,
    local: LocalTensor,
    sources: list[SourceTensor],
    target_ep_size: int,
    target_ep_rank: int,
    source_ep_size: int,
    global_num_experts: int,
    placement: str,
    require_complete: bool = True,
) -> list[SubTransfer]:
    """Plan the per-expert reads that fill one local expert tensor.

    With ``require_complete`` (default), raises ValueError if a needed expert
    is offered by no source; pass False when gathering from one source of
    several, so unoffered experts are left for another source to fill.
    """
    stride = local.size // local.num_local_experts
    target_map = build_expert_map(target_ep_size, target_ep_rank, global_num_experts, placement)
    slot_for_global = {g: s for g, s in target_map.items()}

    src_by_rank = {s.ep_rank: s for s in sources}
    src_maps = {
        s.ep_rank: build_expert_map(source_ep_size, s.ep_rank, global_num_experts, placement)
        for s in sources
    }

    plan: list[SubTransfer] = []
    for g, s_t in slot_for_global.items():
        holder = next((r for r, m in src_maps.items() if g in m), None)
        if holder is None:
            if require_complete:
                raise ValueError(f"{name}: no source offers expert {g}")
            continue
        s_s = src_maps[holder][g]
        src = src_by_rank[holder]
        if src.size % stride != 0:
            raise ValueError(f"{name}: source stride mismatch on rank {holder}")
        plan.append(
            SubTransfer(
                src_ep_rank=holder,
                remote=(src.addr + s_s * stride, stride, src.device_id),
                local=(local.addr + s_t * stride, stride, local.device_id),
            )
        )
    return plan


def is_expert_tensor(name: str) -> bool:
    """True for fused expert weight/scale tensors laid out [num_experts, ...]."""
    return ".experts." in name
