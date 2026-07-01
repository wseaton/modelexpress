# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-topology expert-gather planning."""

import pytest

from modelexpress.expert_gather import (
    LocalTensor,
    SourceTensor,
    build_expert_map,
    plan_gather,
)

G = 8
STRIDE = 10


def _local(num_local_experts, addr=1000):
    return LocalTensor(
        addr=addr, size=num_local_experts * STRIDE, device_id=0,
        num_local_experts=num_local_experts,
    )


def _src(ep_rank, num_experts, addr):
    return SourceTensor(addr=addr, size=num_experts * STRIDE, device_id=0, ep_rank=ep_rank)


def test_build_map_linear_and_round_robin():
    assert build_expert_map(2, 0, G, "linear") == {0: 0, 1: 1, 2: 2, 3: 3}
    assert build_expert_map(2, 1, G, "linear") == {4: 0, 5: 1, 6: 2, 7: 3}
    assert build_expert_map(2, 0, G, "round_robin") == {0: 0, 2: 1, 4: 2, 6: 3}


def test_2a_single_full_holder_to_sharded():
    # source ep_size=1 (one rank holds all 8), target ep_size=2 rank 0
    plan = plan_gather(
        "layers.0.mlp.experts.w13_weight",
        _local(4), [_src(0, G, addr=5000)],
        target_ep_size=2, target_ep_rank=0, source_ep_size=1,
        global_num_experts=G, placement="linear",
    )
    assert len(plan) == 4
    assert all(st.src_ep_rank == 0 for st in plan)
    # target slot s gets global s from source offset s*STRIDE
    by_local = {st.local[0]: st for st in plan}
    for s in range(4):
        st = by_local[1000 + s * STRIDE]
        assert st.remote == (5000 + s * STRIDE, STRIDE, 0)


def test_2b_multi_source_gather():
    # source ep_size=4 (ranks 0..3, 2 experts each), target ep_size=2 rank 0.
    # target rank 0 needs globals [0,1,2,3]: 0,1 from src rank0; 2,3 from src rank1.
    sources = [_src(r, 2, addr=5000 + r * 100) for r in range(4)]
    plan = plan_gather(
        "layers.0.mlp.experts.w2_weight",
        _local(4), sources,
        target_ep_size=2, target_ep_rank=0, source_ep_size=4,
        global_num_experts=G, placement="linear",
    )
    assert len(plan) == 4
    holders = sorted({st.src_ep_rank for st in plan})
    assert holders == [0, 1]  # spans two source workers
    by_local = {st.local[0]: st for st in plan}
    # global 2 lives on src rank1, local slot 0 there -> offset 0 of rank1's tensor
    st = by_local[1000 + 2 * STRIDE]
    assert st.remote == (5100 + 0 * STRIDE, STRIDE, 0)


def test_missing_expert_raises():
    with pytest.raises(ValueError, match="no source offers expert"):
        plan_gather(
            "layers.0.mlp.experts.w13_weight",
            _local(4), [_src(0, 2, addr=5000)],  # source only offers 2 of 8
            target_ep_size=2, target_ep_rank=0, source_ep_size=4,
            global_num_experts=G, placement="linear",
        )
