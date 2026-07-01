# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM engine adapter."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modelexpress.engines.vllm.adapter import (
    VllmAdapter,
    _get_vllm_device_id,
    _get_vllm_worker_rank,
    build_vllm_load_context,
)
from modelexpress.metadata.publish import build_source_identity


def _vllm_config(*, tp_size=1, pp_size=1, dp_size=1, dp_rank=0, is_moe=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
        ),
        model_config=SimpleNamespace(is_moe=is_moe),
    )


def _cfg(**kw):
    c = _vllm_config(**kw)
    return c, c.model_config


def _patch_ranks(tp_rank, pp_rank):
    return (
        patch("modelexpress.engines.vllm.adapter._get_tp_rank", return_value=tp_rank),
        patch("modelexpress.engines.vllm.adapter._get_pp_rank", return_value=pp_rank),
    )


def test_shard_key_dense_excludes_dp_rank():
    config = _vllm_config(tp_size=4, pp_size=2, dp_size=8, dp_rank=5, is_moe=False)
    tp, pp = _patch_ranks(2, 1)
    with tp, pp:
        assert _get_vllm_worker_rank(config, config.model_config) == 1 * 4 + 2


def test_shard_key_dense_dp_replicas_share_key():
    tp, pp = _patch_ranks(0, 0)
    with tp, pp:
        r0 = _get_vllm_worker_rank(*_cfg(dp_size=2, dp_rank=0, is_moe=False))
        r1 = _get_vllm_worker_rank(*_cfg(dp_size=2, dp_rank=1, is_moe=False))
    assert r0 == r1 == 0


def test_shard_key_moe_includes_dp_rank():
    config = _vllm_config(tp_size=4, pp_size=2, dp_size=8, dp_rank=3, is_moe=True)
    tp, pp = _patch_ranks(2, 1)
    with tp, pp:
        assert _get_vllm_worker_rank(config, config.model_config) == 1 * (8 * 4) + 3 * 4 + 2


def test_shard_key_moe_tp1_equals_dp_rank():
    tp, pp = _patch_ranks(0, 0)
    with tp, pp:
        r0 = _get_vllm_worker_rank(*_cfg(dp_size=2, dp_rank=0, is_moe=True))
        r1 = _get_vllm_worker_rank(*_cfg(dp_size=2, dp_rank=1, is_moe=True))
    assert r0 == 0
    assert r1 == 1


def _identity_model_config(*, is_moe):
    return SimpleNamespace(
        dtype=torch.bfloat16,
        model="test-model",
        quantization=None,
        revision=None,
        is_moe=is_moe,
    )


def test_identity_dense_has_no_ep_or_placement():
    config = _vllm_config(tp_size=2, dp_size=4, is_moe=False)
    identity = build_source_identity(config, _identity_model_config(is_moe=False))
    assert identity.expert_parallel_size == 0
    assert "expert_placement_strategy" not in identity.extra_parameters
    assert "enable_eplb" not in identity.extra_parameters


def test_identity_moe_sets_ep_size_and_placement():
    config = _vllm_config(tp_size=1, dp_size=2, is_moe=True)
    config.parallel_config.expert_placement_strategy = "linear"
    config.parallel_config.enable_eplb = False
    identity = build_source_identity(config, _identity_model_config(is_moe=True))
    assert identity.expert_parallel_size == 2
    assert identity.extra_parameters["expert_placement_strategy"] == "linear"
    assert identity.extra_parameters["enable_eplb"] == "false"


def test_vllm_device_id_uses_current_platform_device(monkeypatch):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)

    assert _get_vllm_device_id(torch.device("cuda")) == 2


def test_vllm_is_cuda_alike_uses_current_platform(monkeypatch):
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            is_cuda_alike=lambda: True,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)
    adapter = VllmAdapter(_context_config(load_device="cpu"), _model_config())

    assert adapter.is_cuda_alike() is True


def test_build_vllm_load_context_uses_current_platform_for_bare_cuda(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device=None)

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda")
    assert ctx.target_device.index is None
    assert ctx.device_id == 2


def test_build_vllm_load_context_keeps_explicit_cuda_index(monkeypatch):
    _stub_vllm_current_device(monkeypatch, current_device=2)
    _stub_metadata_client(monkeypatch)
    vllm_config = _context_config(load_device="cuda:3")

    ctx = build_vllm_load_context(vllm_config, _model_config())

    assert ctx.target_device == torch.device("cuda:3")
    assert ctx.target_device.index == 3
    assert ctx.device_id == ctx.target_device.index


def _stub_vllm_current_device(monkeypatch, *, current_device: int) -> None:
    fake_platforms = SimpleNamespace(
        current_platform=SimpleNamespace(
            current_device=lambda: current_device,
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", fake_platforms)


def _stub_metadata_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelexpress.engines.vllm.adapter.create_metadata_client",
        lambda worker_rank: object(),
    )


def _context_config(*, load_device):
    return SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        load_config=SimpleNamespace(device=load_device),
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
    )


def _model_config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        model="test-model",
        quantization=None,
        revision=None,
    )
