# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``_resolve_server`` address + scheme resolution."""

import pytest

from modelexpress.client import _resolve_server


@pytest.fixture(autouse=True)
def _clear_url_env(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    monkeypatch.delenv("MX_SERVER_ADDRESS", raising=False)


def test_resolve_server_default_is_plaintext_localhost():
    assert _resolve_server() == ("localhost:8001", False)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("host:9000", ("host:9000", False)),
        ("http://host:9000", ("host:9000", False)),
        ("grpc://host:9000", ("host:9000", False)),
        ("https://host:9000", ("host:9000", True)),
        ("grpcs://host:9000", ("host:9000", True)),
    ],
)
def test_resolve_server_explicit_scheme(url, expected):
    assert _resolve_server(url) == expected


def test_resolve_server_reads_model_express_url(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_URL", "grpcs://mx.example:9000")
    assert _resolve_server() == ("mx.example:9000", True)


def test_resolve_server_falls_back_to_mx_server_address(monkeypatch):
    monkeypatch.setenv("MX_SERVER_ADDRESS", "https://legacy:1234")
    assert _resolve_server() == ("legacy:1234", True)


def test_resolve_server_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_URL", "grpcs://env:1")
    assert _resolve_server("plain:2") == ("plain:2", False)
