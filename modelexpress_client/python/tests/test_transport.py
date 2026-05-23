# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for gRPC channel construction and scheme-driven TLS selection.

Plaintext is exercised against a real in-process gRPC server; TLS paths only assert the
secure channel is built (a full handshake needs a real cert).
"""

from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from modelexpress.transport import ENV_TLS_CA, create_channel, parse_target

_METHOD = "/test.Echo/Call"


class _CaptureHandler(grpc.GenericRpcHandler):
    def service(self, handler_call_details):
        def handler(request, context):
            return b""

        return grpc.unary_unary_rpc_method_handler(
            handler,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        )


def _serve(handler):
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    server.add_generic_rpc_handlers((handler,))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def _call(channel):
    channel.unary_unary(
        _METHOD,
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )(b"", timeout=5)


def test_parse_target():
    assert parse_target("host:1") == ("host:1", False)
    assert parse_target("http://host:1") == ("host:1", False)
    assert parse_target("grpc://host:1") == ("host:1", False)
    assert parse_target("https://host:1") == ("host:1", True)
    assert parse_target("grpcs://host:1") == ("host:1", True)
    # path/query are discarded; case is normalized
    assert parse_target("HTTPS://Host:1/foo?bar=baz") == ("host:1", True)
    # no port
    assert parse_target("grpcs://host") == ("host", True)
    # unknown scheme fails fast rather than silently routing to gRPC
    with pytest.raises(ValueError):
        parse_target("ftp://host:1")


def test_create_channel_plaintext_by_default():
    handler = _CaptureHandler()
    server, port = _serve(handler)
    try:
        channel = create_channel(f"127.0.0.1:{port}")
        _call(channel)
    finally:
        server.stop(0)


def test_create_channel_uses_tls_when_requested(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"-----BEGIN CERTIFICATE-----\nnot-a-real-cert\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv(ENV_TLS_CA, str(ca))
    channel = create_channel("127.0.0.1:1", use_tls=True)
    assert isinstance(channel, grpc.Channel)
    channel.close()


def test_create_channel_tls_via_scheme(monkeypatch):
    # No MX_TLS_CA: falls back to gRPC's default root store.
    monkeypatch.delenv(ENV_TLS_CA, raising=False)
    channel = create_channel("grpcs://127.0.0.1:1")
    assert isinstance(channel, grpc.Channel)
    channel.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
