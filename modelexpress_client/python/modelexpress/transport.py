# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC channel construction with optional transport TLS.

TLS is selected by a ``grpcs://``/``https://`` target scheme or ``use_tls=True``; a bare
``host:port`` (or ``grpc://``/``http://``) stays plaintext. When TLS is on the server cert
is verified against ``MX_TLS_CA`` if set, else gRPC's default root store.
``MX_TLS_SERVER_NAME`` overrides the authority for verification.
"""

import os
from urllib.parse import urlsplit

import grpc

ENV_TLS_CA = "MX_TLS_CA"
ENV_TLS_SERVER_NAME = "MX_TLS_SERVER_NAME"

_TLS_SCHEMES = frozenset({"grpcs", "https"})
_PLAINTEXT_SCHEMES = frozenset({"grpc", "http"})


def parse_target(address: str) -> tuple[str, bool]:
    """Parse ``address`` into ``(host:port, use_tls)``.

    Accepts a bare ``host:port`` (plaintext) or a URL with scheme ``grpcs``/``https``
    (TLS) or ``grpc``/``http`` (plaintext). Path/query are discarded since gRPC's target
    is a bare authority. Unknown schemes raise ``ValueError``.
    """
    if "://" not in address:
        return address, False
    parts = urlsplit(address)
    scheme = parts.scheme.lower()
    if scheme not in _TLS_SCHEMES and scheme not in _PLAINTEXT_SCHEMES:
        raise ValueError(
            f"unsupported scheme {scheme!r}; expected one of "
            f"{sorted(_TLS_SCHEMES | _PLAINTEXT_SCHEMES)} or a bare host:port"
        )
    host = parts.hostname or ""
    target = f"{host}:{parts.port}" if parts.port is not None else host
    return target, scheme in _TLS_SCHEMES


def create_channel(
    target: str,
    options: list[tuple[str, object]] | None = None,
    *,
    use_tls: bool = False,
) -> grpc.Channel:
    """Open a gRPC channel to ``target`` (see module docstring for TLS selection)."""
    options = options or []
    target, scheme_is_tls = parse_target(target)
    if not (use_tls or scheme_is_tls):
        return grpc.insecure_channel(target, options=options)

    ca_path = os.environ.get(ENV_TLS_CA)
    root_certificates = None
    if ca_path:
        with open(ca_path, "rb") as handle:
            root_certificates = handle.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
    server_name = os.environ.get(ENV_TLS_SERVER_NAME)
    if server_name:
        options = [*options, ("grpc.ssl_target_name_override", server_name)]
    return grpc.secure_channel(target, credentials, options=options)
