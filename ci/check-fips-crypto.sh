#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Assert a built binary links no non-FIPS crypto. RHEL's openssl is the
# validated implementation; ring holds no certificate and aws-lc-rs qualifies
# only through aws-lc-fips-sys, not the aws-lc-sys rustls pulls.
#
# Reads the symbol table of the binary itself, so it sees exactly what was
# linked regardless of how cargo resolved features. If the binary carries
# cargo-auditable data, that must not list the banned crates either. The
# Konflux Dockerfiles run it on the binary they ship.
#
# Usage: ci/check-fips-crypto.sh <binary>...

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <binary>..." >&2
    exit 2
fi

# Rust paths (ring::, rustls::, ...) and the prefixed C symbols ring and
# aws-lc build with. \b keeps string:: from matching ring::.
BANNED='\b(ring|rustls|aws_lc_rs|aws_lc_sys)::|\bring_core_[0-9]|\baws_lc_[0-9]'

failed=0
for bin in "$@"; do
    echo "==> ${bin}"
    if [[ ! -f "${bin}" ]]; then
        echo "    FAIL: no such file"
        failed=1
        continue
    fi
    symbols=$(nm --demangle "${bin}" 2>/dev/null || true)

    if [[ -z "${symbols}" ]]; then
        echo "    FAIL: no symbol table; a stripped binary cannot be checked"
        failed=1
        continue
    fi

    banned=$(grep -oE "${BANNED}" <<<"${symbols}" | sort | uniq -c || true)
    if [[ -n "${banned}" ]]; then
        echo "    FAIL: non-FIPS crypto linked (symbol count per match):"
        sed 's/^/      /' <<<"${banned}"
        failed=1
    else
        echo "    ok: no ring, rustls or aws-lc symbols"
    fi

    if grep -qE '\bopenssl::' <<<"${symbols}"; then
        echo "    ok: openssl linked"
    else
        echo "    FAIL: no openssl symbols; the openssl feature did not apply"
        failed=1
    fi

    if [[ "$(objdump -h "${bin}")" == *.dep-v0* ]]; then
        section=$(mktemp)
        copy=$(mktemp)
        objcopy --dump-section .dep-v0="${section}" "${bin}" "${copy}"
        rm -f "${copy}"
        stamped=$(python3 - "${section}" <<'EOF'
import json, sys, zlib
banned = {"ring", "rustls", "aws-lc-rs", "aws-lc-sys"}
with open(sys.argv[1], "rb") as f:
    packages = json.loads(zlib.decompress(f.read()))["packages"]
print(" ".join(sorted({p["name"] for p in packages} & banned)))
EOF
)
        rm -f "${section}"
        if [[ -n "${stamped}" ]]; then
            echo "    FAIL: cargo-auditable data lists ${stamped}; build with -Z sbom"
            failed=1
        else
            echo "    ok: cargo-auditable data lists no ring, rustls or aws-lc"
        fi
    fi
done

if [[ ${failed} -ne 0 ]]; then
    cat >&2 <<'EOF'

Non-FIPS crypto reached a shipped binary.

Build with --package <crate> --no-default-features --features openssl.
Without --package, cargo resolves features across the whole workspace and
rustls comes back through other members.

A new dependency that defaults to rustls needs default-features = false and
its native-tls/openssl backend, the way kube (openssl-tls) and reqwest
(native-tls) are wired in Cargo.toml.

The gcs feature can never be part of a FIPS build: google-cloud-auth
hardcodes rustls.
EOF
    exit 1
fi
