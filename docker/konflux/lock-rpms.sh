#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate docker/konflux/<image>/rpms.lock.yaml from rpms.in.yaml.
#
# rpm-lockfile-prototype needs the system python's dnf bindings, so it runs in
# a UBI9 container rather than a venv.
#
# Usage: ./docker/konflux/lock-rpms.sh {server|operator|all}

set -euo pipefail

RPM_LOCKFILE_PROTOTYPE_VERSION=v0.30.1
IMAGE=registry.access.redhat.com/ubi9/ubi:9.8
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

case "${1:-}" in
    server|operator) images=("$1") ;;
    all) images=(server operator) ;;
    *) echo "usage: $0 {server|operator|all}" >&2; exit 1 ;;
esac

script='dnf install -y -q --setopt=install_weak_deps=0 python3-pip python3-dnf rpm >/dev/null
python3 -m pip install -q \
    "https://github.com/konflux-ci/rpm-lockfile-prototype/archive/refs/tags/${VERSION}.tar.gz"
for image in "$@"; do
    echo "==> ${image}"
    rpm-lockfile-prototype --outfile "docker/konflux/${image}/rpms.lock.yaml" \
        "docker/konflux/${image}/rpms.in.yaml"
done'

podman run --rm \
    --volume "${REPO_ROOT}:/work:Z" \
    --workdir /work \
    --env VERSION="${RPM_LOCKFILE_PROTOTYPE_VERSION}" \
    "${IMAGE}" bash -euo pipefail -c "${script}" _ "${images[@]}"
