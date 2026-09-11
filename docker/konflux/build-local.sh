#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the Konflux images the way Konflux does: prefetch with Hermeto, then
# build with --network none. A dependency missing from the prefetch configs
# fails here instead of in the pipeline.
#
# Needs podman. Hermeto runs from its container image.
#
# Usage: ./docker/konflux/build-local.sh {server|operator|all}
# Other arch: PLATFORM=linux/arm64 ./docker/konflux/build-local.sh operator

set -euo pipefail

HERMETO_IMAGE=quay.io/konflux-ci/hermeto:0.62.0
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/hermeto-output"

case "$(uname -m)" in
    x86_64|amd64)  DEFAULT_PLATFORM=linux/amd64 ;;
    aarch64|arm64) DEFAULT_PLATFORM=linux/arm64 ;;
    *) echo "unsupported host arch $(uname -m)" >&2; exit 1 ;;
esac
PLATFORM="${PLATFORM:-${DEFAULT_PLATFORM}}"
case "${PLATFORM}" in
    */amd64)  RPM_ARCH=x86_64 ;;
    */arm64)  RPM_ARCH=aarch64 ;;
    *) echo "unsupported PLATFORM ${PLATFORM}" >&2; exit 1 ;;
esac

CLEANUP_PATHS=()
cleanup() {
    rm -rf "${CLEANUP_PATHS[@]}"
}
trap cleanup EXIT

# Hermeto points cargo at the vendored crates by writing .cargo/config.toml
# into the source tree
if [[ -e "${REPO_ROOT}/.cargo" ]]; then
    echo ".cargo/ exists; hermeto inject-files would overwrite it" >&2
    exit 1
fi
CLEANUP_PATHS+=("${REPO_ROOT}/.cargo")

# hermeto reads the origin remote; a worktree's .git points outside the repo
GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --path-format=absolute --git-common-dir)"

hermeto() {
    podman run --rm \
        --volume "${REPO_ROOT}:/source:Z" \
        --volume "${GIT_COMMON_DIR}:${GIT_COMMON_DIR}:ro" \
        --volume "${OUTPUT_ROOT}:/output:Z" \
        "${HERMETO_IMAGE}" "$@"
}

build_image() {
    local image="$1" dockerfile tag
    case "${image}" in
        server)   dockerfile=docker/Dockerfile.ubi9 ;;
        operator) dockerfile=docker/Dockerfile.operator ;;
    esac
    tag="localhost/odh-modelexpress-${image}:hermetic"
    local konflux_dir="docker/konflux/${image}"
    local output="${OUTPUT_ROOT}/${image}"

    echo "==> prefetching ${image}"
    rm -rf "${output}"
    mkdir -p "${output}"
    hermeto fetch-deps --source /source --output "/output/${image}" "[
        {\"type\": \"cargo\", \"path\": \".\"},
        {\"type\": \"rpm\", \"path\": \"${konflux_dir}\"},
        {\"type\": \"generic\", \"path\": \"${konflux_dir}\", \"lockfile\": \"generic-fetcher.yaml\"}
    ]"
    hermeto inject-files "/output/${image}" --for-output-dir /cachi2/output
    hermeto generate-env "/output/${image}" --format env \
        --for-output-dir /cachi2/output --output "/output/${image}/cachi2.env"

    local repos_dir hermetic_dockerfile sm_conf empty_secrets
    repos_dir=$(mktemp -d)
    hermetic_dockerfile=$(mktemp)
    sm_conf=$(mktemp)
    empty_secrets=$(mktemp -d)
    CLEANUP_PATHS+=("${repos_dir}" "${hermetic_dockerfile}" "${sm_conf}" "${empty_secrets}")

    cp "${output}/deps/rpm/${RPM_ARCH}/repos.d/hermeto.repo" "${repos_dir}/cachi2.repo"
    chmod -R go+rX "${repos_dir}"

    # Konflux's buildah task sources the Hermeto env in every RUN; do the same.
    sed 's|^RUN |RUN . /cachi2/cachi2.env \&\& |' "${REPO_ROOT}/${dockerfile}" > "${hermetic_dockerfile}"

    # Keep subscription-manager and host RHEL secrets from adding repos that
    # cannot resolve under --network none.
    printf '[main]\nenabled=0\n' > "${sm_conf}"

    echo "==> building ${image} (${PLATFORM}, --network none)"
    podman build \
        --file "${hermetic_dockerfile}" \
        --platform "${PLATFORM}" \
        --network none \
        --volume "${output}:/cachi2/output:Z" \
        --volume "${output}/cachi2.env:/cachi2/cachi2.env:Z" \
        --volume "${repos_dir}:/etc/yum.repos.d:Z" \
        --volume "${sm_conf}:/etc/dnf/plugins/subscription-manager.conf:Z" \
        --volume "${empty_secrets}:/run/secrets:Z" \
        --tag "${tag}" \
        "${REPO_ROOT}"

    echo "==> ${image} built as ${tag}"
}

case "${1:-}" in
    server|operator) build_image "$1" ;;
    all) build_image server; build_image operator ;;
    *) echo "usage: $0 {server|operator|all}" >&2; exit 1 ;;
esac
