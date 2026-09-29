#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the cluster TLS profile e2e on kind.
#
# kind stands in for OpenShift: the overlay in operator-openshift/tests/tls_kind
# installs the operator's OpenShift manifests plus the apiservers.config.openshift.io
# CRD, so the test can edit apiservers/cluster (tlsSecurityProfile, tlsAdherence)
# without touching a real cluster. The test logic lives in
# operator-openshift/tests/tls_kind.rs; this script only prepares the cluster.
#
# --overlay odh runs the same scenario against config/manifests/overlays/odh,
# installed the way a platform operator installs it
# (operator-openshift/tests/odh_kind): into a namespace the overlay does not
# name, with both images set by rewriting base/params.env, the file the
# platform resolves for that overlay, in a staged copy of the manifests.
#
# --overlay odh-xks installs config/manifests/overlays/odh-xks the same way and
# runs its own scenario. No stand-in CRDs here: kind is the target, a cluster
# with no OpenShift APIs.
#
# Prerequisites: docker (with buildx), kind, kubectl, cargo.
#
# Usage:
#   ./test_tls_kind.sh [--overlay openshift|odh|odh-xks] [--skip-build] [--delete] [-- extra cargo test args...]
#
#   --overlay     which install to test (default openshift)
#   --skip-build  reuse the mx-e2e/*:kind images already in the local docker
#   --delete      delete the kind cluster afterwards, pass or fail
#
# KIND_CLUSTER overrides the cluster name (default mx-tls-e2e, mx-tls-e2e-odh or
# mx-tls-e2e-xks by overlay: the installs share cluster-scoped names, so each
# gets its own cluster). The test runs against a kubeconfig exported from kind, never
# the current kubectl context.

set -euo pipefail
cd "$(dirname "$0")"

IMAGES=(operator server-openssl server-rustls)
OVERLAY=openshift
SKIP_BUILD=false
DELETE=false
TEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --overlay) OVERLAY="${2:?--overlay needs a value}"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --delete) DELETE=true; shift ;;
        --) shift; TEST_ARGS=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${OVERLAY}" in
    openshift) CLUSTER="${KIND_CLUSTER:-mx-tls-e2e}"; OPERATOR_NS=modelexpress-operator-system ;;
    odh) CLUSTER="${KIND_CLUSTER:-mx-tls-e2e-odh}"; OPERATOR_NS=mx-platform ;;
    odh-xks) CLUSTER="${KIND_CLUSTER:-mx-tls-e2e-xks}"; OPERATOR_NS=mx-platform ;;
    *) echo "unknown overlay: ${OVERLAY}" >&2; exit 2 ;;
esac

KUBECONFIG_FILE="$(mktemp)"
STAGE="$(mktemp -d)"
cleanup() {
    rm -rf "${KUBECONFIG_FILE}" "${STAGE}"
    if [ "${DELETE}" = true ]; then
        kind delete cluster --name "${CLUSTER}"
    fi
}
trap cleanup EXIT

if ! kind get clusters | grep -qx "${CLUSTER}"; then
    echo "==> creating kind cluster ${CLUSTER}"
    kind create cluster --name "${CLUSTER}" --wait 120s
fi
kind get kubeconfig --name "${CLUSTER}" > "${KUBECONFIG_FILE}"
export KUBECONFIG="${KUBECONFIG_FILE}"
# the test refuses to touch a cluster it was not pointed at
export MX_TLS_KIND_E2E=1
export MX_TLS_KIND_CONTEXT="kind-${CLUSTER}"
export MX_TLS_KIND_OPERATOR_NS="${OPERATOR_NS}"

if [ "${SKIP_BUILD}" = false ]; then
    for image in "${IMAGES[@]}"; do
        echo "==> building mx-e2e/${image}:kind"
        docker build --load -f docker/Dockerfile.e2e --target "${image}" -t "mx-e2e/${image}:kind" .
    done
fi
for image in "${IMAGES[@]}"; do
    kind load docker-image --name "${CLUSTER}" "mx-e2e/${image}:kind"
done

SCENARIO=cluster_tls_profile_end_to_end
WAIT_CRDS=(crd/apiservers.config.openshift.io crd/modelexpressservers.modelexpress.opendatahub.io)
if [ "${OVERLAY}" != openshift ]; then
    # Same relative layout as the repo, so the harness kustomization resolves.
    mkdir -p "${STAGE}/config" "${STAGE}/operator-openshift"
    cp -R config/manifests "${STAGE}/config/manifests"
    cp -R operator-openshift/tests "${STAGE}/operator-openshift/tests"
    printf 'MODELEXPRESS_OPERATOR_IMAGE=%s\nMODELEXPRESS_SERVER_IMAGE=%s\n' \
        mx-e2e/operator:kind mx-e2e/server-openssl:kind \
        > "${STAGE}/config/manifests/base/params.env"
    HARNESS="${STAGE}/operator-openshift/tests/odh_kind"
    if [ "${OVERLAY}" = odh-xks ]; then
        HARNESS="${STAGE}/operator-openshift/tests/odh_xks_kind"
        SCENARIO=platform_install_without_openshift
        WAIT_CRDS=(crd/modelexpressservers.modelexpress.opendatahub.io)
    fi
else
    HARNESS=operator-openshift/tests/tls_kind
fi

echo "==> installing the operator (${OVERLAY} overlay, namespace ${OPERATOR_NS})"
kubectl apply --server-side --force-conflicts -k "${HARNESS}"
kubectl wait --for=condition=Established --timeout=60s "${WAIT_CRDS[@]}"

echo "==> running operator-openshift/tests/tls_kind.rs: ${SCENARIO}"
cargo test -p modelexpress-operator-openshift --test tls_kind -- --ignored --exact "${SCENARIO}" --nocapture ${TEST_ARGS[@]+"${TEST_ARGS[@]}"}
