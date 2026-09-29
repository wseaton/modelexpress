#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build every overlay under config/manifests, then treat the platform overlays
# (overlays/odh for OpenShift, overlays/odh-xks for any other Kubernetes) the
# way the KServe module controller does: rewrite params.env in place, force a
# namespace over the render, and check what comes out.
#
# The controller only ever rewrites the params.env the OpenShift overlay
# resolves to, including when it goes on to render the xKS overlay, so the xKS
# overlay has to pick its images up from that same file.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifests="${root}/config/manifests"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

OCP_OVERLAY=overlays/odh
PLATFORM_OVERLAYS=("${OCP_OVERLAY}" overlays/odh-xks)

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

# The controller's resolveParamsEnv: the overlay's own params.env, else
# base/params.env when the overlay sits directly under a directory named
# "overlays".
resolve_params_env() {
    local bundle="$1" overlay="$2"
    if [[ -f "${bundle}/${overlay}/params.env" ]]; then
        echo "${bundle}/${overlay}/params.env"
    elif [[ "$(basename "$(dirname "${overlay}")")" == overlays \
            && -f "${bundle}/$(dirname "$(dirname "${overlay}")")/base/params.env" ]]; then
        echo "${bundle}/$(dirname "$(dirname "${overlay}")")/base/params.env"
    else
        fail "${overlay}: no params.env the platform would find"
    fi
}

for overlay in default openshift "${PLATFORM_OVERLAYS[@]}"; do
    echo "==> kustomize build ${overlay}"
    kustomize build "${manifests}/${overlay}" > /dev/null
done

operator_image="registry.example/operator@sha256:1111"
server_image="registry.example/server@sha256:2222"
namespace="platform-applications"

cp -R "${manifests}" "${work}/bundle"
params="$(resolve_params_env "${work}/bundle" "${OCP_OVERLAY}")"
echo "==> the platform rewrites ${params#"${work}/bundle/"}"
keys="$(cut -d= -f1 "${params}" | sort | tr '\n' ' ')"
[[ "${keys}" == "MODELEXPRESS_OPERATOR_IMAGE MODELEXPRESS_SERVER_IMAGE " ]] \
    || fail "unexpected keys in ${params}: ${keys}"
printf 'MODELEXPRESS_OPERATOR_IMAGE=%s\nMODELEXPRESS_SERVER_IMAGE=%s\n' \
    "${operator_image}" "${server_image}" > "${params}"

for overlay in "${PLATFORM_OVERLAYS[@]}"; do
    echo "==> ${overlay} as the platform renders it"
    name="$(basename "${overlay}")"
    mkdir -p "${work}/platform/${name}"
    cat > "${work}/platform/${name}/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: ${namespace}
resources:
  - ../../bundle/${overlay}
EOF
    out="${work}/${name}.yaml"
    kustomize build "${work}/platform/${name}" > "${out}"

    if grep -q '^kind: Namespace$' "${out}"; then
        fail "${overlay} renders a Namespace; the platform owns it"
    fi
    grep -q "image: ${operator_image}\$" "${out}" \
        || fail "${overlay}: the operator image did not reach the Deployment"
    grep -q "value: ${server_image}\$" "${out}" \
        || fail "${overlay}: the server image did not reach the operator's env"
    if grep -n 'quay.io/opendatahub' "${out}"; then
        fail "${overlay}: a default image survived substitution"
    fi

    stray="$(grep -E '^  namespace: ' "${out}" \
        | grep -v "^  namespace: ${namespace}\$" || true)"
    [[ -z "${stray}" ]] || fail "${overlay}: objects outside ${namespace}: ${stray}"
    bindings="$(grep -c '^kind: ClusterRoleBinding$' "${out}")"
    subjects="$(grep -A2 '^- kind: ServiceAccount$' "${out}" \
        | grep -c "^  namespace: ${namespace}\$" || true)"
    [[ "${bindings}" -gt 0 && "${subjects}" -eq "${bindings}" ]] \
        || fail "${overlay}: ${subjects} of ${bindings} ClusterRoleBinding subjects were namespaced"
done

echo "==> overlays/odh-xks needs nothing OpenShift provides"
if grep -n -E 'service\.beta\.openshift\.io|openshift-service-ca|metrics-tls' "${work}/odh-xks.yaml"; then
    fail "overlays/odh-xks depends on service-ca"
fi

echo "all clear: overlays build and both platform overlays substitute cleanly"
