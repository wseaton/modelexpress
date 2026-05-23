#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU e2e of a real P2P weight transfer with p2pAuth enforce. One source + one target
# vLLM worker (Qwen2.5-0.5B), central metadata. Asserts the target logs the RDMA marker.
#
# Teardown is in-cluster and disconnect-safe: worker Jobs have activeDeadlineSeconds, and a
# reaper Job deletes the namespace + cluster-scoped RBAC after TTL even if this script dies.
#
# Env: REGISTRY (required); overrides NS, TAG, MODEL, CONTEXT, TTL, DEADLINE, HF_TOKEN.
set -euo pipefail

CONTEXT="${CONTEXT:-$(kubectl config current-context)}"
TAG="${TAG:-$(cat /tmp/mx-build-tag)}"
NS="${NS:-mx-gpu-e2e-$(date +%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TTL="${TTL:-1800}"          # reaper deletes the namespace this many seconds after start
DEADLINE="${DEADLINE:-1500}" # per-worker Job activeDeadlineSeconds
REGISTRY="${REGISTRY:?set REGISTRY to your container registry, e.g. quay.io/you}"
SERVER_IMAGE="${REGISTRY}/mx-server:${TAG}"
WORKER_IMAGE="${REGISTRY}/mx-worker-vllm:${TAG}"
K="kubectl --context ${CONTEXT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

echo "==> context=${CONTEXT} ns=${NS} tag=${TAG} model=${MODEL}"
${K} create namespace "${NS}" >/dev/null
${K} label namespace "${NS}" mx-e2e=gpu --overwrite >/dev/null

# Prereqs (installed with the caller's elevated creds): CRDs + HF secret + worker SA.
${K} apply -f "${ROOT}/examples/crds.yaml" >/dev/null
${K} -n "${NS}" create secret generic hf-token-secret \
    --from-literal=HF_TOKEN="${HF_TOKEN:-placeholder}" >/dev/null
${K} -n "${NS}" create serviceaccount modelexpress >/dev/null

echo "==> deploying mx-server via helm (enforce)"
# persistence.enabled=false for this throwaway run: the default storageclass uses Retain,
# so a cache PVC would leak a dangling PV past teardown.
helm --kube-context "${CONTEXT}" install mx "${ROOT}/helm" --namespace "${NS}" \
    --set fullnameOverride=mx-server \
    --set image.repository="${REGISTRY}/mx-server" \
    --set image.tag="${TAG}" \
    --set image.pullPolicy=Always \
    --set service.port=8000 \
    --set persistence.enabled=false \
    --set 'env.MX_METADATA_BACKEND=kubernetes' \
    --set "env.MX_METADATA_NAMESPACE=${NS}" \
    --set 'env.MODEL_EXPRESS_SERVER_PORT=8000' \
    --set serviceAccount.create=true \
    --set serviceAccount.rbac.enabled=true \
    --set p2pAuth.enabled=true \
    --set p2pAuth.mode=enforce \
    --set p2pAuth.audiences=modelexpress-p2p \
    --set p2pAuth.deviceResources=rdma/ib \
    `# CI server image (ci/k8s/server/Dockerfile.server) runs as root on port 8000,` \
    `# while the chart defaults target the official image (non-root, 8001).` \
    --set podSecurityContext.runAsNonRoot=false \
    --set securityContext.runAsNonRoot=false \
    --set livenessProbe.tcpSocket.port=8000 \
    --set readinessProbe.tcpSocket.port=8000 >/dev/null

echo "==> installing reaper (deletes ns + cluster RBAC after ${TTL}s, disconnect-safe)"
${K} create clusterrole "ns-reaper-${NS}" \
    --verb=get,delete --resource=namespaces,clusterroles,clusterrolebindings >/dev/null
${K} -n "${NS}" create serviceaccount reaper >/dev/null
${K} create clusterrolebinding "ns-reaper-${NS}" \
    --clusterrole="ns-reaper-${NS}" --serviceaccount="${NS}:reaper" >/dev/null
REAP_CMD="sleep ${TTL}; kubectl delete clusterrole mx-server-p2p-auth ns-reaper-${NS} --ignore-not-found; kubectl delete clusterrolebinding mx-server-p2p-auth ns-reaper-${NS} --ignore-not-found; kubectl delete namespace ${NS} --ignore-not-found"
JOB_NAME=reaper REAP_TTL=$((TTL + 600)) NS="${NS}" REAP_CMD="${REAP_CMD}" \
    envsubst < "${HERE}/reaper-job.yaml" | ${K} apply -n "${NS}" -f - >/dev/null

echo "==> waiting for server rollout"
${K} -n "${NS}" rollout status deployment/mx-server --timeout=180s

echo "==> deploying source + target workers (central mode, token)"
for spec in "mx-source source 0" "mx-target target 1"; do
    set -- ${spec}
    JOB_NAME="$1" ROLE="$2" USE_PLAN="$3" \
        WORKER_IMAGE="${WORKER_IMAGE}" MX_CI_MODEL="${MODEL}" WORKER_PORT=8000 \
        DEADLINE="${DEADLINE}" \
        envsubst < "${HERE}/gpu-worker.yaml" | ${K} apply -n "${NS}" -f - >/dev/null
done

echo "==> waiting for target to complete (or fail). Tailing markers..."
deadline=$((SECONDS + DEADLINE))
while [ $SECONDS -lt $deadline ]; do
    phase=$(${K} -n "${NS}" get pods -l role=target -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
    [ "${phase}" = "Succeeded" ] && break
    [ "${phase}" = "Failed" ] && { echo "target pod Failed"; break; }
    sleep 15
done

echo; echo "===== target RDMA transfer marker ====="
TPOD=$(${K} -n "${NS}" get pods -l role=target -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if ${K} -n "${NS}" logs "${TPOD}" --tail=-1 2>/dev/null | grep -q "RDMA transfer complete"; then
    echo "PASS: RDMA transfer complete found in target logs (transfer ran with enforce)"
else
    echo "FAIL: marker not found. Recent target log:"
    ${K} -n "${NS}" logs "${TPOD}" --tail=40 2>/dev/null || true
fi
echo; echo "===== server auth audit (should show authenticated callers, no denials) ====="
${K} -n "${NS}" logs deployment/mx-server --tail=-1 2>/dev/null | grep -i "device authorization" | tail -10 || true

echo; echo "Namespace ${NS} will be reaped in ~${TTL}s by the in-cluster reaper Job."
echo "Delete now with: ${K} delete ns ${NS}; ${K} delete clusterrole,clusterrolebinding mx-server-p2p-auth ns-reaper-${NS} --ignore-not-found"
