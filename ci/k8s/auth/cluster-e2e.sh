#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU-less e2e of P2pService auth against a real cluster (selected by kubeconfig context).
# Server runs locally against the cluster apiserver via a SA-token kubeconfig (avoids the
# cluster's OIDC exec plugin). Caller pods only need to EXIST (Pending is fine) — the
# device check reads the spec, so no fabric capacity is consumed.
#
# Asserts (enforce mode): no token -> Unauthenticated; token + device-requesting pod ->
# OK; token + device-less pod -> PermissionDenied. Plaintext gRPC.
#
# Env overrides: NS, DEVICE, SELECTOR, CONTEXT, PORT.
set -euo pipefail

NS="${NS:-mx-auth-e2e}"
DEVICE="${DEVICE:-rdma/ib}"
SELECTOR="${SELECTOR:-mx-p2p=true}"
CONTEXT="${CONTEXT:-$(kubectl config current-context)}"
PORT="${PORT:-8001}"
AUDIENCE="modelexpress-p2p"
K="kubectl --context ${CONTEXT}"
TMP="$(mktemp -d)"
SEL_KEY="${SELECTOR%%=*}"; SEL_VAL="${SELECTOR#*=}"

cleanup() {
    pkill -f 'target/debug/modelexpress-server' 2>/dev/null || true
    ${K} delete ns "${NS}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
    ${K} delete clusterrole modelexpress-p2p-auth --ignore-not-found >/dev/null 2>&1 || true
    ${K} delete clusterrolebinding modelexpress-p2p-auth --ignore-not-found >/dev/null 2>&1 || true
    rm -rf "${TMP}"
}
trap cleanup EXIT

echo "==> building server"
cargo build -p modelexpress-server -q

echo "==> namespace ${NS} on ${CONTEXT}"
${K} create namespace "${NS}" >/dev/null

# Install the chart-rendered RBAC under the caller's creds, then run the server under the
# chart's SA so the chart's RBAC is itself exercised.
echo "==> prereqs: CRDs + chart-rendered RBAC for the server ServiceAccount"
${K} apply -f examples/crds.yaml >/dev/null
helm template mx ./helm --namespace "${NS}" \
    --set fullnameOverride=modelexpress \
    --set serviceAccount.create=true \
    --set serviceAccount.rbac.enabled=true \
    --set p2pAuth.enabled=true \
    --set p2pAuth.createClusterRBAC=true \
    --show-only templates/serviceaccount.yaml \
    --show-only templates/rbac.yaml \
    --show-only templates/rbac-p2p-auth.yaml \
    | ${K} apply -n "${NS}" -f - >/dev/null

echo "==> caller pods (Pending is fine; device check reads spec, not scheduling)"
${K} -n "${NS}" create serviceaccount tester >/dev/null
${K} -n "${NS}" run podwith --image=registry.k8s.io/pause:3.9 --restart=Never \
    --labels="${SEL_KEY}=${SEL_VAL}" \
    --overrides="{\"spec\":{\"serviceAccountName\":\"tester\",\"containers\":[{\"name\":\"c\",\"image\":\"registry.k8s.io/pause:3.9\",\"resources\":{\"limits\":{\"${DEVICE}\":\"1\"}}}]}}" >/dev/null
${K} -n "${NS}" run podwithout --image=registry.k8s.io/pause:3.9 --restart=Never \
    --labels="${SEL_KEY}=${SEL_VAL}" \
    --overrides='{"spec":{"serviceAccountName":"tester"}}' >/dev/null
${K} -n "${NS}" get pod podwith podwithout >/dev/null

echo "==> building a plain-token kubeconfig for the server (kube-rs-friendly)"
SRV=$(${K} config view --minify -o jsonpath='{.clusters[0].cluster.server}')
CADATA=$(${K} config view --minify --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')
SATOKEN=$(${K} -n "${NS}" create token modelexpress --duration=2h)
if [ -n "${CADATA}" ]; then CALINE="    certificate-authority-data: ${CADATA}"; else CALINE="    insecure-skip-tls-verify: true"; fi
cat > "${TMP}/kc.yaml" <<EOF
apiVersion: v1
kind: Config
current-context: c
clusters:
- name: c
  cluster:
    server: ${SRV}
${CALINE}
contexts:
- name: c
  context:
    cluster: c
    user: u
    namespace: ${NS}
users:
- name: u
  user:
    token: ${SATOKEN}
EOF

echo "==> starting server (enforce, label-selector ${SELECTOR})"
KUBECONFIG="${TMP}/kc.yaml" \
    MX_METADATA_BACKEND=kubernetes MX_METADATA_NAMESPACE="${NS}" \
    MODEL_EXPRESS_SECURITY_MODE=enforce \
    MODEL_EXPRESS_SECURITY_TOKEN_AUDIENCES="${AUDIENCE}" \
    MODEL_EXPRESS_SECURITY_DEVICE_RESOURCES="${DEVICE}" \
    MODEL_EXPRESS_SECURITY_POD_LABEL_SELECTOR="${SELECTOR}" \
    MODEL_EXPRESS_LOG_LEVEL=info \
    ./target/debug/modelexpress-server >"${TMP}/srv.log" 2>&1 &
for _ in $(seq 1 60); do grep -q "Starting gRPC server" "${TMP}/srv.log" && break; sleep 0.5; done
if ! grep -q "Starting gRPC server" "${TMP}/srv.log"; then
    echo "ERROR: server did not start; log:"; cat "${TMP}/srv.log"; exit 1
fi

TWITH=$(${K} -n "${NS}" create token tester --audience "${AUDIENCE}" --bound-object-kind Pod --bound-object-name podwith --duration 30m)
TWITHOUT=$(${K} -n "${NS}" create token tester --audience "${AUDIENCE}" --bound-object-kind Pod --bound-object-name podwithout --duration 30m)
G="grpcurl -plaintext -import-path modelexpress_common/proto -proto p2p.proto"
M=model_express.p2p.P2pService/ListSources

echo; echo "===== no token ====="
${G} -d '{}' localhost:${PORT} ${M} || true
echo; echo "===== valid token, pod HAS ${DEVICE} (expect OK) ====="
${G} -H "authorization: Bearer ${TWITH}" -d '{}' localhost:${PORT} ${M} || true
echo; echo "===== valid token, pod LACKS ${DEVICE} (expect PermissionDenied) ====="
${G} -H "authorization: Bearer ${TWITHOUT}" -d '{}' localhost:${PORT} ${M} || true
echo; echo "===== server auth audit log ====="
grep "device authorization" "${TMP}/srv.log" || true
