#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# kind e2e for server transport TLS via the helm chart + cert-manager. Asserts a TLS RPC
# verified against the issued CA succeeds and a plaintext RPC to the same port is rejected.
# Kubernetes metadata backend (no Redis). KEEP=1 leaves the cluster running.
set -euo pipefail

ENGINE="${ENGINE:-docker}"
[ "${ENGINE}" = "podman" ] && export KIND_EXPERIMENTAL_PROVIDER=podman

CLUSTER="${CLUSTER:-mx-tls-e2e}"
NS="${NS:-mx-tls}"
IMG="${IMG:-modelexpress-server:tls-e2e}"
# podman tags local images localhost/<name>; the pod must reference that.
if [ "${ENGINE}" = "podman" ]; then NODE_IMG="localhost/${IMG}"; else NODE_IMG="${IMG}"; fi
CM_VERSION="${CM_VERSION:-v1.16.2}"
RELEASE=mx
FULLNAME=modelexpress
PORT=8001
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP="$(mktemp -d)"
PF_PID=""

cleanup() {
    [ -n "${PF_PID}" ] && kill "${PF_PID}" 2>/dev/null || true
    if [ "${KEEP:-0}" != "1" ]; then
        kind delete cluster --name "${CLUSTER}" >/dev/null 2>&1 || true
    fi
    rm -rf "${TMP}"
}
trap cleanup EXIT

dump_and_die() {
    echo "ERROR: $1"
    kubectl -n "${NS}" get pods,certificate,secret 2>/dev/null || true
    kubectl -n "${NS}" describe deploy/${FULLNAME} 2>/dev/null | tail -40 || true
    kubectl -n "${NS}" logs deploy/${FULLNAME} --all-containers --tail=80 2>/dev/null || true
    exit 1
}

cd "${ROOT}"

echo "==> build ${IMG} via ${ENGINE} (first build compiles the workspace)"
"${ENGINE}" build -t "${IMG}" -f ci/k8s/tls/Dockerfile .

echo "==> kind cluster ${CLUSTER}"
kind create cluster --name "${CLUSTER}" --wait 120s
kind load docker-image "${NODE_IMG}" --name "${CLUSTER}"

echo "==> cert-manager ${CM_VERSION}"
kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CM_VERSION}/cert-manager.yaml"
kubectl -n cert-manager rollout status deploy/cert-manager --timeout=180s
kubectl -n cert-manager rollout status deploy/cert-manager-webhook --timeout=180s
kubectl -n cert-manager rollout status deploy/cert-manager-cainjector --timeout=180s

echo "==> namespace + CRDs"
kubectl create namespace "${NS}"
kubectl apply -f examples/crds.yaml

echo "==> helm install (tls.enabled=true)"
helm install "${RELEASE}" ./helm -n "${NS}" \
    --set fullnameOverride="${FULLNAME}" \
    --set image.repository="${NODE_IMG%:*}" \
    --set image.tag="${NODE_IMG##*:}" \
    --set image.pullPolicy=Never \
    --set-json imagePullSecrets='[]' \
    --set serviceAccount.create=true \
    --set serviceAccount.rbac.enabled=true \
    --set tls.enabled=true \
    --set-string env.MX_METADATA_BACKEND=kubernetes \
    --set-string env.MX_METADATA_NAMESPACE="${NS}" \
    --set-string env.POD_NAMESPACE="${NS}" \
    `# the slim debug image runs as root` \
    --set podSecurityContext.runAsNonRoot=false \
    --set securityContext.runAsNonRoot=false \
    --wait --timeout 240s || true

echo "==> wait for leaf cert + deployment"
kubectl -n "${NS}" wait --for=condition=Ready certificate/${FULLNAME}-tls --timeout=120s \
    || dump_and_die "leaf certificate never became Ready"
kubectl -n "${NS}" rollout status deploy/${FULLNAME} --timeout=180s \
    || dump_and_die "server deployment never became ready"

echo "==> assert chart wired TLS (env + reloader annotation)"
RELOADER=$(kubectl -n "${NS}" get deploy ${FULLNAME} \
    -o jsonpath='{.metadata.annotations.secret\.reloader\.stakater\.com/reload}')
[ "${RELOADER}" = "${FULLNAME}-tls" ] || dump_and_die "reloader annotation missing/wrong: '${RELOADER}'"
kubectl -n "${NS}" get deploy ${FULLNAME} \
    -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}={.value}{"\n"}{end}' \
    | grep -q "MODEL_EXPRESS_TLS_CERT=/etc/modelexpress/tls/tls.crt" \
    || dump_and_die "MODEL_EXPRESS_TLS_CERT env not injected"

kubectl -n "${NS}" get secret ${FULLNAME}-tls -o jsonpath='{.data.ca\.crt}' | base64 -d > "${TMP}/ca.pem"

echo "==> port-forward svc/${FULLNAME} ${PORT}"
kubectl -n "${NS}" port-forward svc/${FULLNAME} ${PORT}:${PORT} >/dev/null 2>&1 &
PF_PID=$!
for _ in $(seq 1 40); do (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && { exec 3>&-; break; }; sleep 0.5; done

# Cert SANs are the in-cluster Service names; -authority makes verification accept that.
SNI="${FULLNAME}.${NS}.svc"
G="grpcurl -import-path modelexpress_common/proto -proto p2p.proto"
M=model_express.p2p.P2pService/ListSources

echo; echo "===== TLS RPC verifying against issued CA (expect OK) ====="
${G} -cacert "${TMP}/ca.pem" -authority "${SNI}" -d '{}' localhost:${PORT} ${M} \
    || dump_and_die "TLS ListSources failed"

echo; echo "===== plaintext RPC to the TLS port (expect rejection) ====="
if ${G} -plaintext -d '{}' localhost:${PORT} ${M} 2>"${TMP}/plain.err"; then
    dump_and_die "plaintext RPC unexpectedly succeeded against a TLS server"
fi
echo "plaintext correctly rejected: $(head -1 "${TMP}/plain.err")"

echo; echo "==> server TLS log line"
kubectl -n "${NS}" logs deploy/${FULLNAME} | grep -i "TLS" || true

echo; echo "PASS: end-to-end gRPC-over-TLS verified via helm + cert-manager"
