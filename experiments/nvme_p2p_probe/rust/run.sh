#!/usr/bin/env bash
# Two-node registry self-heal run on coreweave-waldorf.
# Usage: run.sh <image-ref> <registry-endpoint>
#   e.g. run.sh quay.io/wseaton/modelexpress-cached:dev-20260605-... http://mx-registry:8001
#
# Node A (serve) downloads a model into the HF cache layout, advertises it to the
# registry, and serves it. Node B (pull) discovers A through the registry (no
# hand-passed metadata) and pulls the whole snapshot over RDMA. Requires a
# reachable ModelExpress server at <registry-endpoint> (deploy one separately;
# the daemon only needs ListSources/GetMetadata/PublishMetadata/UpdateStatus).
set -euo pipefail
IMAGE="${1:?usage: run.sh <image-ref> <registry-endpoint>}"
REGISTRY="${2:?usage: run.sh <image-ref> <registry-endpoint>}"
CTX=coreweave-waldorf
NS=weaton-dev
DIR="$(cd "$(dirname "$0")" && pwd)"
K="kubectl --context $CTX -n $NS"

echo "== cleanup =="
$K delete pod cached-serve cached-pull --ignore-not-found >/dev/null 2>&1 || true

echo "== launch serve node ($IMAGE -> $REGISTRY) =="
sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__REGISTRY__|$REGISTRY|g" "$DIR/serve.yaml" | $K apply -f -

echo "== wait for serve to download + advertise =="
for i in $(seq 1 120); do
  logs=$($K logs cached-serve -c serve 2>/dev/null || true)
  grep -q "advertised" <<<"$logs" && { echo "   advertised"; break; }
  st=$($K get pod cached-serve --no-headers 2>/dev/null | awk '{print $3}')
  [ "$st" = "Error" -o "$st" = "Failed" ] && { echo "serve failed:"; $K logs cached-serve --all-containers; exit 1; }
  sleep 5
done

echo "== launch pull node =="
sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__REGISTRY__|$REGISTRY|g" "$DIR/pull.yaml.tmpl" | $K apply -f -

echo "== stream until pull completes =="
for i in $(seq 1 120); do
  if $K logs cached-pull 2>/dev/null | grep -qE "pull complete|no peer advertises|sha mismatch|error"; then break; fi
  st=$($K get pod cached-pull --no-headers 2>/dev/null | awk '{print $3}')
  [ "$st" = "Completed" -o "$st" = "Error" -o "$st" = "Failed" ] && break
  sleep 5
done

echo; echo "===== SERVE ====="; $K logs cached-serve -c serve 2>&1 | tail -25
echo; echo "===== PULL ====="; $K logs cached-pull 2>&1 | tail -25
