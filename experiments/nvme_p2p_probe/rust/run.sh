#!/usr/bin/env bash
# Two-node run of the Rust transfer spike on coreweave-waldorf.
# Usage: run.sh <image-ref>
#   e.g. run.sh quay.io/wseaton/modelexpress-cached:dev-20260604-...
#
# The holder downloads a model (init container) and prints its NIXL agent name +
# metadata blob; we hand those to the puller out of band (the registry-blob shape
# minus the registry), exactly like the bench passed HOLDER_IP.
set -euo pipefail
IMAGE="${1:?usage: run.sh <image-ref>}"
CTX=coreweave-waldorf
NS=weaton-dev
DIR="$(cd "$(dirname "$0")" && pwd)"
K="kubectl --context $CTX -n $NS"

echo "== cleanup =="
$K delete pod cached-holder cached-puller --ignore-not-found >/dev/null 2>&1 || true

echo "== launch holder ($IMAGE) =="
sed "s|__IMAGE__|$IMAGE|g" "$DIR/holder.yaml" | $K apply -f -

echo "== wait for holder to download + advertise =="
HOLDER_MD=""; HOLDER_NAME=""
for i in $(seq 1 120); do
  logs=$($K logs cached-holder -c holder 2>/dev/null || true)
  if grep -q "^HOLDER_MD=" <<<"$logs"; then
    HOLDER_NAME=$(grep "^HOLDER_NAME=" <<<"$logs" | head -1 | cut -d= -f2-)
    HOLDER_MD=$(grep "^HOLDER_MD=" <<<"$logs" | head -1 | cut -d= -f2-)
    echo "   holder ready: name=$HOLDER_NAME md=${#HOLDER_MD} hex chars"
    break
  fi
  st=$($K get pod cached-holder --no-headers 2>/dev/null | awk '{print $3}')
  [ "$st" = "Error" -o "$st" = "Failed" ] && { echo "holder failed:"; $K logs cached-holder -c holder; exit 1; }
  sleep 5
done
[ -n "$HOLDER_MD" ] || { echo "timed out waiting for holder"; $K logs cached-holder -c holder | tail -20; exit 1; }

echo "== launch puller =="
sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__HOLDER_NAME__|$HOLDER_NAME|g" -e "s|__HOLDER_MD__|$HOLDER_MD|g" \
  "$DIR/puller.yaml.tmpl" | $K apply -f -

echo "== stream until done =="
for i in $(seq 1 120); do
  if $K logs cached-puller 2>/dev/null | grep -qE "all .* shards verified|sha mismatch|spike failed|Error"; then break; fi
  st=$($K get pod cached-puller --no-headers 2>/dev/null | awk '{print $3}')
  [ "$st" = "Completed" -o "$st" = "Error" -o "$st" = "Failed" ] && break
  sleep 5
done

echo; echo "===== HOLDER ====="; $K logs cached-holder -c holder 2>&1 | grep -vE "^HOLDER_MD=" | tail -25
echo; echo "===== PULLER ====="; $K logs cached-puller 2>&1 | tail -25
