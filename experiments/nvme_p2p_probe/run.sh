#!/usr/bin/env bash
# Orchestrates the two-node NVMe->NVMe RDMA benchmark on coreweave-waldorf.
set -euo pipefail
CTX=coreweave-waldorf
NS=weaton-dev
DIR="$(cd "$(dirname "$0")" && pwd)"
K="kubectl --context $CTX -n $NS"

echo "== cleanup any prior run =="
$K delete pod bench-holder bench-puller --ignore-not-found >/dev/null 2>&1 || true
$K delete configmap nvme-bench-src --ignore-not-found >/dev/null 2>&1 || true

echo "== publish bench.py =="
$K create configmap nvme-bench-src --from-file=bench.py="$DIR/bench.py"

echo "== launch holder (downloads model, then serves) =="
$K apply -f "$DIR/holder.yaml"

echo "== wait for holder to be Running =="
for i in $(seq 1 60); do
  st=$($K get pod bench-holder --no-headers 2>/dev/null | awk '{print $3}')
  [ "$st" = "Running" ] && break
  [ "$st" = "Error" -o "$st" = "Failed" ] && { $K logs bench-holder; exit 1; }
  sleep 5
done
HOLDER_IP=$($K get pod bench-holder -o jsonpath='{.status.podIP}')
echo "   holder IP: $HOLDER_IP"

echo "== wait for holder to finish download + open its listener =="
for i in $(seq 1 120); do
  if $K logs bench-holder 2>/dev/null | grep -q "waiting for puller"; then
    echo "   holder ready"; break
  fi
  sleep 10
done

echo "== launch puller =="
sed "s/__HOLDER_IP__/$HOLDER_IP/" "$DIR/puller.yaml.tmpl" | $K apply -f -

echo "== stream puller until done =="
for i in $(seq 1 120); do
  st=$($K get pod bench-puller --no-headers 2>/dev/null | awk '{print $3}')
  if $K logs bench-puller 2>/dev/null | grep -qE "SUMMARY|MISMATCH|Traceback"; then break; fi
  [ "$st" = "Completed" -o "$st" = "Error" -o "$st" = "Failed" ] && break
  sleep 5
done

echo; echo "===== HOLDER LOG ====="; $K logs bench-holder 2>&1 | grep -vE "NIXL INFO|instantiated|Initialized NIXL" | tail -40
echo; echo "===== PULLER LOG ====="; $K logs bench-puller 2>&1 | grep -vE "NIXL INFO|instantiated|Initialized NIXL" | tail -40
