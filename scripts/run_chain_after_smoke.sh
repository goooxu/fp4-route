#!/usr/bin/env bash
# After smoke finishes, run mainline 360M (seeds 42/43) then ensure P2 done.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

SMOKE_SUMMARY="$ROOT/results/smoke_135m/summary.md"
echo "[chain] Waiting for smoke summary at $SMOKE_SUMMARY ..."
while [[ ! -f "$SMOKE_SUMMARY" ]]; do
  sleep 60
done
echo "[chain] Smoke done. Starting main 360M..."
bash scripts/run_main_360m.sh > logs/main_360m.log 2>&1

echo "[chain] Main done. Running P2/P3 if needed..."
if [[ ! -f "$ROOT/results/qat_pretrained/qat_pretrained.json" ]]; then
  bash scripts/run_p2_p3.sh > logs/p2_p3_chain.log 2>&1
fi
echo "[chain] ALL DONE"
