#!/usr/bin/env bash
# 135M smoke on FineWeb-Edu (~200M tokens)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NPROC="${NPROC:-1}"
if [[ "$NPROC" -gt 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
bash scripts/06_run_all.sh --config configs/smoke_135m.yaml --nproc "$NPROC"
