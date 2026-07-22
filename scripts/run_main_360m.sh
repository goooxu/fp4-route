#!/usr/bin/env bash
# 360M mainline × seeds 42 and 43, DDP by default
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NPROC="${NPROC:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
for SEED in 42 43; do
  echo "########## MAIN 360M seed=$SEED nproc=$NPROC ##########"
  bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed "$SEED" --nproc "$NPROC"
done
