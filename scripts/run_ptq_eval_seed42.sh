#!/usr/bin/env bash
# Seed 42: PTQ (R2) + WikiText-2 eval for R1/R2/R3. Training already complete.
set -euo pipefail
ROOT="/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare"
cd "$ROOT"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/ptq_eval_seed42_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[ptq_eval] host=$(hostname) date=$(date -Is)"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo "[ptq_eval] PTQ from BF16 ckpt..."
python scripts/04_ptq_mxfp4.py --config configs/main_360m.yaml --seed 42
echo "[ptq_eval] PTQ done $(date -Is)"

echo "[ptq_eval] WikiText-2 PPL for R1/R2/R3..."
python scripts/05_eval_ppl.py --config configs/main_360m.yaml --seed 42
echo "[ptq_eval] ALL DONE seed42 $(date -Is)"
