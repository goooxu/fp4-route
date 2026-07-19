#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/hf_cache/transformers}"

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

echo "===== [1/6] prepare data ====="
python scripts/01_prepare_data.py

echo "===== [2/6] init random model ====="
python scripts/01b_init_model.py

echo "===== [3/6] train BF16 (R1/R2) ====="
python scripts/02_train_bf16.py

echo "===== [4/6] train MXFP4 FQ (R3) ====="
python scripts/03_train_mxfp4.py

echo "===== [5/6] PTQ MXFP4 (R2) ====="
python scripts/04_ptq_mxfp4.py

echo "===== [6/6] eval all routes ====="
python scripts/05_eval_ppl.py

echo "===== DONE ====="
cat results/summary.md
