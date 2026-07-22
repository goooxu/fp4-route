#!/usr/bin/env bash
# Full pipeline. Usage:
#   bash scripts/06_run_all.sh [--config CONFIG] [--seed N] [--nproc N]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/smoke_135m.yaml}"
SEED=""
NPROC="${NPROC:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT/hf_cache/transformers}"

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed "$SEED")
fi

run_py() {
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" "$@"
  else
    python "$@"
  fi
}

echo "===== [1/6] prepare data (config=$CONFIG seed=${SEED:-cfg}) ====="
python scripts/01_prepare_data.py --config "$CONFIG" "${SEED_ARGS[@]}" --prefetch-fineweb

echo "===== [2/6] init random model ====="
python scripts/01b_init_model.py --config "$CONFIG" "${SEED_ARGS[@]}"

echo "===== [3/6] train BF16 (R1/R2) nproc=$NPROC ====="
run_py scripts/02_train_bf16.py --config "$CONFIG" "${SEED_ARGS[@]}"

echo "===== [4/6] train MXFP4 FQ (R3) nproc=$NPROC ====="
run_py scripts/03_train_mxfp4.py --config "$CONFIG" "${SEED_ARGS[@]}"

echo "===== [5/6] PTQ MXFP4 (R2) ====="
python scripts/04_ptq_mxfp4.py --config "$CONFIG" "${SEED_ARGS[@]}"

echo "===== [6/6] eval all routes ====="
python scripts/05_eval_ppl.py --config "$CONFIG" "${SEED_ARGS[@]}"

echo "===== DONE ====="
