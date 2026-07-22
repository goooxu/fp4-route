#!/usr/bin/env bash
# P2/P3 pretrained evals + ablations (needs WikiText-2 prepared + GPU)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"

CFG="${1:-configs/main_360m.yaml}"

echo "===== pretrained baseline ====="
python scripts/07_eval_pretrained.py --config "$CFG"

echo "===== lm_head ablation ====="
python scripts/09_lm_head_ablation.py --config "$CFG"

echo "===== scale_mode ablation ====="
python scripts/10_scale_mode_ablation.py --config "$CFG"

echo "===== QAT-from-pretrained (R3') ====="
python scripts/08_qat_pretrained.py --config "$CFG"

echo "===== unit tests ====="
python tests/test_quant.py

echo "===== P2/P3 DONE ====="
