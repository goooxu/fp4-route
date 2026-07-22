#!/usr/bin/env bash
# Resume mainline on a (possibly new) machine after artifact sync.
# Expects FineWeb caches + code already present under project root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NPROC="${NPROC:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

CFG="${1:-configs/main_360m.yaml}"
SEED="${2:-}"

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed "$SEED")
fi

echo "[resume] config=$CFG seed=${SEED:-cfg} nproc=$NPROC"
# Skip prepare if token cache exists; still ensure tokenizer/wikitext
if [[ ! -d data/wikitext2 ]]; then
  python scripts/01_prepare_data.py --config "$CFG" "${SEED_ARGS[@]}"
else
  echo "[resume] data/wikitext2 present — skip full prepare"
fi

# Init only if missing
INIT=$(python - <<PY
import sys
sys.path.insert(0, ".")
from mxfp4_lib.util import load_cfg
cfg = load_cfg("$CFG", seed=int("$SEED") if "$SEED" else None)
print(cfg["paths"]["init_model"])
PY
)
if [[ ! -f "$INIT/config.json" ]]; then
  python scripts/01b_init_model.py --config "$CFG" "${SEED_ARGS[@]}"
else
  echo "[resume] init_model present: $INIT"
fi

run_py() {
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" "$@"
  else
    python "$@"
  fi
}

echo "[resume] BF16 (will auto-resume from checkpoints/*/resume if present)"
run_py scripts/02_train_bf16.py --config "$CFG" "${SEED_ARGS[@]}"

echo "[resume] MXFP4 FQ"
run_py scripts/03_train_mxfp4.py --config "$CFG" "${SEED_ARGS[@]}"

echo "[resume] PTQ + eval"
python scripts/04_ptq_mxfp4.py --config "$CFG" "${SEED_ARGS[@]}"
python scripts/05_eval_ppl.py --config "$CFG" "${SEED_ARGS[@]}"
echo "[resume] DONE"
