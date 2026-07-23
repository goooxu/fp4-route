#!/usr/bin/env bash
# Resume R1/R2 (BF16) + R3 (NVFP4) after artifact sync.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NPROC="${NPROC:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -f "$ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

CFG="${1:-configs/main_360m.yaml}"
SEED="${2:-}"

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed "$SEED")
fi

echo "[resume] config=$CFG seed=${SEED:-cfg} nproc=$NPROC"
if [[ ! -d data/wikitext2 ]]; then
  python scripts/01_prepare_data.py --config "$CFG" "${SEED_ARGS[@]}"
else
  echo "[resume] data/wikitext2 present — skip full prepare"
fi

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

echo "[resume] R1/R2 BF16"
run_py scripts/02_train_bf16.py --config "$CFG" "${SEED_ARGS[@]}"

echo "[resume] R3 TE NVFP4"
run_py scripts/03_train_nvfp4.py --config "$CFG" "${SEED_ARGS[@]}"

echo "[resume] eval R1,R2,R3"
python scripts/05_eval_ppl.py --config "$CFG" "${SEED_ARGS[@]}" --routes R1,R2,R3
echo "[resume] DONE"
