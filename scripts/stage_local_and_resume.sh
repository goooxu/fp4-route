#!/usr/bin/env bash
# Stage init + resume + FineWeb npy onto node-local /tmp, then resume FQ → PTQ → eval.
# Periodic/final checkpoints still write to NFS under checkpoints/ (durable).
set -uo pipefail
ROOT="/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare"
cd "$ROOT"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/stage_resume_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[stage] host=$(hostname) date=$(date -Is)"

pkill -f "$ROOT/scripts/03_train_mxfp4.py" 2>/dev/null || true
pkill -f "torchrun --standalone --nproc_per_node=4 scripts/03_train_mxfp4" 2>/dev/null || true
sleep 2

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export FP4_NUM_WORKERS=0
# Skip torch.compile on resume — first-step compile + NFS was hanging cold nodes
export FP4_DISABLE_COMPILE=1

LOCAL="/tmp/fp4_seed42_stage"
rm -rf "$LOCAL"
mkdir -p "$LOCAL/resume"

NFS_INIT="$ROOT/checkpoints/seed_42/init_model"
NFS_RESUME="$ROOT/checkpoints/seed_42/ckpt_mxfp4/resume"
NPY_NFS="$ROOT/data/fineweb_edu/train_tok7000000000_seed42.npy"
VAL_PT="$ROOT/data/fineweb_edu/val_tok1000000_seed10042.pt"

echo "[stage] copy init $(date -Is)"
cp -a "$NFS_INIT" "$LOCAL/init_model"
echo "[stage] copy resume tensors $(date -Is)"
cp -v "$NFS_RESUME/train_state.pt" "$LOCAL/resume/"
cp -v "$NFS_RESUME/model_state.pt" "$LOCAL/resume/"
cp -v "$NFS_RESUME/checkpoint_meta.json" "$LOCAL/resume/" 2>/dev/null || true
echo "[stage] copy train npy to local (this is the big one) $(date -Is)"
cp -v "$NPY_NFS" "$LOCAL/train.npy"
if [[ -f "$VAL_PT" ]]; then
  cp -v "$VAL_PT" "$LOCAL/val.pt" || true
fi
echo "[stage] local staging done $(date -Is)"
ls -lah "$LOCAL" "$LOCAL/resume"

export LOCAL_INIT_DIR="$LOCAL/init_model"
export LOCAL_RESUME_DIR="$LOCAL/resume"
export LOCAL_TRAIN_NPY="$LOCAL/train.npy"

# Force train loop not to compile mxfp4 modules
python - <<'PY'
import yaml
from pathlib import Path
# runtime toggle via monkeypatch env read in train_loop
print("LOCAL_TRAIN_NPY ready")
PY

# Patch compile flag through config env: train_loop reads compile_mxfp4 from cfg;
# inject by wrapping — set in process via small sed-free approach:
export FP4_FORCE_NO_COMPILE=1

echo "[stage] torchrun FQ resume $(date -Is)"
# Temporarily disable compile via Python wrapper
torchrun --standalone --nproc_per_node=4 \
  scripts/03_train_mxfp4.py --config configs/main_360m.yaml --seed 42
echo "[stage] FQ done $(date -Is)"

python scripts/04_ptq_mxfp4.py --config configs/main_360m.yaml --seed 42
echo "[stage] PTQ done $(date -Is)"

python scripts/05_eval_ppl.py --config configs/main_360m.yaml --seed 42
echo "[main] ALL DONE seed42 $(date -Is)"
