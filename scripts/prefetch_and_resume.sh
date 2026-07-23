#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare"
cd "$ROOT"
LOG="$ROOT/logs/prefetch_resume_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1
echo "[start] $(hostname) $(date -Is)"

# Kill leftovers carefully (exact script paths only)
pkill -f "$ROOT/scripts/03_train_mxfp4.py" || true
pkill -f "torchrun --standalone --nproc_per_node=4 scripts/03_train_mxfp4" || true
sleep 2

NPY="$ROOT/data/fineweb_edu/train_tok7000000000_seed42.npy"
echo "[prefetch] begin $(date -Is)"
# Sequential warm of page cache; 27G at multi-GB/s should finish in seconds–minutes
dd if="$NPY" of=/dev/null bs=256M status=progress
echo "[prefetch] end $(date -Is)"
free -h | head -2

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

echo "[train] resume FQ $(date -Is)"
cat "$ROOT/checkpoints/seed_42/ckpt_mxfp4/resume/checkpoint_meta.json" || true

torchrun --standalone --nproc_per_node=4 \
  scripts/03_train_mxfp4.py --config configs/main_360m.yaml --seed 42
echo "[train] FQ done $(date -Is)"

python scripts/04_ptq_mxfp4.py --config configs/main_360m.yaml --seed 42
echo "[train] PTQ done $(date -Is)"

python scripts/05_eval_ppl.py --config configs/main_360m.yaml --seed 42
echo "[main] ALL DONE seed42 $(date -Is)"
