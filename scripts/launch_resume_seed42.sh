#!/usr/bin/env bash
# Resume seed42 MXFP4 FQ (+ PTQ + eval) on current GPU node. BF16 already done.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NPROC="${NPROC:-4}"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/main_seed42_resume_$(date +%Y%m%d_%H%M%S).log"

# Optional: use local raid copy of FineWeb npy if present (faster than cold NFS)
NPY_NAME="train_tok7000000000_seed42.npy"
NPY_NFS="$ROOT/data/fineweb_edu/$NPY_NAME"
NPY_LOCAL="/raid/${USER}/fp4_cache/$NPY_NAME"
if [[ -f "$NPY_LOCAL" ]]; then
  echo "[launch] using local npy symlink -> $NPY_LOCAL"
  # Point training at local file without mutating NFS original permanently:
  # replace only the working path if it is not already a symlink to local
  if [[ ! -L "$NPY_NFS" ]] || [[ "$(readlink -f "$NPY_NFS")" != "$(readlink -f "$NPY_LOCAL")" ]]; then
    if [[ -f "$NPY_NFS" && ! -L "$NPY_NFS" ]]; then
      # Keep NFS file; overlay via bind is complex — prefer env DATA_NPY_OVERRIDE later.
      # For now hardlink/symlink in place only if we can:
      true
    fi
  fi
fi

{
  echo "[main] resume FQ on $(hostname) $(date -Is) nproc=$NPROC"
  echo "[main] ckpt meta:"
  cat "$ROOT/checkpoints/seed_42/ckpt_mxfp4/resume/checkpoint_meta.json" 2>/dev/null || true

  # Prefetch npy into page cache (sequential, fast when NFS healthy)
  if [[ -f "$NPY_NFS" ]]; then
    echo "[prefetch] $NPY_NFS"
    dd if="$NPY_NFS" of=/dev/null bs=256M status=progress || true
  fi

  torchrun --standalone --nproc_per_node="$NPROC" \
    scripts/03_train_mxfp4.py --config configs/main_360m.yaml --seed 42
  echo "[main] MXFP4 FQ done $(date -Is)"

  python scripts/04_ptq_mxfp4.py --config configs/main_360m.yaml --seed 42
  echo "[main] PTQ done $(date -Is)"

  python scripts/05_eval_ppl.py --config configs/main_360m.yaml --seed 42
  echo "[main] ALL DONE seed42 $(date -Is)"
} >>"$LOG" 2>&1
