#!/usr/bin/env bash
# Generate qualitative samples for report (R1/R2/R3).
set -euo pipefail
ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
IMG="${IMG:-nvcr.io/nvidia/pytorch:26.06-py3}"
SEED="${SEED:-42}"
MAX_NEW="${MAX_NEW:-64}"
GEN_SEED="${GEN_SEED:-0}"
LANG="${LANG:-both}"
LOG="${LOG:-$ROOT/logs/generate_samples_seed${SEED}_${LANG}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$ROOT/logs" "$ROOT/docs"
echo "[gen] log=$LOG host=$(hostname) seed=$SEED lang=$LANG" | tee -a "$LOG"

docker run --rm -i --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/work \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e ARCH_CONFIG_DIR=/work/checkpoints/seed_${SEED}_legacy_swfq_ARCH \
  -e HF_HOME=/tmp/hf_home \
  -v "$ROOT:/work" -w /work "$IMG" \
  bash -lc "
set -euo pipefail
python -c 'import transformers' 2>/dev/null || \
  pip install -q --root-user-action=ignore transformers datasets pyyaml tqdm accelerate safetensors
python scripts/13_generate_samples.py --seed $SEED --max-new-tokens $MAX_NEW --gen-seed $GEN_SEED \
  --lang $LANG --routes R1,R2,R3
" 2>&1 | tee -a "$LOG"

echo "[gen] done; samples under docs/generation_samples_seed${SEED}*.md"
