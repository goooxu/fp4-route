#!/usr/bin/env bash
# End-to-end: from-scratch R1/R2/R3 train + WikiText PPL + throughput benches.
# Runs inside NGC PyTorch (TE + NVFP4). Prefer on GPU node:
#
#   IMG=nvcr.io/nvidia/pytorch:26.06-py3 \
#   NPROC=4 SEED=42 \
#   bash scripts/run_full_r123.sh
#
# From login:
#   REMOTE_HOST=10.x.x.x bash scripts/run_full_r123.sh --remote
#
# Env:
#   BACKUP_LEGACY=1   rename checkpoints/seed_$SEED → seed_${SEED}_legacy_swfq (default 1)
#   SKIP_TRAIN=0      set 1 to only eval+bench existing ckpts
#   SKIP_BENCH=0
#   MAX_BATCH=192     bench sweep upper bound
set -euo pipefail

IMG="${IMG:-nvcr.io/nvidia/pytorch:26.06-py3}"
NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
REMOTE_HOST="${REMOTE_HOST:-}"
NPROC="${NPROC:-4}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/main_360m.yaml}"
BACKUP_LEGACY="${BACKUP_LEGACY:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
MAX_BATCH="${MAX_BATCH:-192}"

run_on_gpu() {
  mkdir -p "$NFS_ROOT/logs" "$NFS_ROOT/results/perf"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$NFS_ROOT/logs/full_r123_seed${SEED}_${TS}.log"
  echo "[full-r123] log=$LOG img=$IMG nproc=$NPROC seed=$SEED" | tee -a "$LOG"

  # Backup legacy software-FQ tree so we do not resume wrong weights
  if [[ "$BACKUP_LEGACY" == "1" && -d "$NFS_ROOT/checkpoints/seed_${SEED}" ]]; then
    # skip if already only a fresh empty tree mid-run
    if [[ -d "$NFS_ROOT/checkpoints/seed_${SEED}/ckpt_bf16" ]] || [[ -f "$NFS_ROOT/checkpoints/seed_${SEED}/init_model/model.safetensors" ]]; then
      LEGACY="$NFS_ROOT/checkpoints/seed_${SEED}_legacy_swfq_${TS}"
      if [[ ! -d "$LEGACY" ]]; then
        echo "[full-r123] backup checkpoints/seed_${SEED} -> $LEGACY" | tee -a "$LOG"
        mv "$NFS_ROOT/checkpoints/seed_${SEED}" "$LEGACY"
      fi
    fi
  fi
  # Offline arch config (config.json only) for AutoConfig without HF hub write
  ARCH_STUB="$NFS_ROOT/checkpoints/seed_${SEED}_legacy_swfq_ARCH"
  if [[ ! -f "$ARCH_STUB/config.json" ]]; then
    mkdir -p "$ARCH_STUB"
    SRC=$(ls -d "$NFS_ROOT"/checkpoints/seed_${SEED}_legacy_swfq_*/init_model 2>/dev/null | head -1 || true)
    if [[ -n "$SRC" && -f "$SRC/config.json" ]]; then
      cp -f "$SRC/config.json" "$SRC/tokenizer.json" "$SRC/tokenizer_config.json" "$ARCH_STUB/" 2>/dev/null || \
        cp -f "$SRC/config.json" "$ARCH_STUB/"
      echo "[full-r123] arch stub from $SRC" | tee -a "$LOG"
    fi
  fi
  # Clear partial results for this seed quality dir (optional)
  rm -rf "$NFS_ROOT/results/main_360m/seed_${SEED}" 2>/dev/null || true
  mkdir -p "$NFS_ROOT/checkpoints" "$NFS_ROOT/results/main_360m" "$NFS_ROOT/results/perf"
  # Ensure docker (root) can write results/checkpoints on NFS
  chmod -R a+rwX "$NFS_ROOT/checkpoints" "$NFS_ROOT/results" "$NFS_ROOT/logs" 2>/dev/null || true

  docker run --rm --gpus all --network host \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONUNBUFFERED=1 \
    -e NPROC="$NPROC" \
    -e SEED="$SEED" \
    -e CONFIG="$CONFIG" \
    -e SKIP_TRAIN="$SKIP_TRAIN" \
    -e SKIP_BENCH="$SKIP_BENCH" \
    -e MAX_BATCH="$MAX_BATCH" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/tmp/hf_home \
    -e HF_DATASETS_CACHE=/tmp/hf_home/datasets \
    -e TRANSFORMERS_CACHE=/tmp/hf_home/transformers \
    -e HF_HUB_CACHE=/tmp/hf_home/hub \
    -e ARCH_CONFIG_DIR=/work/checkpoints/seed_${SEED}_legacy_swfq_ARCH \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -v "$NFS_ROOT:/work" \
    -w /work \
    "$IMG" \
    bash -lc '
set -euo pipefail
cd /work
echo "[container] $(date -Is) torch=$(python -c "import torch; print(torch.__version__, torch.cuda.device_count())")"
python -c "import transformer_engine as te; print(\"te\", te.__version__)"
python -c "import transformers, datasets, yaml" 2>/dev/null || \
  pip install -q --root-user-action=ignore "transformers>=4.46" datasets pyyaml tqdm accelerate safetensors

NPROC=${NPROC:-4}
SEED=${SEED:-42}
CONFIG=${CONFIG:-configs/main_360m.yaml}
MAX_BATCH=${MAX_BATCH:-192}

run_py() {
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" "$@"
  else
    python "$@"
  fi
}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "===== [1/6] prepare data ====="
  if [[ ! -d data/wikitext2 ]]; then
    python scripts/01_prepare_data.py --config "$CONFIG" --seed "$SEED" --prefetch-fineweb
  else
    echo "wikitext2 present; ensure FineWeb npy exists for seed"
    # still run prepare if npy missing
    python scripts/01_prepare_data.py --config "$CONFIG" --seed "$SEED" --prefetch-fineweb || true
  fi

  echo "===== [2/6] init model ====="
  python scripts/01b_init_model.py --config "$CONFIG" --seed "$SEED"

  echo "===== [3/6] R1/R2 train BF16 ====="
  run_py scripts/02_train_bf16.py --config "$CONFIG" --seed "$SEED"

  echo "===== [4/6] R3 train TE NVFP4 ====="
  run_py scripts/03_train_nvfp4.py --config "$CONFIG" --seed "$SEED"
else
  echo "[skip] training"
fi

echo "===== [5/6] PPL R1,R2,R3 ====="
python scripts/05_eval_ppl.py --config "$CONFIG" --seed "$SEED" --routes R1,R2,R3

if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
  echo "===== [6/6] throughput benches (trained ckpts) ====="
  # Point bench config paths via main config isolate paths by rewriting a temp yaml
  python - <<PY
import yaml
from pathlib import Path
from mxfp4_lib.util import load_cfg
cfg = load_cfg("$CONFIG", seed=int("$SEED"))
bench = yaml.safe_load(Path("configs/bench_360m.yaml").read_text())
bench["paths"] = {
    "root": str(cfg["_root"]),
    "hf_cache": cfg["paths"].get("hf_cache", "hf_cache"),
    "data_dir": cfg["paths"]["data_dir"],
    "init_model": cfg["paths"]["init_model"],
    "ckpt_bf16": cfg["paths"]["ckpt_bf16"],
    "ckpt_nvfp4": cfg["paths"]["ckpt_nvfp4"],
    "results": str(Path(cfg["_root"]) / "results" / "perf"),
}
bench["seed"] = int("$SEED")
Path("/tmp/bench_r123.yaml").write_text(yaml.safe_dump(bench, sort_keys=False))
print("bench yaml", bench["paths"])
PY
  BC=/tmp/bench_r123.yaml
  WARM=3
  MEAS=8
  # R1 train/infer 1GPU
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R1 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_train_n1_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R1 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_infer_n1_best.json || true
  # R2 infer only (BF16 weights + TE)
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R2 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R2_infer_n1_best.json || true
  # R3 train/infer
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R3 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_train_n1_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R3 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_infer_n1_best.json || true
  # 4GPU DDP train R1 + R3
  unset CUDA_VISIBLE_DEVICES
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R1 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_train_n${NPROC}_best.json || true
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R3 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_train_n${NPROC}_best.json || true
else
  echo "[skip] bench"
fi

echo "===== write full_report ====="
python scripts/write_full_report.py --seed "$SEED" --config "$CONFIG" || true
echo "===== FULL R123 DONE $(date -Is) ====="
' 2>&1 | tee -a "$LOG"

  echo "[full-r123] finished; see $LOG"
}

if [[ "${1:-}" == "--remote" ]]; then
  [[ -n "$REMOTE_HOST" ]] || { echo "Set REMOTE_HOST"; exit 2; }
  /usr/bin/ssh -o BatchMode=yes -o ServerAliveInterval=30 \
    "gemsg@${REMOTE_HOST}" \
    "IMG=$IMG NFS_ROOT=$NFS_ROOT NPROC=$NPROC SEED=$SEED CONFIG=$CONFIG BACKUP_LEGACY=$BACKUP_LEGACY SKIP_TRAIN=$SKIP_TRAIN SKIP_BENCH=$SKIP_BENCH MAX_BATCH=$MAX_BATCH bash $NFS_ROOT/scripts/run_full_r123.sh"
else
  run_on_gpu
fi
