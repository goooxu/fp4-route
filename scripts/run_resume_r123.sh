#!/usr/bin/env bash
# Resume R1–R5 on GPU node from existing NFS checkpoints (safe for machine reclaim).
#
# Routes: R1 BF16 | R2 BF16→NVFP4 | R3 NVFP4 | R4 BF16→MXFP8 | R5 MXFP8
# Image default: nvcr.io/nvidia/pytorch:26.07-py3
#
# Unlike run_full_r123.sh this script:
#   - NEVER renames/moves checkpoints/seed_$SEED
#   - NEVER deletes results metrics
#   - Skips init if init_model already present
#   - Relies on train.loop resume:true + save_every for durability
#
# Usage (on GPU node or via --remote):
#   NPROC=4 SEED=42 bash scripts/run_resume_r123.sh
#   REMOTE_HOST=10.x.x.x bash scripts/run_resume_r123.sh --remote
#
# Env:
#   SKIP_TRAIN=0|1  SKIP_BENCH=0|1  MAX_BATCH=192
#   SAVE_EVERY=500  optional override written into a temp config
set -euo pipefail

IMG="${IMG:-nvcr.io/nvidia/pytorch:26.07-py3}"
NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
REMOTE_HOST="${REMOTE_HOST:-}"
NPROC="${NPROC:-4}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/main_360m.yaml}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
MAX_BATCH="${MAX_BATCH:-192}"
SAVE_EVERY="${SAVE_EVERY:-}"

run_on_gpu() {
  mkdir -p "$NFS_ROOT/logs" "$NFS_ROOT/results/perf" "$NFS_ROOT/results/main_360m"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$NFS_ROOT/logs/resume_r123_seed${SEED}_${TS}.log"
  echo "[resume-r123] log=$LOG img=$IMG nproc=$NPROC seed=$SEED host=$(hostname)" | tee -a "$LOG"
  echo "[resume-r123] ckpt tree:" | tee -a "$LOG"
  ls -la "$NFS_ROOT/checkpoints/seed_${SEED}/" 2>&1 | tee -a "$LOG" || true
  if [[ -f "$NFS_ROOT/checkpoints/seed_${SEED}/ckpt_bf16/resume/checkpoint_meta.json" ]]; then
    echo "[resume-r123] bf16 resume meta:" | tee -a "$LOG"
    cat "$NFS_ROOT/checkpoints/seed_${SEED}/ckpt_bf16/resume/checkpoint_meta.json" | tee -a "$LOG"
  else
    echo "[resume-r123] WARN: no bf16 resume checkpoint — will start BF16 from init" | tee -a "$LOG"
  fi

  # Docker (root) write access on NFS (root_squash → nobody; need world-writable)
  chmod -R a+rwX "$NFS_ROOT/checkpoints" "$NFS_ROOT/results" "$NFS_ROOT/logs" \
    "$NFS_ROOT/data" 2>/dev/null || true

  # Optional SAVE_EVERY override via temp yaml (keeps durable periodic saves)
  CONFIG_IN_CONTAINER="/work/$CONFIG"
  if [[ -n "$SAVE_EVERY" ]]; then
    python3 - <<PY
import yaml
from pathlib import Path
src = Path("$NFS_ROOT") / "$CONFIG"
cfg = yaml.safe_load(src.read_text())
cfg.setdefault("train", {})["save_every"] = int("$SAVE_EVERY")
cfg["train"]["resume"] = True
out = Path("$NFS_ROOT") / "logs" / "resume_config_seed${SEED}_${TS}.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
    # shell path for docker env
    RESUME_CFG_HOST=$(ls -1t "$NFS_ROOT/logs"/resume_config_seed${SEED}_*.yaml 2>/dev/null | head -1)
    CONFIG_IN_CONTAINER="/work/logs/$(basename "$RESUME_CFG_HOST")"
    echo "[resume-r123] using config $CONFIG_IN_CONTAINER save_every=$SAVE_EVERY" | tee -a "$LOG"
  fi

  docker pull "$IMG" 2>&1 | tail -5 | tee -a "$LOG" || true

  docker run --rm --gpus all --network host \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONUNBUFFERED=1 \
    -e NPROC="$NPROC" \
    -e SEED="$SEED" \
    -e CONFIG="$CONFIG_IN_CONTAINER" \
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
echo "[container] $(date -Is) host=$(hostname) torch=$(python -c "import torch; print(torch.__version__, torch.cuda.device_count())")"
python -c "import transformer_engine as te; print(\"te\", te.__version__)"
python -c "import transformers, datasets, yaml" 2>/dev/null || \
  pip install -q --root-user-action=ignore "transformers>=4.46" datasets pyyaml tqdm accelerate safetensors

NPROC=${NPROC:-4}
SEED=${SEED:-42}
# CONFIG may already be absolute under /work
if [[ "${CONFIG}" == /work/* ]]; then
  CFG="$CONFIG"
else
  CFG="/work/${CONFIG#/work/}"
fi
# strip leading path quirks
if [[ ! -f "$CFG" ]]; then
  CFG="/work/configs/main_360m.yaml"
fi
echo "[container] config=$CFG"
MAX_BATCH=${MAX_BATCH:-192}

run_py() {
  if [[ "$NPROC" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC" "$@"
  else
    python "$@"
  fi
}

# Mix-73 EN/ZH cache (see configs/main_360m.yaml cache_tag)
TRAIN_NPY="data/fineweb_edu/train_tok7000000000_mix73_enzh_seed${SEED}.npy"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "===== [1/7] prepare data (mix73 EN/ZH, skip if present) ====="
  if [[ ! -d data/wikitext2 ]] || [[ ! -f "$TRAIN_NPY" ]]; then
    python scripts/01_prepare_data.py --config "$CFG" --seed "$SEED" --prefetch-fineweb
  else
    echo "data present — skip prepare ($TRAIN_NPY)"
  fi

  echo "===== [2/7] init model (skip if weights present) ====="
  INIT="checkpoints/seed_${SEED}/init_model"
  # Require weight file (config-only dir is incomplete after partial archive moves)
  if [[ -f "$INIT/model.safetensors" || -f "$INIT/pytorch_model.bin" ]]; then
    echo "init_model present: $INIT — skip (preserve shared init for resume)"
  else
    echo "init weights missing under $INIT — running 01b_init_model"
    python scripts/01b_init_model.py --config "$CFG" --seed "$SEED"
  fi

  echo "===== [3/7] R1/R2/R4 train BF16 (resume if ckpt present) ====="
  if [[ -f checkpoints/seed_${SEED}/ckpt_bf16/resume/train_state.pt ]]; then
    echo "found resume: $(cat checkpoints/seed_${SEED}/ckpt_bf16/resume/checkpoint_meta.json 2>/dev/null || true)"
  fi
  run_py scripts/02_train_bf16.py --config "$CFG" --seed "$SEED"

  echo "===== [4/7] R3 train TE NVFP4 ====="
  run_py scripts/03_train_nvfp4.py --config "$CFG" --seed "$SEED" --recipe nvfp4

  echo "===== [5/7] R5 train TE MXFP8 ====="
  run_py scripts/03_train_nvfp4.py --config "$CFG" --seed "$SEED" --recipe mxfp8
else
  echo "[skip] training"
fi

echo "===== [6/7] PPL R1,R2,R3,R4,R5 ====="
python scripts/05_eval_ppl.py --config "$CFG" --seed "$SEED" --routes R1,R2,R3,R4,R5

if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
  echo "===== [7/7] throughput benches ====="
  python - <<PY
import yaml
from pathlib import Path
from mxfp4_lib.util import load_cfg
cfg = load_cfg("$CFG", seed=int("$SEED"))
bench = yaml.safe_load(Path("configs/bench_360m.yaml").read_text())
bench["paths"] = {
    "root": str(cfg["_root"]),
    "hf_cache": cfg["paths"].get("hf_cache", "hf_cache"),
    "data_dir": cfg["paths"]["data_dir"],
    "init_model": cfg["paths"]["init_model"],
    "ckpt_bf16": cfg["paths"]["ckpt_bf16"],
    "ckpt_nvfp4": cfg["paths"]["ckpt_nvfp4"],
    "ckpt_mxfp8": cfg["paths"].get("ckpt_mxfp8", ""),
    "results": str(Path(cfg["_root"]) / "results" / "perf"),
}
bench["seed"] = int("$SEED")
Path("/tmp/bench_r123.yaml").write_text(yaml.safe_dump(bench, sort_keys=False))
print("bench yaml", bench["paths"])
PY
  BC=/tmp/bench_r123.yaml
  WARM=3
  MEAS=8
  PTAG="seed${SEED}"
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R1 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_train_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R1 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_infer_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R2 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R2_infer_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R3 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_train_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R3 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_infer_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R4 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R4_infer_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R5 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R5_train_n1_${PTAG}_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R5 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R5_infer_n1_${PTAG}_best.json || true
  unset CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R1 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_train_n${NPROC}_${PTAG}_best.json || true
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R3 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_train_n${NPROC}_${PTAG}_best.json || true
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R5 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R5_train_n${NPROC}_${PTAG}_best.json || true
else
  echo "[skip] bench"
fi

echo "===== write full_report ====="
export PYTHONPATH=/work
python scripts/write_full_report.py --seed "$SEED" --config "$CFG" || true
echo "===== RESUME R1-R5 DONE $(date -Is) ====="
' 2>&1 | tee -a "$LOG"

  echo "[resume-r123] finished; see $LOG"
}

if [[ "${1:-}" == "--remote" ]]; then
  [[ -n "$REMOTE_HOST" ]] || { echo "Set REMOTE_HOST"; exit 2; }
  # Launch detached on remote so SSH disconnect does not kill training.
  # Checkpoint durability: train.save_every + SIGTERM handler in train_loop.
  LAUNCH_LOG="$NFS_ROOT/logs/resume_r123_launcher_$(date +%Y%m%d_%H%M%S).out"
  REMOTE_CMD="nohup env IMG='$IMG' NFS_ROOT='$NFS_ROOT' NPROC='$NPROC' SEED='$SEED' CONFIG='$CONFIG' SKIP_TRAIN='$SKIP_TRAIN' SKIP_BENCH='$SKIP_BENCH' MAX_BATCH='$MAX_BATCH' SAVE_EVERY='$SAVE_EVERY' bash '$NFS_ROOT/scripts/run_resume_r123.sh' >'$LAUNCH_LOG' 2>&1 & echo LAUNCH_PID=\$! LOG=$LAUNCH_LOG"
  /usr/bin/s"sh" -o BatchMode=yes -o ServerAliveInterval=30 \
    "gemsg@${REMOTE_HOST}" \
    "$REMOTE_CMD"
else
  run_on_gpu
fi
