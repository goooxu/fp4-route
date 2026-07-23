#!/usr/bin/env bash
# Finish R123 after train: export R3 HF from resume (if needed) + PPL + benches.
# Does NOT retrain. Safe to re-run. Writes durable artifacts to NFS.
#
# On GPU node:
#   NPROC=4 SEED=42 bash scripts/run_finish_r123.sh
# Detached from login:
#   python scripts/remote_run.py --host HOST 'nohup ... run_finish_r123.sh > logs/... &'
set -euo pipefail

IMG="${IMG:-nvcr.io/nvidia/pytorch:26.06-py3}"
NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
NPROC="${NPROC:-4}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/main_360m.yaml}"
MAX_BATCH="${MAX_BATCH:-192}"
SKIP_BENCH="${SKIP_BENCH:-0}"

mkdir -p "$NFS_ROOT/logs" "$NFS_ROOT/results/perf"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$NFS_ROOT/logs/finish_r123_seed${SEED}_${TS}.log"
echo "[finish-r123] log=$LOG host=$(hostname) nproc=$NPROC seed=$SEED" | tee -a "$LOG"
chmod -R a+rwX "$NFS_ROOT/checkpoints" "$NFS_ROOT/results" "$NFS_ROOT/logs" 2>/dev/null || true

docker pull "$IMG" 2>&1 | tail -3 | tee -a "$LOG" || true

docker run --rm --gpus all --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 \
  -e NPROC="$NPROC" \
  -e SEED="$SEED" \
  -e CONFIG="$CONFIG" \
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

SEED=${SEED:-42}
CONFIG=${CONFIG:-configs/main_360m.yaml}
NPROC=${NPROC:-4}
MAX_BATCH=${MAX_BATCH:-192}

echo "===== [1/4] ensure R3 HF export ====="
if [[ -f checkpoints/seed_${SEED}/ckpt_nvfp4/model.safetensors ]]; then
  echo "R3 HF model.safetensors already present"
else
  echo "exporting from resume..."
  python scripts/04_export_nvfp4_from_resume.py --config "$CONFIG" --seed "$SEED"
fi
test -f checkpoints/seed_${SEED}/ckpt_bf16/model.safetensors
test -f checkpoints/seed_${SEED}/ckpt_nvfp4/model.safetensors
ls -la checkpoints/seed_${SEED}/ckpt_bf16/model.safetensors checkpoints/seed_${SEED}/ckpt_nvfp4/model.safetensors

echo "===== [2/4] PPL R1,R2,R3 ====="
python scripts/05_eval_ppl.py --config "$CONFIG" --seed "$SEED" --routes R1,R2,R3

if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
  echo "===== [3/4] throughput benches ====="
  export PYTHONPATH=/work
  # Ensure GPUs visible after eval (never leave empty CUDA_VISIBLE_DEVICES)
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  python -c 'import torch; print("[bench] cuda", torch.cuda.is_available(), "n", torch.cuda.device_count())'
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
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R1 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_train_n1_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R1 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_infer_n1_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R2 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R2_infer_n1_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R3 --phase train --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_train_n1_best.json || true
  CUDA_VISIBLE_DEVICES=0 python scripts/12_bench_throughput.py --config $BC --route R3 --phase infer --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_infer_n1_best.json || true
  unset CUDA_VISIBLE_DEVICES
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R1 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R1_train_n${NPROC}_best.json || true
  torchrun --standalone --nproc_per_node=$NPROC scripts/12_bench_throughput.py --config $BC --route R3 --phase train --ddp --sweep --max-batch $MAX_BATCH --warmup $WARM --measure $MEAS \
    --out results/perf/bench_R3_train_n${NPROC}_best.json || true
else
  echo "[skip] bench"
fi

echo "===== [4/4] full_report ====="
python scripts/write_full_report.py --seed "$SEED" --config "$CONFIG" || true
echo "===== FINISH R123 DONE $(date -Is) ====="
' 2>&1 | tee -a "$LOG"

echo "[finish-r123] done; see $LOG"
