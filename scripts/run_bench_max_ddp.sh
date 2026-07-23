#!/usr/bin/env bash
# Batch-fill (1 GPU) + 4-GPU DDP throughput in NGC container.
# Each backend/phase is a separate process so one CUDA/NVML glitch does not kill the suite.
#
# On GPU node:
#   IMG=nvcr.io/nvidia/pytorch:26.06-py3 bash scripts/run_bench_max_ddp.sh
set -euo pipefail

IMG="${IMG:-nvcr.io/nvidia/pytorch:26.06-py3}"
NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
REMOTE_HOST="${REMOTE_HOST:-}"
NPROC="${NPROC:-4}"
MAX_BATCH="${MAX_BATCH:-256}"

run_local() {
  mkdir -p "$NFS_ROOT/results/perf" "$NFS_ROOT/logs"
  LOCAL_OUT="/tmp/mxfp4_bench_max_$$"
  mkdir -p "$LOCAL_OUT/results/perf"
  chmod -R a+rwx "$LOCAL_OUT" 2>/dev/null || true
  LOG="$NFS_ROOT/logs/bench_max_ddp_$(date +%Y%m%d_%H%M%S).log"
  echo "[bench-max] image=$IMG nproc=$NPROC max_batch=$MAX_BATCH log=$LOG"

  docker run --rm --gpus all --network host \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONUNBUFFERED=1 \
    -e NPROC="$NPROC" \
    -e MAX_BATCH="$MAX_BATCH" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/tmp/hf_cache \
    -v "$NFS_ROOT:/work:ro" \
    -v "$LOCAL_OUT:/out" \
    -w /work \
    "$IMG" \
    bash -lc '
set -uo pipefail
mkdir -p /tmp/proj
for d in mxfp4_lib scripts configs; do cp -a "/work/$d" /tmp/proj/; done
cd /tmp/proj
python - <<PY
import yaml
from pathlib import Path
p = Path("configs/bench_360m.yaml")
cfg = yaml.safe_load(p.read_text())
cfg.setdefault("paths", {})["results"] = "/out/results/perf"
cfg["paths"]["root"] = "/tmp/proj"
cfg["paths"]["ckpt_bf16"] = "/tmp/proj/checkpoints/__none__"
cfg["paths"]["init_model"] = "/tmp/proj/checkpoints/__none__"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("cfg ok")
PY
python -c "import transformers, yaml" 2>/dev/null || pip install -q --root-user-action=ignore "transformers>=4.46" pyyaml tqdm accelerate safetensors
python -c "import torch; print(\"torch\", torch.__version__, \"gpus\", torch.cuda.device_count(), torch.cuda.get_device_name(0))"
python -c "import transformer_engine as te; print(\"te\", te.__version__)"

NPROC=${NPROC:-4}
MAX_BATCH=${MAX_BATCH:-256}
WARM=3
MEAS=8
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "NGPU=$NGPU"

run_one() {
  local backend=$1 phase=$2 mode=$3  # mode: n1 | ddp
  local start=$4
  local out tag
  if [[ "$mode" == "n1" ]]; then
    tag="n1"
    echo "===== SWEEP 1GPU $backend $phase start=$start max=$MAX_BATCH ====="
    CUDA_VISIBLE_DEVICES=0 python -u scripts/12_bench_throughput.py \
      --config configs/bench_360m.yaml \
      --backend "$backend" --phase "$phase" \
      --sweep --start-batch "$start" --max-batch "$MAX_BATCH" \
      --warmup "$WARM" --measure "$MEAS" \
      --out "/out/results/perf/bench_${backend}_${phase}_${tag}_best.json" \
      && return 0
    echo "FAIL $backend $phase $mode"
    return 1
  else
    tag="n${NPROC}"
    echo "===== DDP n=$NPROC $backend $phase start=$start ====="
    torchrun --standalone --nproc_per_node="$NPROC" \
      scripts/12_bench_throughput.py \
      --config configs/bench_360m.yaml \
      --backend "$backend" --phase "$phase" \
      --ddp --sweep --start-batch "$start" --max-batch "$MAX_BATCH" \
      --warmup "$WARM" --measure "$MEAS" \
      --out "/out/results/perf/bench_${backend}_${phase}_${tag}_best.json" \
      && return 0
    echo "FAIL $backend $phase $mode"
    return 1
  fi
}

# 1-GPU sweeps (separate process each)
for backend in bf16 sw_fq te_fp4; do
  for phase in train infer; do
    start=32
    [[ "$backend" == "sw_fq" && "$phase" == "infer" ]] && start=8
    run_one "$backend" "$phase" n1 "$start" || true
  done
done

# 4-GPU DDP train sweeps
if [[ "$NGPU" -ge 2 ]]; then
  NPROC=$(( NGPU < NPROC ? NGPU : NPROC ))
  for backend in bf16 sw_fq te_fp4; do
    start=16
    [[ "$backend" == "sw_fq" ]] && start=8
    run_one "$backend" train ddp "$start" || true
  done
else
  echo "[warn] NGPU=$NGPU < 2; skip DDP"
fi

echo "===== ALL DONE ====="
ls -la /out/results/perf/
python - <<PY
import json, glob
print("\\n=== SUMMARY ===")
for p in sorted(glob.glob("/out/results/perf/bench_*_best.json")):
    d=json.load(open(p))
    print(f"{d[\"backend\"]:16s} {d[\"phase\"]:5s} n={d[\"nproc\"]} bs={d[\"batch_size\"]:4d}  "
          f"{d[\"tokens_per_sec\"]:10.1f} tok/s  mem={d[\"peak_mem_gb\"]:.1f}GB")
for p in sorted(glob.glob("/out/results/perf/sweep_*.json")):
    d=json.load(open(p))
    print(f"sweep {d[\"backend\"]} {d[\"phase\"]} n={d[\"world_size\"]} "
          f"max_bs={d[\"max_batch_no_oom\"]} best_bs={d[\"best_batch\"]} best={d[\"best_tokens_per_sec\"]:.1f}")
PY
'

  mkdir -p "$NFS_ROOT/results/perf"
  cp -f "$LOCAL_OUT/results/perf/"*.json "$NFS_ROOT/results/perf/" 2>/dev/null || true
  echo "[bench-max] copied -> $NFS_ROOT/results/perf"
  ls -la "$NFS_ROOT/results/perf/" | tail -40
}

if [[ "${1:-}" == "--remote" ]]; then
  [[ -n "$REMOTE_HOST" ]] || { echo "Set REMOTE_HOST"; exit 2; }
  /usr/bin/ssh -o BatchMode=yes -o ServerAliveInterval=30 \
    "gemsg@${REMOTE_HOST}" \
    "IMG=$IMG NFS_ROOT=$NFS_ROOT NPROC=$NPROC MAX_BATCH=$MAX_BATCH bash $NFS_ROOT/scripts/run_bench_max_ddp.sh"
else
  run_local 2>&1 | tee "${NFS_ROOT}/logs/bench_max_ddp_latest.log"
fi
