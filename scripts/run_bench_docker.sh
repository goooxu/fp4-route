#!/usr/bin/env bash
# Run throughput benches inside NGC PyTorch container (TE + NVFP4).
#
# Usage (on GPU node with docker):
#   bash scripts/run_bench_docker.sh
#   IMG=nvcr.io/nvidia/pytorch:26.06-py3 bash scripts/run_bench_docker.sh
#
# Or from login:
#   REMOTE_HOST=10.x.x.x bash scripts/run_bench_docker.sh --remote
set -euo pipefail

IMG="${IMG:-nvcr.io/nvidia/pytorch:26.06-py3}"
NFS_ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
REMOTE_HOST="${REMOTE_HOST:-}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

run_local() {
  mkdir -p "$NFS_ROOT/results/perf" "$NFS_ROOT/logs"
  LOG="$NFS_ROOT/logs/bench_docker_$(date +%Y%m%d_%H%M%S).log"
  echo "[bench-docker] image=$IMG log=$LOG"
  # Writable results on node local disk (NFS ownership issues if docker runs as root)
  LOCAL_OUT="/tmp/mxfp4_bench_out_$$"
  mkdir -p "$LOCAL_OUT/results/perf"
  chmod -R a+rwx "$LOCAL_OUT" 2>/dev/null || true
  docker run --rm --gpus all --network host \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/tmp/hf_cache \
    -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    -v "$NFS_ROOT:/work:ro" \
    -v "$LOCAL_OUT:/out" \
    -w /work \
    "$IMG" \
    bash -lc '
set -euo pipefail
# Only code dirs (never copy data/checkpoints/venv from cold NFS)
mkdir -p /tmp/proj
for d in mxfp4_lib scripts configs; do
  cp -a "/work/$d" /tmp/proj/
done
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
print("bench cfg results=", cfg["paths"]["results"])
PY
echo "[container] torch $(python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))")"
echo "[container] te $(python -c "import transformer_engine as te; print(te.__version__)")"
python -c "import transformers, yaml" 2>/dev/null || pip install -q --root-user-action=ignore "transformers>=4.46" pyyaml tqdm accelerate safetensors
python -u scripts/11_probe_hw_fp4.py --config configs/bench_360m.yaml || true
cp -f /tmp/capability.json /out/results/perf/capability.json 2>/dev/null || true
cp -f /out/results/perf/capability.json /out/results/perf/capability.json 2>/dev/null || true
# probe writes under paths.results if writable
ls -la /out/results/perf/ || true
for backend in bf16 sw_fq te_fp4; do
  for phase in train infer; do
    bs=64
    if [[ "$backend" == "sw_fq" && "$phase" == "infer" ]]; then bs=32; fi
    echo "===== $backend $phase bs=$bs ====="
    python -u scripts/12_bench_throughput.py \
      --config configs/bench_360m.yaml \
      --backend "$backend" --phase "$phase" \
      --batch-size "$bs" --warmup 5 --measure 20 \
      --out "/out/results/perf/bench_${backend}_${phase}_bs${bs}.json" \
      || echo "FAIL $backend $phase"
  done
done
ls -la /out/results/perf/
'
  # Copy results back to NFS as the invoking user
  mkdir -p "$NFS_ROOT/results/perf"
  cp -f "$LOCAL_OUT/results/perf/"*.json "$NFS_ROOT/results/perf/" 2>/dev/null || true
  echo "[bench-docker] results copied to $NFS_ROOT/results/perf/"
  ls -la "$NFS_ROOT/results/perf/" || true
}

if [[ "${1:-}" == "--remote" ]]; then
  if [[ -z "$REMOTE_HOST" ]]; then
    echo "Set REMOTE_HOST for --remote" >&2
    exit 2
  fi
  # shellcheck disable=SC2029
  /usr/bin/ssh -o BatchMode=yes -o ServerAliveInterval=30 \
    "gemsg@${REMOTE_HOST}" \
    "IMG=$IMG NFS_ROOT=$NFS_ROOT CUDA_VISIBLE_DEVICES=$GPU bash $NFS_ROOT/scripts/run_bench_docker.sh"
else
  run_local 2>&1 | tee "${NFS_ROOT}/logs/bench_docker_latest.log"
fi
