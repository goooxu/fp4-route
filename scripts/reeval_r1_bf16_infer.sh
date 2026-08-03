#!/usr/bin/env bash
# Re-eval PPL R1/R2/R3 + R1 infer bench with R1 = BF16 infer (no retrain).
set -euo pipefail
ROOT="${NFS_ROOT:-/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare}"
IMG="${IMG:-nvcr.io/nvidia/pytorch:26.07-py3}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG:-$ROOT/logs/r1_bf16_infer_reeval_${TS}.log}"
mkdir -p "$ROOT/logs"
echo "[reeval] log=$LOG host=$(hostname)" | tee -a "$LOG"

docker run --rm -i --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/tmp/hf_home \
  -e PYTHONPATH=/work \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$ROOT:/work" -w /work "$IMG" \
  python - <<'PY' 2>&1 | tee -a "$LOG"
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/work")
os.chdir("/work")

try:
    import transformers  # noqa: F401
except Exception:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore",
         "transformers", "datasets", "pyyaml", "tqdm", "accelerate", "safetensors"]
    )

from mxfp4_lib.util import load_cfg


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


for seed in (42, 43):
    arch = f"/work/checkpoints/seed_{seed}_legacy_swfq_ARCH"
    env = os.environ.copy()
    env["ARCH_CONFIG_DIR"] = arch
    env["PYTHONPATH"] = "/work"
    print(f"===== PPL seed={seed} R1,R2,R3 =====", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/05_eval_ppl.py",
            "--config",
            "configs/main_360m.yaml",
            "--seed",
            str(seed),
            "--routes",
            "R1,R2,R3",
        ],
        env=env,
    )

    print(f"===== R1 infer bench seed={seed} =====", flush=True)
    cfg = load_cfg("configs/main_360m.yaml", seed=seed)
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
    bench["seed"] = seed
    by = Path(f"/tmp/bench_r1_seed{seed}.yaml")
    by.write_text(yaml.safe_dump(bench, sort_keys=False))
    try:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/12_bench_throughput.py",
                "--config",
                str(by),
                "--route",
                "R1",
                "--phase",
                "infer",
                "--sweep",
                "--max-batch",
                "192",
                "--warmup",
                "3",
                "--measure",
                "8",
                "--out",
                f"results/perf/bench_R1_infer_n1_seed{seed}_best.json",
            ],
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"[warn] bench seed={seed} failed: {e}", flush=True)

    try:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/write_full_report.py",
                "--seed",
                str(seed),
                "--config",
                "configs/main_360m.yaml",
            ],
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"[warn] report seed={seed} failed: {e}", flush=True)

print("===== REEVAL DONE =====", flush=True)
PY

echo "[reeval] finished; see $LOG"
