# fp4-route

Compare **three train/infer routes** on a causal LM using **hardware TE NVFP4** (no software fake-quant).

| Route | Train | Infer |
|-------|-------|-------|
| **R1** | From-scratch BF16 | FP16 |
| **R2** | Same BF16 checkpoint | TE **NVFP4** (block Linears) |
| **R3** | TE **NVFP4** train (block Linears) | TE **NVFP4** |

**Quant scope:** transformer-block `Linear` only; `embed_tokens` + `lm_head` stay high precision.

Architecture: [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) via `from_config` (random init).  
Train: FineWeb-Edu. Eval: WikiText-2 **PPL** + throughput (tokens/s).

**Requires Transformer Engine** — use:

```text
nvcr.io/nvidia/pytorch:26.06-py3
```

TE exposes **NVFP4** (not OCP MXFP4 recipe). Software STE fake-quant is **removed**.

## Quick start

```bash
# Inside NGC PyTorch container:
bash scripts/00_setup_remote.sh   # optional host venv for data prep
source venv/bin/activate          # optional

# Smoke 135M — full R1/R2/R3
NPROC=1 bash scripts/06_run_all.sh --config configs/smoke_135m.yaml --seed 42 --nproc 1

# Mainline 360M
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 42 --nproc 4

# Throughput (bf16 ≈ R1; te_fp4 ≈ R2/R3 infer path)
IMG=nvcr.io/nvidia/pytorch:26.06-py3 bash scripts/run_bench_docker.sh
IMG=nvcr.io/nvidia/pytorch:26.06-py3 NPROC=4 bash scripts/run_bench_max_ddp.sh
```

Configs: `configs/smoke_135m.yaml`, `configs/main_360m.yaml`, `configs/bench_360m.yaml`.  
Key numbers: `EXPERIMENT_SUMMARY.md` / `RUN_STATUS.md` (`results/` gitignored).

## Layout

```
configs/       # smoke / main / bench
mxfp4_lib/     # data, train_loop, te_linear (NVFP4), bench
scripts/
  02_train_bf16.py     # R1/R2 shared BF16 train
  03_train_nvfp4.py    # R3 TE NVFP4 train
  05_eval_ppl.py       # R1 + R2 + R3 WikiText-2 PPL
  06_run_all.sh        # full pipeline
  12_bench_throughput.py / run_bench_*.sh
```

## Checkpoints

```
checkpoints/seed_<N>/
  init_model/
  ckpt_bf16/      # R1 weights; also used for R2 infer
  ckpt_nvfp4/     # R3 weights (saved as nn.Linear after TE train)
```

## Remote

```bash
python3 scripts/remote_run.py --check
REMOTE_HOST=10.x.x.x bash scripts/stage_to_gpu.sh
```

Do not commit GPU IPs or bulky `data/` / `checkpoints/` / `results/`.
