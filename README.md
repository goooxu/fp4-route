# fp4-route

Compare **BF16** vs **Transformer Engine NVFP4** (hardware Tensor Core) on a causal LM.

| Route | Train | Infer |
|-------|-------|-------|
| BF16 | From-scratch BF16 | FP16 forward |
| NVFP4 | TE `NVFP4BlockScaling` on block Linears | Same TE recipe |

**Quant scope:** transformer-block `Linear` only; `embed_tokens` + `lm_head` stay high precision.

Architecture: [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) via `from_config` (random init).  
Train data: FineWeb-Edu. Eval: WikiText-2 PPL + throughput (tokens/s).

**Requires Transformer Engine** — use:

```text
nvcr.io/nvidia/pytorch:26.06-py3
```

Software **fake-quant MXFP4 is removed**. Historical software-FQ numbers (if any) are archive-only in `EXPERIMENT_SUMMARY.md`.

## Quick start

```bash
# Inside NGC PyTorch container (or host with TE + CUDA):
bash scripts/00_setup_remote.sh   # optional host venv; TE better from NGC
source venv/bin/activate          # if using host venv for data prep only

# Smoke 135M
NPROC=1 bash scripts/06_run_all.sh --config configs/smoke_135m.yaml --seed 42 --nproc 1

# Mainline 360M × seed
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 42 --nproc 4

# Throughput (BF16 vs TE NVFP4), on GPU node:
IMG=nvcr.io/nvidia/pytorch:26.06-py3 bash scripts/run_bench_docker.sh
IMG=nvcr.io/nvidia/pytorch:26.06-py3 NPROC=4 bash scripts/run_bench_max_ddp.sh
```

Configs: `configs/smoke_135m.yaml`, `configs/main_360m.yaml`, `configs/bench_360m.yaml`.  
Metrics live in `EXPERIMENT_SUMMARY.md` / `RUN_STATUS.md` (`results/` gitignored).

## Layout

```
configs/          # smoke / main / bench
mxfp4_lib/        # data, train_loop, te_linear, bench
scripts/          # prepare → BF16 train → NVFP4 train → eval / bench
```

## Checkpoints

```
checkpoints/seed_<N>/
  init_model/
  ckpt_bf16/
  ckpt_nvfp4/          # HF weights (nn.Linear after save); USE_NVFP4 marker
  .../resume/          # train_state + model_state for resume
```

- `train.save_every` — periodic resume on NFS  
- `train.resume: true` — continue from `resume/`  
- SIGTERM/SIGINT — one more checkpoint then exit  

## Remote GPU

```bash
python3 scripts/remote_run.py --check
python3 scripts/health_check.py --host 10.x.x.x --seed 42
REMOTE_HOST=10.x.x.x bash scripts/stage_to_gpu.sh
```

Do **not** commit GPU IPs or bulky `data/` / `checkpoints/` / `results/`.
