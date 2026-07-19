# fp4-route

Compare three train / infer routes for **full-model MXFP4 (W4A4 semantics)** on a causal LM.

| Route | Train | Infer |
|-------|-------|-------|
| R1 | From-scratch BF16 | FP16 |
| R2 | Same BF16 ckpt as R1 → full-model PTQ | MXFP4 (static weight PTQ + dynamic activation quant) |
| R3 | From-scratch MXFP4 fake-quant (STE) | MXFP4 |

Default setup borrows the **architecture** of [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) via `from_config` (**random weights**; no pretrained tensors). Metric: WikiText-2 sliding-window perplexity.

MXFP4 here follows an OCP-style layout (E2M1 + group=32 + E8M0 scale) implemented in PyTorch fake-quant — not Transformer Engine hardware MXFP4 GEMM.

## Quick start

```bash
# Setup (venv + deps; prefers CUDA wheels when available)
bash scripts/00_setup_remote.sh
source venv/bin/activate

# Full pipeline: data → init → BF16 train → MXFP4 FQ train → PTQ → eval
bash scripts/06_run_all.sh
```

Config: `configs/train.yaml`. Results land in `results/`.

## Layout

```
configs/          # train / eval hyperparams
mxfp4_lib/        # MXFP4 quant, Linear, replace, train loop
scripts/          # prepare → train → PTQ → eval
results/          # PPL metrics, train logs, summary
EXPERIMENT_SUMMARY.md
```

## Hardware note

Experiments in this repo were run on NVIDIA **Blackwell (GB200, SM 10.0)**. The code is standard PyTorch CUDA and should run on other GPUs with enough memory (adjust batch size / model size in the config as needed).
