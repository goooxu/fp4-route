# fp4-route

Compare three train / infer routes for **full-model MXFP4 (W4A4 semantics on transformer-block Linears)** on a causal LM.

| Route | Train | Infer |
|-------|-------|-------|
| R1 | From-scratch BF16 | FP16 (true fp16 forward, no bf16 autocast) |
| R2 | Same BF16 ckpt → PTQ on **block Linears only** | MXFP4 W4A4 (embed + lm_head stay BF16/FP16) |
| R3 | From-scratch MXFP4 fake-quant (blocks) | MXFP4 W4A4 |

**Quant scope (default):** industrial standard — keep `embed_tokens` + `lm_head` in high precision; quantize transformer-block `nn.Linear` only (~224 on SmolLM2-360M). Optional ablation unties `lm_head` and quantizes it too.

Default setup borrows architecture of [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) via `from_config` (random weights). Training data: **FineWeb-Edu** (`sample-10BT`). Eval: **WikiText-2** PPL only.

MXFP4: OCP-style E2M1 + group=32 + E8M0 scale (`scale_mode=rtn` default). PyTorch fake-quant — not Transformer Engine hardware MXFP4 GEMM.

**Scope of this repo:** inference **quality** (PPL). Throughput is not a reported metric, but training is tuned to keep GPUs busy (large micro-batch, TF32, DataLoader prefetch, optimized FQ kernels, optional `torch.compile` on `Mxfp4Linear`).

## Quick start

```bash
bash scripts/00_setup_remote.sh
source venv/bin/activate

# 135M smoke (~200M FineWeb tokens)
bash scripts/run_smoke_135m.sh

# 360M mainline × seeds 42/43 (DDP)
NPROC=4 bash scripts/run_main_360m.sh

# Pretrained baseline + ablations + QAT-from-pretrained
bash scripts/run_p2_p3.sh
```

Configs: `configs/smoke_135m.yaml`, `configs/main_360m.yaml`.

## Layout

```
configs/          # smoke / main hyperparams
mxfp4_lib/        # quant, Linear, replace, data, train loop, asserts
scripts/          # prepare → train → PTQ → eval → ablations
tests/            # quant + tie-scope unit tests
EXPERIMENT_SUMMARY.md   # conclusions + retained dataset paths
data/             # local FineWeb/WikiText caches (gitignored; see SUMMARY)
```

Run artifacts (`results/`, `checkpoints/`) are gitignored; copy key numbers into `EXPERIMENT_SUMMARY.md`.

## Checkpoints & machine migration

**Preferred layout:** train with the project on **NFS** (e.g. scratch) so `checkpoints/` and `results/` are durable without rsync. Checkpoints stay under the project working directory.

Training writes resumable state under each ckpt dir:

```
checkpoints/seed_<N>/.../resume/
  train_state.pt      # step + optimizer + rng
  model_state.pt      # weights
  checkpoint_meta.json
  tokenizer/
```

- `train.save_every` (mainline default **500**) — periodic save on NFS
- `train.resume: true` — auto-continue from `resume/`
- SIGTERM/SIGINT — one more checkpoint then exit (for node reclaim)

**GPU node helpers** (from a login node without CUDA):

```bash
# Alive + GPU + project path
python3 scripts/remote_run.py --check
python3 scripts/remote_run.py --host 10.x.x.x 'nvidia-smi -L'

# Training health (meta + log tail)
python3 scripts/health_check.py --host 10.x.x.x --seed 42
```

If the ephemeral GPU node dies: stop work, get a new IP, then:

```bash
bash scripts/00_setup_remote.sh   # on the new node if venv missing
NPROC=4 bash scripts/resume_main.sh configs/main_360m.yaml 42
```

Optional non-NFS backup (pass host via env, never commit IPs):

```bash
REMOTE=user@host REMOTE_DIR=/tmp/fp4_route \
  bash scripts/sync_artifacts.sh pull|push
```
