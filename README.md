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

**Scope of this repo (dual track):**

1. **Quality:** WikiText-2 **PPL** for R1/R2/R3 (software MXFP4 fake-quant; fair route compare).  
2. **Performance:** train/infer **tokens/s** for `bf16` / `sw_fq` / (when available) TE hardware FP4.

Software MXFP4 is **not** Tensor Core FP4 GEMM. Hardware path uses Transformer Engine recipes (e.g. NVFP4) when installable — see `EXPERIMENT_SUMMARY.md` §3.5.

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

# Official SmolLM2 FP16 + block PTQ only (cold-NFS safe: login streams venv → GPU /tmp)
REMOTE_HOST=<gpu-ip> bash scripts/run_pretrained_baseline.sh

# Throughput microbench (stage venv first on cold NFS)
REMOTE_HOST=<gpu-ip> bash scripts/stage_to_gpu.sh
# on GPU:
python scripts/12_bench_throughput.py --backend bf16 --phase train
python scripts/12_bench_throughput.py --backend sw_fq --phase infer --batch-size 32
# hardware FP4 (needs TE torch build/wheel):
python scripts/11_probe_hw_fp4.py
python scripts/12_bench_throughput.py --backend te_fp4 --phase train
```

Configs: `configs/smoke_135m.yaml`, `configs/main_360m.yaml`, `configs/bench_360m.yaml`.  
Key numbers live in `EXPERIMENT_SUMMARY.md` / `RUN_STATUS.md` (`results/` is gitignored).

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
