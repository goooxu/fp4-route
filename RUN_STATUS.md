# Run status

**Stack:** **R1–R5** with TE **NVFP4** (R2/R3) + **MXFP8** (R4/R5)  
**Image:** `nvcr.io/nvidia/pytorch:26.07-py3`  
**Train data:** EN FineWeb-Edu 70% + ZH FineWeb-2 (`cmn_Hani`) 30% · total ~7B tokens (`cache_tag=mix73_enzh`)  
**Host:** pass via `REMOTE_HOST` only (no IPs in git)

## Routes

| Route | Train | Infer | Weights |
|-------|-------|-------|---------|
| R1 | BF16 | BF16 | `ckpt_bf16` |
| R2 | same BF16 | TE NVFP4 | `ckpt_bf16` |
| R3 | TE NVFP4 | TE NVFP4 | `ckpt_nvfp4` |
| R4 | same BF16 | TE MXFP8 | `ckpt_bf16` |
| R5 | TE MXFP8 | TE MXFP8 | `ckpt_mxfp8` |

## Seed 42 (mix73 + 26.07) — **DONE**

| Stage | Status |
|-------|--------|
| Prefetch mix73 train cache | Done (`train_tok7000000000_mix73_enzh_seed42.npy`) |
| BF16 / NVFP4 / MXFP8 train (4× GPU DDP) | Done (`checkpoints/seed_42/ckpt_{bf16,nvfp4,mxfp8}/`) |
| WikiText-2 PPL R1–R5 | Done (`results/main_360m/seed_42/metrics.json`) |
| Throughput benches | Done (`results/perf/bench_*_seed42_best.json`) |
| Full report | Done (`results/main_360m/seed_42/full_report.md`) |

### WikiText-2 PPL (seed 42)

| Route | Train | Infer | PPL | vs R1 |
|-------|-------|-------|-----|------:|
| R1 | BF16 | BF16 | **43.12** | 1.00× |
| R2 | BF16 | TE NVFP4 | **45.88** | 1.06× |
| R3 | TE NVFP4 | TE NVFP4 | **43.98** | 1.02× |
| R4 | BF16 | TE MXFP8 | **43.19** | 1.00× |
| R5 | TE MXFP8 | TE MXFP8 | **44.00** | 1.02× |

### Train loss (final / best val)

| Backend | final train loss | best val_loss |
|---------|-----------------:|-------------:|
| BF16 | 2.660 | 2.665 |
| NVFP4 | 2.685 | 2.721 |
| MXFP8 | 2.634 | 2.671 |

### Throughput (best microbench, seed 42, GB200)

| Route | Phase | nGPU | bs/gpu | tokens/s |
|-------|-------|-----:|-------:|---------:|
| R1 BF16 | train | 1 | 192 | 173k |
| R1 BF16 | train | 4 | 192 | **685k** |
| R1 BF16 | infer | 1 | 192 | 527k |
| R2 NVFP4 | infer | 1 | 192 | 439k |
| R3 NVFP4 | train | 1 | 192 | 159k |
| R3 NVFP4 | train | 4 | 160 | 622k |
| R3 NVFP4 | infer | 1 | 192 | 439k |
| R4 MXFP8 | infer | 1 | 192 | 502k |
| R5 MXFP8 | train | 1 | 192 | 173k |
| R5 MXFP8 | train | 4 | 192 | **685k** |
| R5 MXFP8 | infer | 1 | 192 | 502k |

Steady jsonl (4 GPU, from train logs): BF16 ~580k · NVFP4 ~439k · MXFP8 ~515k tok/s.

## Seed 43 (mix73 + 26.07) — **DONE**

| Stage | Status |
|-------|--------|
| Prefetch mix73 train cache | Done (`train_tok7000000000_mix73_enzh_seed43.npy`) |
| BF16 / NVFP4 / MXFP8 train (4× GPU DDP) | Done (`checkpoints/seed_43/ckpt_{bf16,nvfp4,mxfp8}/`) |
| WikiText-2 PPL R1–R5 | Done (`results/main_360m/seed_43/metrics.json`) |
| Throughput benches | Done (`results/perf/bench_*_seed43_best.json`) |
| Full report | Done (`results/main_360m/seed_43/full_report.md`) |

### WikiText-2 PPL (seed 43)

| Route | Train | Infer | PPL | vs R1 |
|-------|-------|-------|-----|------:|
| R1 | BF16 | BF16 | **42.14** | 1.00× |
| R2 | BF16 | TE NVFP4 | **45.61** | 1.08× |
| R3 | TE NVFP4 | TE NVFP4 | **45.50** | 1.08× |
| R4 | BF16 | TE MXFP8 | **42.12** | 1.00× |
| R5 | TE MXFP8 | TE MXFP8 | **42.81** | 1.02× |

### Train loss (final / best val)

| Backend | final train loss | best val_loss |
|---------|-----------------:|-------------:|
| BF16 | 2.624 | 2.675 |
| NVFP4 | 2.667 | 2.726 |
| MXFP8 | 2.599 | 2.679 |

### Throughput (best microbench, seed 43, GB200)

| Route | Phase | nGPU | bs/gpu | tokens/s |
|-------|-------|-----:|-------:|---------:|
| R1 BF16 | train | 1 | 192 | 173k |
| R1 BF16 | train | 4 | 192 | **686k** |
| R1 BF16 | infer | 1 | 192 | 526k |
| R2 NVFP4 | infer | 1 | 192 | 439k |
| R3 NVFP4 | train | 1 | 192 | 159k |
| R3 NVFP4 | train | 4 | 160 | 621k |
| R3 NVFP4 | infer | 1 | 192 | 439k |
| R4 MXFP8 | infer | 1 | 192 | 501k |
| R5 MXFP8 | train | 1 | 192 | 173k |
| R5 MXFP8 | train | 4 | 192 | **685k** |
| R5 MXFP8 | infer | 1 | 192 | 502k |

Steady jsonl (4 GPU): BF16 ~579k · NVFP4 ~464k · MXFP8 ~506k tok/s.

## Ops notes

- Checkpoint durability: `save_every=500`, atomic `.tmp`→rename, SIGTERM/SIGINT save  
- Resume: `scripts/run_resume_r123.sh` (safe; no `seed_*` rename)  
- Eval-only after train: `SKIP_TRAIN=1 bash scripts/run_resume_r123.sh`  
- TE PPL pad: NVFP4 `S%16==0`, MXFP8 `S%32==0` (`scripts/05_eval_ppl.py`)  
- FineWeb-2 ZH config is **`cmn_Hani`** (not BCP-47 `zho_Hans`)  
- Legacy pure-English 26.06 runs: `*_legacy_en_2606_*`  
- Reports filter benches by `seed${SEED}` (`scripts/write_full_report.py`)

> Absolute PPL is **not** comparable to pure-English 26.06 runs (data mix changed).  
> Compare **R1–R5 relative** under the same mix73 + 26.07 stack.  
> Across seeds: R4≈R1 is stable; R3 shows larger seed variance than R1/R4/R5.
