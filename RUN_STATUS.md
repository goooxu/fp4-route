# Run status

**Stack:** **R1–R5** with TE **NVFP4** (R2/R3) + **MXFP8** (R4/R5)  
**Image:** `nvcr.io/nvidia/pytorch:26.07-py3`  
**Train data:** EN FineWeb-Edu 70% + ZH FineWeb-2 (`zho_Hans`) 30% · total ~7B tokens (`cache_tag=mix73_enzh`)  
**Host:** pass via `REMOTE_HOST` only (no IPs in git)

## Routes

| Route | Train | Infer | Weights |
|-------|-------|-------|---------|
| R1 | BF16 | BF16 | `ckpt_bf16` |
| R2 | same BF16 | TE NVFP4 | `ckpt_bf16` |
| R3 | TE NVFP4 | TE NVFP4 | `ckpt_nvfp4` |
| R4 | same BF16 | TE MXFP8 | `ckpt_bf16` |
| R5 | TE MXFP8 | TE MXFP8 | `ckpt_mxfp8` |

## Ops

- Checkpoint durability: `save_every=500`, atomic `.tmp`→rename, SIGTERM/SIGINT save  
- Resume: `scripts/run_resume_r123.sh` (safe; no `seed_*` rename)  
- Recipe selection: `get_recipe("nvfp4"|"mxfp8")` (explicit; not auto-prefer)  
- Legacy pure-English 26.06 runs backed up as `*_legacy_en_2606_*`  

## Seed 42 / 43 (mix73 + 26.07) — IN PROGRESS

| Stage | Status |
|-------|--------|
| Backup old EN-only ckpts/results | Done |
| Pull `pytorch:26.07-py3` | In progress / done on GPU node |
| Prefetch mix73 train cache | Pending / running |
| BF16 + NVFP4 + MXFP8 train | Pending after data |
| PPL R1–R5 + benches | Pending |

Logs: `logs/resume_r123_seed42_*.log` / `logs/full_r15_launcher_seed42_*.out`

> Absolute PPL **not** comparable to pure-English 26.06 runs (data mix changed).  
> Compare **R1–R5 relative** under the same mix73 + 26.07 stack.
