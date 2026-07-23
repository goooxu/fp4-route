# Run status

**Stack:** **R1 / R2 / R3** with TE **NVFP4** (software fake-quant removed)  
**Image:** `nvcr.io/nvidia/pytorch:26.06-py3`  
**Host:** pass via `REMOTE_HOST` only (no IPs in git)

## Routes

| Route | Train | Infer |
|-------|-------|-------|
| R1 | BF16 | FP16 |
| R2 | same BF16 ckpt | TE NVFP4 |
| R3 | TE NVFP4 | TE NVFP4 |

## Seed 42 from-scratch — **COMPLETE** (2026-07-23)

| Stage | Status |
|-------|--------|
| Init + BF16 train (R1/R2) | **Done** 53406/53406 |
| R3 TE NVFP4 train | **Done** 53406/53406 (resume ok; final HF via `04_export_nvfp4_from_resume.py`) |
| PPL R1/R2/R3 | **Done** → `results/main_360m/seed_42/metrics.json` |
| Throughput benches | **Done** (incl. R3 train 4GPU) |
| Full report | `results/main_360m/seed_42/full_report.md` |

Checkpoint durability: `save_every=500`, atomic `.tmp`→rename, SIGTERM/SIGINT save.  
Resume scripts: `run_resume_r123.sh` / `run_finish_r123.sh`. No scheduled tasks.

### Quality (WikiText-2 PPL)

| Route | Train | Infer | PPL |
|-------|-------|-------|-----|
| R1 | bf16 | fp16 | **51.94** |
| R2 | bf16 | te_nvfp4 | **54.07** |
| R3 | te_nvfp4 | te_nvfp4 | **53.32** |

### Perf best (tokens/s, seq=512)

| Config | nGPU | bs/gpu | tok/s |
|--------|-----:|-------:|------:|
| R1 train | 1 | 192 | **174672** |
| R1 infer | 1 | 192 | **524557** |
| R1 train | 4 | 192 | **688153** |
| R2 infer | 1 | 192 | **444015** |
| R3 train | 1 | 192 | **156553** |
| R3 infer | 1 | 192 | **443179** |
| R3 train | 4 | 192 | **616741** |

### Fixes applied during close-out

1. `revert_te_to_linear`: ignore empty TE bias Parameters  
2. CPU export from `ckpt_nvfp4/resume` when final HF save fails  
3. PPL TE pad seq to multiple of **16** (NVFP4 block)  
4. `write_full_report.py` `sys.path` fix  

## Seed 43

**Not started**
