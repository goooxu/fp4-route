# Run status

**Stack:** **R1 / R2 / R3** with Transformer Engine **NVFP4**  
**Image:** `nvcr.io/nvidia/pytorch:26.06-py3`  
**Host:** pass via `REMOTE_HOST` only (no IPs in git)

## Routes

| Route | Train | Infer |
|-------|-------|-------|
| R1 | BF16 | FP16 |
| R2 | same BF16 ckpt | TE NVFP4 |
| R3 | TE NVFP4 | TE NVFP4 |

## Ops

- Checkpoint durability: `save_every=500`, atomic `.tmp`→rename, SIGTERM/SIGINT save  
- Resume (safe, no rename of `seed_*`): `scripts/run_resume_r123.sh`  
- Finish-only (export/PPL/bench if train done): `scripts/run_finish_r123.sh`  
- R3 HF export from resume if TE final save fails: `scripts/04_export_nvfp4_from_resume.py`  
- Bench outputs seed-tagged: `results/perf/bench_*_seed${SEED}_best.json`  
- No scheduled tasks; re-launch on new GPU node via `REMOTE_HOST` + `run_resume_r123.sh`

---

## Seed 42 — **COMPLETE** (2026-07-23)

| Stage | Status |
|-------|--------|
| Init + BF16 train (R1/R2) | **Done** 53406/53406 |
| R3 TE NVFP4 train | **Done** 53406/53406 (resume ok; final HF via export if needed) |
| PPL R1/R2/R3 | **Done** → `results/main_360m/seed_42/metrics.json` |
| Throughput benches | **Done** (incl. R3 train 4GPU) |
| Full report | `results/main_360m/seed_42/full_report.md` |

### Quality (WikiText-2 PPL)

| Route | Train | Infer | PPL |
|-------|-------|-------|-----|
| R1 | bf16 | fp16 | **51.94** |
| R2 | bf16 | te_nvfp4 | **54.07** |
| R3 | te_nvfp4 | te_nvfp4 | **53.32** |

### Perf best (tokens/s, seq=512, trained ckpts)

| Config | nGPU | bs/gpu | tok/s |
|--------|-----:|-------:|------:|
| R1 train | 1 | 192 | **174672** |
| R1 infer | 1 | 192 | **524557** |
| R1 train | 4 | 192 | **688153** |
| R2 infer | 1 | 192 | **444015** |
| R3 train | 1 | 192 | **156553** |
| R3 infer | 1 | 192 | **443179** |
| R3 train | 4 | 192 | **616741** |

Steady train logs (jsonl): BF16 ~**555k** tok/s; NVFP4 ~**448k** tok/s (4GPU).

---

## Seed 43 — **COMPLETE** (2026-08-01)

| Stage | Status |
|-------|--------|
| Init + BF16 train (R1/R2) | **Done** 53406/53406 |
| R3 TE NVFP4 train | **Done** 53406/53406 (resumed mid-run from step 1500) |
| PPL R1/R2/R3 | **Done** → `results/main_360m/seed_43/metrics.json` |
| Throughput benches | **Done** → `results/perf/*_seed43_best.json` |
| Full report | `results/main_360m/seed_43/full_report.md` |

### Quality (WikiText-2 PPL)

| Route | Train | Infer | PPL |
|-------|-------|-------|-----|
| R1 | bf16 | fp16 | **37.07** |
| R2 | bf16 | te_nvfp4 | **40.08** |
| R3 | te_nvfp4 | te_nvfp4 | **39.62** |

### Perf best (tokens/s, seq=512, seed43 tags)

| Config | nGPU | bs/gpu | tok/s |
|--------|-----:|-------:|------:|
| R1 train | 1 | 160 | **171657** |
| R1 infer | 1 | 192 | **521817** |
| R1 train | 4 | 192 | **681069** |
| R2 infer | 1 | 192 | **441062** |
| R3 train | 1 | 192 | **155668** |
| R3 infer | 1 | 192 | **440909** |
| R3 train | 4 | 192 | **602222** |

Steady train logs (jsonl): BF16 ~**573k** tok/s; NVFP4 ~**439k** tok/s (4GPU).

---

## Cross-seed notes

| | Seed 42 | Seed 43 |
|--|--------:|--------:|
| R1 PPL | 51.94 | **37.07** |
| R2 PPL | 54.07 | **40.08** |
| R3 PPL | 53.32 | **39.62** |
| R3 vs R1 ΔPPL | +1.38 | +2.55 |
| R3 train 4GPU / R1 | 0.90× | 0.88× |

Pattern holds on both seeds: **R3 略好于 R2**；相对 R1 有小幅 PPL 代价；NVFP4 训练吞吐约 R1 的 **0.85–0.90×**。

### Close-out fixes (keep)

1. `revert_te_to_linear`: ignore empty TE bias Parameters  
2. CPU export from `ckpt_nvfp4/resume` when final HF save fails  
3. PPL TE pad seq to multiple of **16** (NVFP4 block)  
4. `write_full_report.py` `sys.path` / `PYTHONPATH`  
5. Bench filenames include `seed${SEED}` to avoid clobber  
