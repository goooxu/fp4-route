# Run status

**Stack:** **R1 / R2 / R3** with TE **NVFP4** (software fake-quant removed)  
**Image:** `nvcr.io/nvidia/pytorch:26.06-py3`  
**Host:** `REMOTE_HOST` only (no IPs in git)

## Routes

| Route | Train | Infer |
|-------|-------|-------|
| R1 | BF16 | FP16 |
| R2 | same BF16 ckpt | TE NVFP4 |
| R3 | TE NVFP4 | TE NVFP4 |

## Quality (PPL)

| Item | Status |
|------|--------|
| Seed 42 R1/R2/R3 with TE code | **Not started** (need retrain) |
| Seed 43 | **Not started** |

```bash
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 42 --nproc 4
# eval only:
python scripts/05_eval_ppl.py --config configs/main_360m.yaml --seed 42 --routes R1,R2,R3
```

## Perf (tokens/s) — DONE 2026-07-22

`bf16` ≈ R1; `te_nvfp4` ≈ R2/R3 hardware path.

| Backend | Phase | nGPU | best tok/s |
|---------|-------|-----:|-----------:|
| bf16 | train | 1 | 173578 (bs192) |
| bf16 | infer | 1 | 532828 (bs192) |
| te_nvfp4 | train | 1 | 156443 (bs192) |
| te_nvfp4 | infer | 1 | 445168 (bs192) |
| bf16 | train | 4 | 669799 |
| te_nvfp4 | train | 4 | 587034 |
