# Run status

**Project (NFS):** `/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare`  
**Stack:** BF16 + **TE NVFP4 only** (software fake-quant **removed**)  
**Image:** `nvcr.io/nvidia/pytorch:26.06-py3`  
**GPU host:** set via `REMOTE_HOST` (no IPs in git)

## Quality (PPL)

| Item | Status |
|------|--------|
| Seed 42 BF16 + NVFP4 retrain with new code | **Not started** (code cut over) |
| Seed 43 | **Not started** |

```bash
# On GPU with TE:
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 42 --nproc 4
```

## Perf (tokens/s) — DONE 2026-07-22

### 1×GPU max-batch

| Backend | Phase | bs | tok/s |
|---------|-------|---:|------:|
| bf16 | train | 192 | **173578** |
| bf16 | infer | 192 | **532828** |
| te_nvfp4 | train | 192 | **156443** |
| te_nvfp4 | infer | 192 | **445168** |

### 4×GPU DDP train

| Backend | bs/gpu | tok/s |
|---------|-------:|------:|
| bf16 | 128 | **669799** |
| te_nvfp4 | 112 | **587034** |

```bash
IMG=nvcr.io/nvidia/pytorch:26.06-py3 NPROC=4 bash scripts/run_bench_max_ddp.sh
```
