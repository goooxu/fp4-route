# Run status

**Project (NFS):** `/home/scratch.gemsg_sw/grokbuild/mxfp4_route_compare`  
**Last GPU host:** set via `REMOTE_HOST` (ephemeral; do not hardcode IPs in git)

## Seed 42 — COMPLETE (2026-07-22)

| Stage | Status | Notes |
|-------|--------|--------|
| BF16 (R1) | **DONE** | 53406 steps; best val 3.24 |
| MXFP4 FQ (R3) | **DONE** | 53406 steps; best val 3.30 |
| PTQ (R2) | **DONE** | `checkpoints/seed_42/ckpt_bf16_mxfp4_ptq` |
| WikiText-2 eval | **DONE** | `results/main_360m/seed_42/metrics.json` |

### WikiText-2 PPL (seed 42)

| Route | PPL |
|-------|-----|
| R1 BF16→FP16 | **52.37** |
| R2 BF16→PTQ blocks | **67.24** |
| R3 MXFP4 FQ | **53.73** |

Logs: `logs/stage_resume_20260722_050732.log` (FQ), `logs/ptq_eval_seed42_20260722_165338.log` (PTQ+eval).

## Seed 43 — NOT STARTED

```bash
# On GPU node:
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 43 --nproc 4
# Or stage-local resume helper if cold NFS:
# bash scripts/stage_local_and_resume.sh  # currently hardcodes seed 42 — adapt if needed
```
