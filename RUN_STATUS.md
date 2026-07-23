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

### WikiText-2 PPL (seed 42, from-scratch)

| Route | PPL |
|-------|-----|
| R1 BF16→FP16 | **52.37** |
| R2 BF16→PTQ blocks | **67.24** |
| R3 MXFP4 FQ | **53.73** |

Logs: `logs/stage_resume_20260722_050732.log` (FQ), `logs/ptq_eval_seed42_20260722_165338.log` (PTQ+eval).

## Official pretrained baseline — COMPLETE (2026-07-22 re-run)

Same eval recipe as seed42 (WikiText-2, seq 512, stride 256, `scale_mode=rtn`, block Linears only).

| Setting | PPL |
|---------|-----|
| Official FP16 | **18.97** |
| Official MXFP4 PTQ blocks (224 Linear) | **51.01** |

- Artifact: `results/pretrained/pretrained_baseline.json` (gitignored)
- Helper: `REMOTE_HOST=<gpu> bash scripts/run_pretrained_baseline.sh`  
  (login streams full `venv` → GPU `/tmp`, then runs `07_eval_pretrained.py`)

### Side-by-side

| Role | High precision | MXFP4 |
|------|----------------|-------|
| Official pretrained | 18.97 | 51.01 (PTQ) |
| From-scratch seed42 | 52.37 (R1) | 67.24 (R2) / 53.73 (R3) |

## Seed 43 — NOT STARTED

```bash
# On GPU node (or after staging venv to /tmp on cold NFS):
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 43 --nproc 4
```
