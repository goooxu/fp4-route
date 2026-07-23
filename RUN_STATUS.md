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

## Perf track (throughput) — max-batch + 4×DDP (2026-07-22)

**Image:** `nvcr.io/nvidia/pytorch:26.06-py3` · GB200 · seq=512 · TE NVFP4

### 1×GPU best (batch sweep to ~full HBM)

| Backend | Phase | bs | tok/s | mem |
|---------|-------|---:|------:|----:|
| bf16 | train | 192 | **173578** | 177G |
| bf16 | infer | 192 | **532828** | 131G |
| sw_fq | train | 160 | **47571** | 167G |
| sw_fq | infer | 48 | **55518** | 151G |
| te_nvfp4 | train | 192 | **156443** | 165G |
| te_nvfp4 | infer | 192 | **445168** | 119G |

### 4×GPU DDP train (best)

| Backend | bs/gpu | global | tok/s | vs bf16 |
|---------|-------:|-------:|------:|--------:|
| bf16 | 128 | 512 | **669799** | 1.00× |
| sw_fq | 128 | 512 | **185992** | 0.28× |
| te_nvfp4 | 112 | 448 | **587034** | 0.88× |

```bash
# On GPU node:
IMG=nvcr.io/nvidia/pytorch:26.06-py3 NPROC=4 bash scripts/run_bench_max_ddp.sh
```

## Seed 43 — NOT STARTED

```bash
# On GPU node (or after staging venv to /tmp on cold NFS):
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 43 --nproc 4
```
