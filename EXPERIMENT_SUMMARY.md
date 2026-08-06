# R1–R5 路线对比实验总结

> 五条路线：**R1 BF16** · **R2/R3 TE NVFP4** · **R4/R5 TE MXFP8**。  
> 环境：`nvcr.io/nvidia/pytorch:26.07-py3` · 4×GB200 DDP。  
> 训练语料：总 ~7B tokens，**英文 FineWeb-Edu 70% + 中文 FineWeb-2 (`cmn_Hani`) 30%**（`cache_tag=mix73_enzh`）。

## 1. 五条路线

| 路线 | 训练 | 推理 | 权重 |
|------|------|------|------|
| **R1** | BF16 从零 | BF16 | `ckpt_bf16` |
| **R2** | 同 R1 | TE NVFP4 | `ckpt_bf16` |
| **R3** | TE NVFP4 从零 | TE NVFP4 | `ckpt_nvfp4` |
| **R4** | 同 R1 | TE MXFP8 | `ckpt_bf16` |
| **R5** | TE MXFP8 从零 | TE MXFP8 | `ckpt_mxfp8` |

Scope：block Linear；`embed` + `lm_head` 高精度。

脚本：`02_train_bf16` → `03_train_nvfp4 --recipe nvfp4|mxfp8` → `05_eval_ppl --routes R1,R2,R3,R4,R5`。  
编排：`scripts/run_resume_r123.sh`（`IMG=…:26.07-py3`；训练后可用 `SKIP_TRAIN=1` 只跑 PPL/bench）。

> **注意：** 与历史「纯英文 + 26.06」结果的绝对 PPL **不可直接对比**；本栈内比较 R1–R5。

## 2. 质量（WikiText-2 PPL）— mix73 · 26.07

### 2.1 双 seed 对照

| 路线 | Train | Infer | seed42 | seed43 | mean | mean vs R1 |
|------|-------|-------|-------:|-------:|-----:|-----------:|
| R1 | BF16 | BF16 | **43.12** | **42.14** | **42.63** | 1.00× |
| R2 | BF16 | TE NVFP4 | **45.88** | **45.61** | **45.75** | 1.07× |
| R3 | TE NVFP4 | TE NVFP4 | **43.98** | **45.50** | **44.74** | 1.05× |
| R4 | BF16 | TE MXFP8 | **43.19** | **42.12** | **42.66** | 1.00× |
| R5 | TE MXFP8 | TE MXFP8 | **44.00** | **42.81** | **43.41** | 1.02× |

报告：`results/main_360m/seed_{42,43}/metrics.json` · `full_report.md`

### 2.2 训练 loss（~7B tokens 后）

| Backend | seed42 final / best val | seed43 final / best val |
|---------|------------------------:|------------------------:|
| BF16 | 2.660 / 2.665 | 2.624 / 2.675 |
| NVFP4 | 2.685 / 2.721 | 2.667 / 2.726 |
| MXFP8 | 2.634 / 2.671 | 2.599 / 2.679 |

### 2.3 解读

- **R4（BF16 权 + MXFP8 推）** 在两个 seed 上都几乎贴齐 R1（相对差 ≤0.2%），是「推理解耦低精度」里最稳的。  
- **R2（BF16 权 + NVFP4 推）** 质量最差一侧（mean +7% vs R1），与「只在推理时用 NVFP4」的预期一致。  
- **R3** 对 seed 更敏感：seed42 仅 +2% vs R1，seed43 约 +8% 且接近 R2——NVFP4 从零训练的质量增益**不总是**稳定超过 PTQ 式 R2。  
- **R5** 相对 R1 约 +2%（双 seed 一致量级）；seed43 上 R5 更接近 R1。  
- 训练 loss：MXFP8 终损在两 seed 上均最低，但英语 WikiText-2 PPL 仍以 R1/R4 略优——语料 30% 中文，eval 与 train loss 不完全对齐。

### 附录：历史纯英文 + 26.06（不可与上表绝对对比）

| Seed | R1 | R2 | R3 | 备注 |
|------|----|----|-----|------|
| 42 (EN-only) | 51.95 | 54.07 | 53.32 | 旧栈；已归档 `*_legacy_en_2606_*` |
| 43 (EN-only) | 37.07 | 40.08 | 39.62 | 同上 |

## 3. 性能（tokens/s）— GB200 · 26.07 · trained ckpts

### 3.1 Microbench best（seed 42；seed 43 几乎相同）

| Route / path | Phase | nGPU | bs | tokens/s | vs R1 train/infer |
|--------------|-------|-----:|---:|---------:|------------------:|
| R1 bf16 | train | 1 | 192 | **173045** | 1.00× |
| R1 bf16 | infer | 1 | 192 | **526805** | 1.00× |
| R3 te_nvfp4 | train | 1 | 192 | **159256** | 0.92× |
| R2 te_nvfp4 | infer | 1 | 192 | **439125** | 0.83× |
| R3 te_nvfp4 | infer | 1 | 192 | **439085** | 0.83× |
| R5 te_mxfp8 | train | 1 | 192 | **172680** | 1.00× |
| R4 te_mxfp8 | infer | 1 | 192 | **501659** | 0.95× |
| R5 te_mxfp8 | infer | 1 | 192 | **501741** | 0.95× |
| R1 bf16 | train | 4 | 192 | **685250** | 1.00× |
| R3 te_nvfp4 | train | 4 | 160 | **622077** | 0.91× |
| R5 te_mxfp8 | train | 4 | 192 | **685374** | 1.00× |

seed 43 对应 best 与上表偏差通常 **&lt;1%**（见 `results/perf/bench_*_seed43_best.json`）。

### 3.2 实训稳态（jsonl 4GPU 平均）

| Backend | seed42 | seed43 |
|---------|-------:|-------:|
| BF16 | ~580k | ~579k |
| NVFP4 | ~439k | ~464k |
| MXFP8 | ~515k | ~506k |

**吞吐结论（本规模）：**

- MXFP8 训练 4 卡 best 与 BF16 基本持平（~685k）；稳态日志仍略慢于 BF16。  
- NVFP4 训练约 R1 的 **0.90–0.92×**；推理约 **0.83×** R1 BF16 infer。  
- MXFP8 推理约 **0.95×** R1，优于 NVFP4 推理。  
- 360M 规模下 NVFP4 **尚未**稳定超过 BF16 吞吐；MXFP8 更接近持平。

## 4. 数据与工程

- Train：FineWeb-Edu `sample-10BT` 70% + FineWeb-2 `cmn_Hani` 30%（`cache_tag=mix73_enzh`）  
- Eval：WikiText-2（英语）  
- TE 评测 pad：NVFP4 序列长对齐 16，MXFP8 对齐 32（`05_eval_ppl.py`）  
- `data/`、`checkpoints/`、`results/`、`logs/`、`snapshots/` 不入 git  
- 断点：`checkpoints/seed_<N>/ckpt_*/resume/`；`SEED=N bash scripts/run_resume_r123.sh`  
- `write_full_report.py` 只汇总当前 seed 的 bench 文件，避免混入其它 seed 产物  
