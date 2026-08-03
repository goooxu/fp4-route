# R1–R5 路线对比实验总结

> 五条路线：**R1 BF16** · **R2/R3 TE NVFP4** · **R4/R5 TE MXFP8**。  
> 环境：`nvcr.io/nvidia/pytorch:26.07-py3`。  
> 训练语料：总 ~7B tokens，**英文 FineWeb-Edu 70% + 中文 FineWeb-2 (`zho_Hans`) 30%**。

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
编排：`scripts/run_resume_r123.sh`（`IMG=…:26.07-py3`）。

> **注意：** 与历史「纯英文 + 26.06」结果的绝对 PPL **不可直接对比**；本栈内比较 R1–R5。

## 2. 质量（WikiText-2 PPL）

### Seed 42（2026-07-23）

| 路线 | Train | Infer | PPL | vs R1 |
|------|-------|-------|-----|------:|
| R1 | BF16 | BF16 | **51.95** | 1.00× |
| R2 | BF16 | TE NVFP4 | **54.07** | 1.04× |
| R3 | TE NVFP4 | TE NVFP4 | **53.32** | 1.03× |

报告：`results/main_360m/seed_42/full_report.md`

### Seed 43（2026-08-01）

| 路线 | Train | Infer | PPL | vs R1 |
|------|-------|-------|-----|------:|
| R1 | BF16 | BF16 | **37.07** | 1.00× |
| R2 | BF16 | TE NVFP4 | **40.08** | 1.08× |
| R3 | TE NVFP4 | TE NVFP4 | **39.62** | 1.07× |

> R1 PPL/infer 均为 **BF16** 推理重测（与训练 dtype 一致）。

报告：`results/main_360m/seed_43/full_report.md`

### 跨 seed 解读

- 两 seed 均：**R3（NVFP4 训）略好于 R2（BF16 权 + NVFP4 推）**。  
- 相对 R1：PPL 有小幅上升（seed42 ~+1–2，seed43 ~+2.5–3）。  
- 绝对 PPL 对 random seed 敏感（42 vs 43 差距大），**路线相对序一致**。

## 3. 性能（tokens/s）— GB200 · NGC 26.06 · TE · **trained ckpts**

### 3.1 Seed 42 — 1×GPU / 4×GPU

| Route / path | Phase | nGPU | bs | tokens/s | vs R1 |
|--------------|-------|-----:|---:|---------:|------:|
| R1 bf16 | train | 1 | 192 | **174672** | 1.00× |
| R1 bf16 | infer | 1 | 192 | **526460** | 1.00× |
| R3 te_nvfp4 | train | 1 | 192 | **156553** | 0.90× |
| R2 te_nvfp4 | infer | 1 | 192 | **444015** | 0.85× |
| R3 te_nvfp4 | infer | 1 | 192 | **443179** | 0.84× |
| R1 bf16 | train | 4 | 192 | **688153** | 1.00× |
| R3 te_nvfp4 | train | 4 | 192 | **616741** | 0.90× |

稳态 jsonl（4GPU）：BF16 ~**555k**；NVFP4 ~**448k**。

### 3.2 Seed 43 — 1×GPU / 4×GPU（seed-tagged benches）

| Route / path | Phase | nGPU | bs | tokens/s | vs R1 |
|--------------|-------|-----:|---:|---------:|------:|
| R1 bf16 | train | 1 | 160 | **171657** | 1.00× |
| R1 bf16 | infer | 1 | 192 | **526795** | 1.00× |
| R3 te_nvfp4 | train | 1 | 192 | **155668** | 0.91× |
| R2 te_nvfp4 | infer | 1 | 192 | **441062** | 0.85× |
| R3 te_nvfp4 | infer | 1 | 192 | **440909** | 0.84× |
| R1 bf16 | train | 4 | 192 | **681069** | 1.00× |
| R3 te_nvfp4 | train | 4 | 192 | **602222** | 0.88× |

稳态 jsonl（4GPU）：BF16 ~**573k**；NVFP4 ~**439k**。

**吞吐结论：** NVFP4 训练约 R1 的 **0.85–0.90×**；NVFP4 推理约 R1 BF16 的 **0.84–0.85×**（两 seed 一致）。

## 4. 数据与工程

- Train: FineWeb-Edu `sample-10BT`（seed 相关 token 缓存）  
- Eval: WikiText-2  
- `data/`、`checkpoints/`、`results/`、`logs/`、`snapshots/` 不入 git  
- 断点：`checkpoints/seed_<N>/ckpt_*/resume/`；机器过期后 `SEED=N bash scripts/run_resume_r123.sh`  
