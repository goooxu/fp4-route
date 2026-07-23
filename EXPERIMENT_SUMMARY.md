# R1 / R2 / R3 路线对比实验总结

> **当前口径（软件 fake-quant 已删除）**  
> 三条路线仍为 **R1 / R2 / R3**，低精度统一为 **Transformer Engine `NVFP4BlockScaling`**（硬件 Tensor Core）。  
> 环境优先：`nvcr.io/nvidia/pytorch:26.06-py3`。  
> TE **没有** OCP MXFP4 recipe；硬件 4bit = **NVFP4**（与旧软件 MXFP4 数值网格不同）。

## 1. 三条路线

| 路线 | 训练 | 推理 | 权重 |
|------|------|------|------|
| **R1** | BF16 从零 | FP16 | `ckpt_bf16` |
| **R2** | 同 R1（共享 BF16 ckpt） | TE NVFP4（block Linear） | `ckpt_bf16` + 推理时 `te.Linear` |
| **R3** | TE NVFP4 从零 | TE NVFP4 | `ckpt_nvfp4` |

Scope：block Linear；`embed` + `lm_head` 高精度。

脚本：`02_train_bf16`（R1/R2）→ `03_train_nvfp4`（R3）→ `05_eval_ppl --routes R1,R2,R3`。

## 2. 质量（WikiText-2 PPL）— seed42 from-scratch · 2026-07-23

| 路线 | Train | Infer | PPL | vs R1 |
|------|-------|-------|-----|------:|
| R1 | BF16 | FP16 | **51.94** | 1.00× |
| R2 | BF16 | TE NVFP4 | **54.07** | 1.04× |
| R3 | TE NVFP4 | TE NVFP4 | **53.32** | 1.03× |

解读：R3（NVFP4 训练）略好于 R2（BF16 权重 + NVFP4 推理）；相对 R1 约 +1.4 / +2.1 PPL。

报告：`results/main_360m/seed_42/full_report.md`  
指标：`results/main_360m/seed_42/metrics.json`

## 3. 性能（tokens/s）— GB200 · NGC 26.06 · TE 2.16 · **trained ckpts**

### 3.1 1×GPU 打满（best，seq=512）

| Route / path | Phase | bs | tokens/s | vs R1 same phase |
|--------------|-------|---:|---------:|-----------------:|
| R1 bf16/fp16 | train | 192 | **174672** | 1.00× |
| R1 fp16 | infer | 192 | **524557** | 1.00× |
| R3 te_nvfp4 | train | 192 | **156553** | 0.90× |
| R2 te_nvfp4 | infer | 192 | **444015** | 0.85× |
| R3 te_nvfp4 | infer | 192 | **443179** | 0.84× |

### 3.2 4×GPU DDP train（best）

| Route | bs/gpu | tokens/s | vs R1 |
|-------|-------:|---------:|------:|
| R1 bf16 | 192 | **688153** | 1.00× |
| R3 te_nvfp4 | 192 | **616741** | 0.90× |

稳态训练日志（jsonl 后 20%）：BF16 ~**555k** tok/s；NVFP4 ~**448k** tok/s（4GPU 实训）。

## 4. 归档：软件 fake-quant（已删除）

| 历史设置 | PPL |
|----------|-----|
| 官方 FP16 / 软件 PTQ | 18.97 / 51.01 |
| 从零 R1 / R2 / R3（软件 FQ） | 52.37 / 67.24 / 53.73 |

## 5. 数据

- Train: FineWeb-Edu `sample-10BT`  
- Eval: WikiText-2  
- `data/`、`checkpoints/`、`results/` 不入 git  
