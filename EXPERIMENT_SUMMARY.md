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

## 2. 质量（WikiText-2 PPL）

用**新代码**重训后填表（旧软件 FQ 结果仅归档）：

| 路线 | Train | Infer | PPL |
|------|-------|-------|-----|
| R1 | BF16 | FP16 | *TBD* |
| R2 | BF16 | TE NVFP4 | *TBD* |
| R3 | TE NVFP4 | TE NVFP4 | *TBD* |

```bash
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 42 --nproc 4
```

## 3. 性能（tokens/s）— GB200 · NGC 26.06 · TE 2.16

对应关系：bench `bf16` ≈ R1 算力；`te_nvfp4` ≈ R2/R3 推理侧硬件路径（R3 另含 NVFP4 训练）。

### 3.1 1×GPU 打满显存（best）

| Backend | Phase | bs | tokens/s | vs bf16 |
|---------|-------|---:|---------:|--------:|
| bf16 (R1 向) | train | 192 | **173578** | 1.00× |
| bf16 | infer | 192 | **532828** | 1.00× |
| te_nvfp4 (R2/R3 向) | train | 192 | **156443** | 0.90× |
| te_nvfp4 | infer | 192 | **445168** | 0.84× |

### 3.2 4×GPU DDP train（best）

| Backend | bs/gpu | tokens/s | vs bf16 |
|---------|-------:|---------:|--------:|
| bf16 | 128 | **669799** | 1.00× |
| te_nvfp4 | 112 | **587034** | 0.88× |

## 4. 归档：软件 fake-quant（已删除）

| 历史设置 | PPL |
|----------|-----|
| 官方 FP16 / 软件 PTQ | 18.97 / 51.01 |
| 从零 R1 / R2 / R3（软件 FQ） | 52.37 / 67.24 / 53.73 |

## 5. 数据

- Train: FineWeb-Edu `sample-10BT`  
- Eval: WikiText-2  
- `data/`、`checkpoints/`、`results/` 不入 git  
