# BF16 vs TE NVFP4 路线实验总结

> **当前口径（2026-07-22 起）**  
> - **已彻底移除软件 fake-quant**（OCP MXFP4 STE / `Mxfp4Linear` / 软件 PTQ 网格）。  
> - 低精度路径仅 **Transformer Engine + `NVFP4BlockScaling`**（Blackwell Tensor Core）。  
> - 质量：WikiText-2 **PPL**；性能：**tokens/s**（1GPU 打满 / 4GPU DDP）。  
> - 运行环境优先：`nvcr.io/nvidia/pytorch:26.06-py3`。

## 1. 路线

| 路线 | 训练 | 推理 |
|------|------|------|
| BF16 | 从零 BF16 | FP16 前向 |
| NVFP4 | TE NVFP4（block Linear） | TE NVFP4 |

Scope：block Linear；`embed` + `lm_head` 高精度。

## 2. 质量（PPL）— 待用新代码重训后填表

旧「软件 FQ R1/R2/R3」结果**作废为正式结论**（实现已删除）。新表：

| 路线 | WikiText-2 PPL | 备注 |
|------|----------------|------|
| BF16 | *TBD* | `ckpt_bf16` |
| NVFP4 | *TBD* | `ckpt_nvfp4` + TE infer |

流水线：`scripts/06_run_all.sh`（prepare → init → `02_train_bf16` → `03_train_nvfp4` → `05_eval_ppl`）。

## 3. 性能（吞吐）— GB200，NGC 26.06，TE 2.16

协议：SmolLM2-360M、`seq=512`、warmup/measure 短测、`from_config` 随机权重。

### 3.1 1×GPU 打满显存（best tok/s）

| Backend | Phase | best bs | tokens/s | peak mem | vs bf16 |
|---------|-------|--------:|---------:|---------:|--------:|
| bf16 | train | 192 | **173578** | 177 GB | 1.00× |
| bf16 | infer | 192 | **532828** | 131 GB | 1.00× |
| te_nvfp4 | train | 192 | **156443** | 165 GB | **0.90×** |
| te_nvfp4 | infer | 192 | **445168** | 119 GB | **0.84×** |

### 3.2 4×GPU DDP train（best）

| Backend | bs/gpu | global | tokens/s | vs bf16 | vs 1GPU |
|---------|-------:|-------:|---------:|--------:|--------:|
| bf16 | 128 | 512 | **669799** | 1.00× | ~3.9× |
| te_nvfp4 | 112 | 448 | **587034** | **0.88×** | ~3.8× |

说明：360M 上 NVFP4 接近但未稳定超过 BF16；扩展性近线性。

### 3.3 TE 与 MXFP4

- TE **无** `MXFP4*` recipe；硬件 4bit 为 **`NVFP4BlockScaling`**（另有 MXFP8）。  
- **NVFP4 ≠ 旧软件 OCP MXFP4**（scale 块大小/类型与训练配方不同）。

## 4. 归档：软件 fake-quant（已删除，仅供历史对照）

以下数字来自已移除的 STE / 软件 PTQ 代码，**不再维护、不可复现于当前仓库**：

| 设置 | PPL（历史） |
|------|-------------|
| 官方预训练 FP16 | 18.97 |
| 官方软件 PTQ blocks | 51.01 |
| 从零 BF16 R1 | 52.37 |
| 从零软件 PTQ R2 | 67.24 |
| 从零软件 FQ R3 | 53.73 |
| 软件 FQ 训练吞吐（相对 BF16） | ~0.28× |

## 5. 数据与产物

- Train: `HuggingFaceFW/fineweb-edu` `sample-10BT`  
- Eval: `Salesforce/wikitext` `wikitext-2-raw-v1`  
- 本地 `data/`、`checkpoints/`、`results/` **不入 git**  

## 6. 主要命令

```bash
# 全流程（容器内，需 TE）
NPROC=4 bash scripts/06_run_all.sh --config configs/main_360m.yaml --seed 42 --nproc 4

# 吞吐
IMG=nvcr.io/nvidia/pytorch:26.06-py3 bash scripts/run_bench_docker.sh
IMG=nvcr.io/nvidia/pytorch:26.06-py3 NPROC=4 bash scripts/run_bench_max_ddp.sh
```
