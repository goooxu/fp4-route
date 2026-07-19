# MXFP4 三条训推路线对比实验总结

## 1. 需求

在 NVIDIA **Blackwell（GB200）** 上，对比低精度（**MXFP4、全模型 W4A4 语义**）相关训推路线的**推理效果差异**，主指标为 WikiText-2 **困惑度（PPL，越低越好）**。

约定的三条路线：

| 路线 | 训练 | 推理 |
|------|------|------|
| R1 | BF16 | FP16 |
| R2 | BF16 训练后的同一 checkpoint → **全层 PTQ** | MXFP4（权重静态 PTQ + 激活动态量化） |
| R3 | 原生 MXFP4 语义训练（fake-quant + STE） | MXFP4 |

约束与演进：

1. 先限定：**Blackwell + 推理 W4A4 + 全模型 MXFP4**。
2. 实验要求在 NVIDIA GB200 测试机上**从头实现并跑通**整条流水线。
3. 一度要求**不使用公开预训练权重、从零随机初始化**；后改为**借用开源小模型结构**（`SmolLM2-360M` 的 config），仍随机初始化。
4. 最后补充：用**官方已训练好的权重**测同一 PPL，作为上限参考。

---

## 2. 怎么做的

### 2.1 环境与工程

- **机器**：NVIDIA GB200（SM 10.0，aarch64）
- **代码仓库**：本仓库（`fp4-route`）
- **栈**：PyTorch（CUDA）+ HuggingFace Transformers / Datasets；自研 OCP 风格 MXFP4（E2M1 + group=32 + E8M0 scale）
- **说明**：R3 为 **fake-quant 训练**（数值语义对齐 MXFP4），非 Transformer Engine 硬件 MXFP4 GEMM

### 2.2 实现要点

1. **MXFP4 库**：`mxfp4_lib/quant.py`、`linear.py`、`replace.py`（全层 `nn.Linear` 替换，含 `lm_head`）
2. **数据**：WikiText-2（`Salesforce/wikitext`）；滑动窗口评 PPL（seq=512，stride=256）
3. **训练**：AdamW、BF16 autocast、cosine LR；R1/R3 共享同一 `init_model/`（同 seed）
4. **R2**：对 BF16 checkpoint 做全层权重 MXFP4 RTN；推理时激活动态 MXFP4
5. **编排**：`scripts/06_run_all.sh`（prepare → init → BF16 训 → MXFP4 训 → PTQ → eval）

### 2.3 实验批次

| 批次 | 模型 | 权重 | 训练步数 |
|------|------|------|----------|
| A | 自建 ~52M Llama | 随机初始化 | 3000 |
| B | 借用 `SmolLM2-360M` 结构（~362M） | 随机初始化（`from_config`） | 3000 |
| C | `HuggingFaceTB/SmolLM2-360M` | **官方预训练权重**（不训练） | —（仅评 PPL / PTQ） |

---

## 3. 结论（结果）

### 3.1 批次 A：自建 52M，从零训练

| 路线 | WikiText-2 PPL | 最终 train loss |
|------|----------------|-----------------|
| R1 | 1367.7 | 0.99 |
| R2 | 1302.5 | 0.99 |
| R3 | **350.8** | 1.81 |

- R1 ≈ R2：全层 MXFP4 PTQ 相对 FP16 几乎不伤测试 PPL。
- R3 明显更好：小数据从零训时，MXFP4 fake-quant 更像**强正则**，减轻过拟合。
- 绝对 PPL 很高（模型小 + 从零 + WikiText-2），宜看相对差距。

### 3.2 批次 B：SmolLM2-360M 结构，仍随机初始化

| 路线 | WikiText-2 PPL | 最终 train loss |
|------|----------------|-----------------|
| R1 | 720.3 | 0.097 |
| R2 | 697.8 | 0.097 |
| R3 | **681.1** | 0.482 |

- 更大结构后，R1/R2 绝对 PPL 低于 52M 批次。
- 三路线**彼此接近**（差距约 6%）：PTQ 仍近乎无损；FQ 训推与 BF16 基线接近。
- 仍远高于「真预训练」水平（见下）。

### 3.3 批次 C：官方预训练权重（上限参考）

| 设置 | WikiText-2 PPL |
|------|----------------|
| 预训练 + FP16 | **≈ 19.0** |
| 预训练 + 全层 MXFP4 PTQ | ≈ 84.4 |

- 官方权重比从零训好约一个数量级以上（~19 vs ~700）。
- 预训练模型上做激进全层 MXFP4 PTQ 会明显掉点（19 → 84），但仍远好于从零训。

### 3.4 总括

1. **推理效果上**：在「同结构、同数据、从零训」设定下，全层 MXFP4 PTQ（R2）相对 FP16（R1）损失很小；MXFP4 FQ 训练（R3）在小模型过拟合场景可更好，在更大结构上则与 R1/R2 接近。
2. **绝对质量**主要由「是否使用大规模预训练权重」决定，而不是这三条低精度路线 alone。
3. **工程上**：已在 GB200 上跑通自研 MXFP4 训/PTQ/评全流程；指标与日志见本仓库 `results/`。

---

## 4. 关键产物路径

| 内容 | 路径 |
|------|------|
| 从零训结果（360M 结构） | `results/summary.md`、`results/metrics.json` |
| 预训练基线 | `results/pretrained_baseline.json` |
| 实验总结 | `EXPERIMENT_SUMMARY.md` |
