# MXFP4 三条训推路线对比实验总结（TODO 全量重做）

> **范围声明**：本实验只测**推理质量**（WikiText-2 PPL）；吞吐 / 延迟不在范围内。  
> 量化主口径：transformer **block 内 Linear**（约 210/224 层）；`embed_tokens` + `lm_head` 保持高精度。  
> MXFP4 实现为 **PyTorch fake-quant（STE）**，非 Transformer Engine / 硬件 MXFP4 GEMM。

## 1. 需求

在 NVIDIA Blackwell（GB200）上对比三条路线：

| 路线 | 训练 | 推理 |
|------|------|------|
| R1 | BF16 从零 | FP16（真 fp16 前向，无 bf16 autocast） |
| R2 | 同 R1 ckpt → block PTQ | MXFP4 W4A4 |
| R3 | MXFP4 fake-quant（STE）从零 | MXFP4 W4A4 |

训练语料：**FineWeb-Edu**（`sample-10BT`）；评测：**WikiText-2**。主线结构：SmolLM2-360M；冒烟：SmolLM2-135M。

## 2. 怎么做的

### 正确性（P0）

- 修复 tie / `lm_head`：默认跳过 `embed_tokens` + `lm_head`；`include_lm_head` 时先 untie 再量化。
- Eval 断言：tied 权重共享 + PTQ 权重落在 MXFP4 网格（抽样加速；RTN 网格检查向量化）。
- R1 名实一致：关闭 autocast，真 FP16 前向。

### 数据与训练（P1）

- FineWeb-Edu streaming → **int32 memmap `.npy` token cache**（避免 7B tokens 时 list + `torch.cat` OOM）。
- Val loss 写入 jsonl；可选 `torchrun` DDP（冒烟 / 主线默认 4×GPU）。
- 修复 DDP 与 `CUDA_VISIBLE_DEVICES` 单卡冲突（`NPROC>1` 时暴露 `0,1,2,3`）。
- Seed：冒烟 42；主线 42 / 43（`isolate_seed_paths`）。
- 断点续训：`save_every`（主线 1000）写 `checkpoints/**/resume/`；`resume: true`；SIGTERM/SIGINT 再存一版。
- 换机：`scripts/sync_artifacts.sh`（`REMOTE` / `REMOTE_DIR` 经 env，不写死 IP）+ `scripts/resume_main.sh`。

### 扩展（P2）与工程（P3）

- R3' QAT-from-pretrained；lm_head / E8M0 `rtn`|`floor` 消融；`07_eval_pretrained.py`；`tests/test_quant.py`。

### 速度说明（非目标；工程侧已优化）

- 主线 4×GB200、batch 64、DDP：BF16 ~**4.2 it/s**；MXFP4 FQ ~**1.24 it/s**（无 compile 续跑）或更高（compile 开启时稳态约 3 it/s）。  
- FQ 仍是软件仿真（group=32 quant + `F.linear`），慢于 BF16 属预期，**不代表**硬件 MXFP4 吞吐。  
- 换机冷启动：将 `init` / `resume` / FineWeb `.npy` 暂存到节点 `/tmp` 可避免 NFS thrash（见 `scripts/stage_local_and_resume.sh`）。

## 3. 结论

### 3.1 冒烟：SmolLM2-135M 结构，FineWeb ~200M tokens（从零）

| 路线 | WikiText-2 PPL | Val loss |
|------|----------------|----------|
| R1 | **450.6** | 4.82 |
| R2 | 529.3 | 4.82 |
| R3 | 457.6 | 4.88 |

- 欠拟合区间：train/val loss ~4.8–4.9，绝对 PPL 仍高。
- **R2 相对 R1 变差约 17%**（450→529）：修正 tie 口径后，**不能**再说「block PTQ 几乎无损」。
- R3 ≈ R1（FQ 训推与 BF16 接近）。

### 3.2 预训练上限（官方 SmolLM2-360M 权重）

**2026-07-22 重跑确认**（`scripts/07_eval_pretrained.py` + `scripts/run_pretrained_baseline.sh`；与 seed42 同口径：WikiText-2、seq 512、stride 256、`scale_mode=rtn`、block Linear 仅）：

| 设置 | PPL | #Linear | 备注 |
|------|-----|---------|------|
| FP16 | **18.97** | — | 旧文档写 19.0；重跑一致 |
| PTQ blocks（跳过 lm_head） | **51.01** | 224 | 旧文档写 51.0；重跑一致 |
| PTQ + untied lm_head | **84.4** | 225 | 更早消融，本次未重跑 |
| R3' QAT 500 步 | **50.6** | 224 | 更早；相对 PTQ 仅微弱恢复 |
| E8M0 `rtn` / `floor` | 51.0 / ~1.7e4 | 224 | **默认 rtn** |

产物：`results/pretrained/pretrained_baseline.json`（gitignored；数字以上表为准）。

- 旧「全量 PTQ ≈84」几乎全是 **lm_head 被量化** 的代价；默认口径 51 已明显更好。

### 3.3 主线 360M × seed 42（FineWeb ~7B，2026-07 重跑）

配置：SmolLM2-360M arch 从零；seq 512；4×GB200 DDP；batch 64；grad_accum 1；tokens/step 131072；**53406** steps；seed 42；block Linear MXFP4（embed+lm_head 高精度）；`scale_mode=rtn`。

| 路线 | Train | Infer | WikiText-2 PPL | Train loss | Best val loss |
|------|-------|-------|----------------|------------|---------------|
| R1 | BF16 | FP16 | **52.37** | 0.748 | **3.24** |
| R2 | 同 R1 → block PTQ | MXFP4 W4A4 | **67.24** | 0.748 | 3.24 |
| R3 | MXFP4 FQ | MXFP4 W4A4 | **53.73** | 0.762 | **3.30** |

解读：

1. **R3 ≈ R1**（PPL 53.7 vs 52.4，相对仅 ~2.6%）：从零 FQ 训推可接近 BF16 质量。  
2. **R2 有明显 PTQ 代价**（52.4 → 67.2，约 **+28%** PPL）：block PTQ 在充分训练后仍不可忽略。  
3. 相对冒烟（PPL~450）：7B tokens 后进入可用收敛区，结论与「欠拟合时 R1≈R3、R2 更差」一致，且 R2 差距在收敛区仍清晰。  
4. 产物：`results/main_360m/seed_42/metrics.json`、`summary.md`；权重在 `checkpoints/seed_42/`。

### 3.3.1 官方预训练 vs 从零 seed42（同评测口径）

| 角色 | 高精度推理 | MXFP4（block） |
|------|------------|----------------|
| **官方预训练** | FP16 **18.97** | PTQ **51.01** |
| **从零 7B seed42** | R1 **52.37** | R2 PTQ **67.24** / R3 FQ **53.73** |

1. 官方 FP16 远强于从零 R1（数据量 / 配方 / 训练预算不同，属预期）。  
2. MXFP4 推理下：从零 R3（53.7）≈ 官方 PTQ（51.0）；从零 R2（67）更差。  
3. PTQ 相对代价：官方 19→51（基线更强，绝对掉点大）；从零 R1→R2 约 +28% PPL。  
4. 实现仍是 **软件 fake-quant + `F.linear`**，非硬件 MXFP4 GEMM。

| 项 | 状态 |
|----|------|
| Seed 42 全流程（BF16→FQ→PTQ→eval） | **完成** |
| 官方预训练基线重跑（FP16 + block PTQ） | **完成**（2026-07-22） |
| Seed 43 全流程 | **未开始** |
| 吞吐 bench（bf16 / sw_fq） | **完成**（见 §3.5） |
| 硬件 TE FP4 吞吐 | **阻塞**（见 §3.5.2） |

### 3.4 总括

1. **从零充分训练（7B tokens）**：R1≈R3，R2 有可见 PTQ 质量损失。  
2. **预训练收敛区（官方权重）**：质量由权重决定；block PTQ 损失可控（19→51），含 lm_head 则大幅变差（→84）；**2026-07 重跑复核**。  
3. 短 QAT 未能显著挽回 PTQ 损失。  
4. **双轨**：质量用软件 FQ 公平比 PPL；性能用 tokens/s（见 §3.5）。软件 FQ 训练吞吐约为 BF16 的 ~1/4，与「仿真 quant 非硬件 GEMM」一致。  
5. 冷 NFS GPU 节点：登录机 tar 流完整 `venv`→节点 `/tmp`（`scripts/stage_to_gpu.sh` / `run_pretrained_baseline.sh`）。

### 3.5 性能轨（吞吐）— 2026-07-22 GB200 单卡

协议：SmolLM2-360M arch、`seq=512`、warmup=5、measure=20、`from_config` 随机权重（吞吐不依赖收敛）、单卡。  
脚本：`scripts/12_bench_throughput.py`；原始 JSON 在 `results/perf/`（gitignored）。

| Backend | Phase | batch | tokens/s | ms/step | peak mem |
|---------|-------|------:|---------:|--------:|---------:|
| **bf16** | train | 64 | **151907** | 216 | 60 GB |
| **bf16** | infer | 64 | **473741** | 69 | 44 GB |
| **sw_fq**（软件 MXFP4 FQ） | train | 64 | **41664** | 786 | 68 GB |
| **sw_fq** | infer | 32 | **46613** | 351 | 103 GB |

说明：

- `sw_fq` infer 在 batch=64 时 OOM（activation quant 中间态）；表中用 batch=32。  
- **sw_fq train ≈ bf16 train × 0.27**；软件路径明显不是硬件 MXFP4 上界。  
- 标签：`sw_fq` = OCP 风格 E2M1 group=32 E8M0 + `F.linear`；**不是** Tensor Core FP4 GEMM。

#### 3.5.1 硬件 FP4 / Transformer Engine 状态

| 项 | 结果 |
|----|------|
| GPU | NVIDIA GB200，capability (10, 0) |
| `transformer_engine_cu12` 2.16.0 | aarch64 wheel **可装** |
| `transformer_engine_torch` | **无 aarch64 预编译 wheel**；源码编译失败（无系统 nvcc / 构建错误） |
| 硬件 NVFP4 吞吐 | **本环境暂不可测** |

后续解锁：提供 aarch64 的 TE torch wheel，或安装完整 CUDA toolkit 后成功编译 `transformer_engine_torch`，再跑 `--backend te_fp4`。代码侧已预留 `mxfp4_lib/te_linear.py` + `--backend te_fp4`。

## 4. 数据集位置（保留，不入 git）

HF 源（训练 / 评测）：

- Train: `HuggingFaceFW/fineweb-edu`，config `sample-10BT`
- Eval: `Salesforce/wikitext`，config `wikitext-2-raw-v1`
- Tokenizer / 结构：`HuggingFaceTB/SmolLM2-360M`（冒烟为 135M）

**本地 durable token / 评测缓存**（仓库根下 `data/`，已被 `.gitignore` 忽略；不入 git）：

```
data/
  fineweb_edu/
    train_tok200000000_seed42.{pt,json}      # 冒烟 ~200M
    train_tok7000000000_seed42.{npy,json}    # 主线 ~7B int32 memmap
    val_tok1000000_seed10042.{pt,json}       # FineWeb val（seed+10000）
  wikitext2/                                 # 若存在：HF datasets 落盘；缺失时 prepare 会重下
```

绝对路径取决于本机 checkout 位置，例如将仓库放在 scratch 下时为  
`<scratch>/cursor/mxfp4_route_compare/data/`。

说明：

- 7B 主缓存为 **`train_tok7000000000_seed42.npy`（约 27G）**；加载走 memmap，勿再拼成巨大 list。
- WikiText-2 在最后一次清场前若未回拉成功，需在新机器上 `01_prepare_data.py` 重新下载（FineWeb token cache 仍可复用）。
- **不要**把 `data/`、权重、checkpoint 提交进 git。

## 5. 仓库与清理声明

本次提交后仓库 **仅保留代码与文档**（`mxfp4_lib/`、`scripts/`、`configs/`、`tests/`、README / 本总结）。

已删除或不入 git：

- `checkpoints/`、训练 resume、模型权重
- `results/` 原始 jsonl / metrics 文件（结论已写入上文）
- `artifact_backup/` 中的 checkpoint / 日志（FineWeb 已迁到 `data/fineweb_edu/`）
- 远程测试机工作目录中的训练产物（节点回收后可能已不可达）

换机续跑：自备 `REMOTE=user@host`，`bash scripts/00_setup_remote.sh`，将本地 `data/` 同步到远程后 `NPROC=4 bash scripts/resume_main.sh configs/main_360m.yaml 42`。
