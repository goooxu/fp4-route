# fp4-route

Compare **five train/infer routes** on a causal LM using hardware TE (**NVFP4** + **MXFP8**).

| Route | Train | Infer |
|-------|-------|-------|
| **R1** | From-scratch BF16 | BF16 |
| **R2** | Same BF16 checkpoint | TE **NVFP4** |
| **R3** | TE **NVFP4** train | TE **NVFP4** |
| **R4** | Same BF16 checkpoint | TE **MXFP8** |
| **R5** | TE **MXFP8** train | TE **MXFP8** |

**Quant scope:** transformer-block `Linear` only; `embed_tokens` + `lm_head` stay high precision.

Architecture: [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) via `from_config` (random init, ~362M).  
Train data: **~7B tokens**, English FineWeb-Edu **70%** + Chinese FineWeb-2 (`cmn_Hani`) **30%**.  
Eval: WikiText-2 **PPL** (English) + throughput + generation samples.

**Requires Transformer Engine** — use `nvcr.io/nvidia/pytorch:26.07-py3`.

**中文技术报告：** [`docs/TECHNICAL_REPORT_zh.md`](docs/TECHNICAL_REPORT_zh.md)（SVG 配图见 `docs/figures/`）。  
**结果摘要：** [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) · [`RUN_STATUS.md`](RUN_STATUS.md)。  
**当前：** seed **42** · mix73 + 26.07 · R1–R5 **已完成**（PPL + bench + report）。旧纯英 26.06 结果在 `*_legacy_en_2606_*`。

### Seed 42 · WikiText-2 PPL（mix73）

| Route | Train | Infer | PPL |
|-------|-------|-------|-----|
| R1 | BF16 | BF16 | **43.12** |
| R2 | BF16 | TE NVFP4 | **45.88** |
| R3 | TE NVFP4 | TE NVFP4 | **43.98** |
| R4 | BF16 | TE MXFP8 | **43.19** |
| R5 | TE MXFP8 | TE MXFP8 | **44.00** |

详情与吞吐：[`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) · `results/main_360m/seed_42/`。

---

## 生成样例（seed=42 · mix73 · R1–R3）

> **权重：** 本页 PPL 同栈的 seed42 mix73 训练权重（EN 70% + ZH 30%）。  
> **规模：** ~360M、从零 ~7B tokens，**不是**产品级模型；事实错误与胡编很常见。  
> **结构：** 共 **5** 例 = **3** 英文 + **2** 中文。  
> - 英文例：模型英文原样输出；*中文* 为**人工翻译**（不是中文续写）。  
> - 中文例：模型**中文续写原样**（已混入中文训练语料）。  
> **采样：** `temperature=0.8`，`top_p=0.9`，`max_new_tokens=64`，`gen_seed=0`。  
> **复现：**  
> `SEED=42 LANG=en MAX_NEW=64 bash scripts/run_generate_samples.sh`  
> `SEED=42 LANG=zh MAX_NEW=64 bash scripts/run_generate_samples.sh`

### 例 1（英文）

- **输入：** The history of artificial intelligence began  
  - *中文：* 人工智能的历史始于  
- **R1 输出：** The history of artificial intelligence began with the work of the French mathematician Joseph Fourier, who developed the first artificial neural networks in 1869. The first artificial neural network was built by John A. Watson in 1930, which had been developed by Albert Einstein. This network was de  
  - *中文：* 把 AI 史错接到傅里叶、1869 年神经网络，以及爱因斯坦/Watson——事实全错，但英语句子较连贯。  
- **R2 输出：** The history of artificial intelligence began in the early 1950s. It was in the early 1960s that artificial intelligence (AI) started to take on a new role. The word AI started to mean ‘Artificial Intelligence’. As AI’s biggest impact came from the beginning of the 2000  
  - *中文：* 年代大致像教科书口吻（1950–60 年代），后文到 2000 年断掉，信息空洞。  
- **R3 输出：** The history of artificial intelligence began with the first artificial intelligence program, Turing's algorithm, which was invented in 1944 by British physicist Sir Henry Simpson. He was working on a system that could be used to answer a computer's problem. Watson, by this time, was creating "intell  
  - *中文：* 提到 Turing 但年份/人物（Henry Simpson 等）胡编，并扯到 Watson。  

### 例 2（英文）

- **输入：** Photosynthesis is the process by which  
  - *中文：* 光合作用是……的过程  
- **R1 输出：** Photosynthesis is the process by which plants convert carbon dioxide, water, and sunlight into energy. Plants produce the energy they need through photosynthesis. The process of photosynthesis is the process by which plants use sunlight to make food. There are four types of photosynthesis: - Enzymes  
  - *中文：* 开头方向正确（植物用 CO₂、水、光），后文类型划分混乱。  
- **R2 输出：** Photosynthesis is the process by which plants convert food into energy and nitrogen. The process of photosynthesis involves the conversion of sunlight energy to chemical energy (see Organic Synthesis). Dissolved Oxygen (DO) The oxygen required for photosynthesis is derived from the breakdown of suga  
  - *中文：* 把光合作用说成「把食物转成能量和氮」，概念错误，并混入溶氧等词条腔。  
- **R3 输出：** Photosynthesis is the process by which plants convert carbon dioxide and water into sugars, a process that can be done with little water. In the absence of light, photosynthesis is reversed. In order for photosynthesis to occur, the sugar molecules that are the primary raw materials must be produced  
  - *中文：* 有「CO₂+水→糖」的正确骨架，但「几乎不需要水」「无光则逆转」等表述不严谨。  

### 例 3（英文）

- **输入：** The capital of France is  
  - *中文：* 法国的首都是  
- **R1 输出：** The capital of France is Paris, the second largest city, and one of the most powerful cities in Europe. It is located on the southern end of the Alps. It is the second largest city in France after Paris. The capital of France is also the capital of France, the second-largest country in the world.  
  - *中文：* 巴黎正确；后文「阿尔卑斯南端」「法国是世界第二大国」等错误，并自我重复。  
- **R2 输出：** The capital of France is that of the France of the World. The capital of France is Paris. It is also the capital of the French province of Aquitaine. It is also the capital of the province of Aquitaine. It is the largest city in France. The population of France is 60 million. It is the capital  
  - *中文：* 能说到 Paris，但前后夹杂空洞重复与行政区错误。  
- **R3 输出：** The capital of France is Paris. The capital of France is Paris. The capital of the United States is New York. The capital of the United Kingdom is Westminster. The capital of the United States is New York. The capital of the United States is Washington. The capital of the United Kingdom is St. Peter  
  - *中文：* 巴黎正确后变成乱列他国「首都」，美/英等多处错误。  

### 例 4（中文）

- **输入：** 人工智能的历史始于  
- **R1 输出：** 人工智能的历史始于1979年，并在十八年前开始走进北京，并在1988年创立了“北京人工智能大会”，  
  - *观感：* 能用中文续写并带出「北京/大会」等实体，但时间线与史实不可信。  
- **R2 输出：** 人工智能的历史始于中国，在智能网联汽车领域拥有多年的技术实力和研发经验。在高科技、人工  
  - *观感：* 直接跳到「智能网联汽车」宣传腔，与「历史始于」脱节。  
- **R3 输出：** 人工智能的历史始于传奇。一、关于人工智能的热点及其在内的问题 传统的智能制造技术与科技相结  
  - *观感：* 像网页目录/小标题拼接，信息空洞。  

### 例 5（中文）

- **输入：** 很久很久以前，在一个小村庄里，  
- **R1 输出：** 很久很久以前，在一个小村庄里，艾菲亚他们必须在每一集节目中做出令艾菲亚心灵深处无法自拔的艾  
  - *观感：* 中文叙事口吻有，但人物名与「每一集节目」像串台剧本，不完整。  
- **R2 输出：** 很久很久以前，在一个小村庄里，张嘉译说了很多遍，那些张嘉译不得不说，自己是哪一个国家的农民。但是这种  
  - *观感：* 突然出现现代人名「张嘉译」，童话设定崩坏。  
- **R3 输出：** 很久很久以前，在一个小村庄里，那个人，它总是无法想象，它们的父母是一个孩子，在一个城市里，我做的事情，它总是  
  - *观感：* 人称/指代混乱，故事推进弱。  

---

## Quick start

```bash
# Smoke 135M
NPROC=1 bash scripts/06_run_all.sh --config configs/smoke_135m.yaml --seed 42 --nproc 1

# Mainline 360M
NPROC=4 SEED=42 bash scripts/run_full_r123.sh
NPROC=4 SEED=43 bash scripts/run_resume_r123.sh

# Generation samples (English / Chinese)
SEED=42 LANG=en MAX_NEW=64 bash scripts/run_generate_samples.sh
SEED=42 LANG=zh MAX_NEW=64 bash scripts/run_generate_samples.sh

# Throughput
IMG=nvcr.io/nvidia/pytorch:26.07-py3 bash scripts/run_bench_docker.sh
```

Configs: `configs/smoke_135m.yaml`, `configs/main_360m.yaml`, `configs/bench_360m.yaml`.

## Layout

```
configs/   mxfp4_lib/   docs/   scripts/
checkpoints/seed_<N>/{init_model,ckpt_bf16,ckpt_nvfp4,ckpt_mxfp8}/
```

## Remote

```bash
python3 scripts/remote_run.py --check
REMOTE_HOST=10.x.x.x bash scripts/stage_to_gpu.sh
```

Do not commit GPU IPs or bulky `data/` / `checkpoints/` / `results/`.
