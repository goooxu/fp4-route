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

## 生成样例（seed=42 · mix73 · R1–R5）

> **权重：** seed42 mix73（EN 70% + ZH 30%）同栈 R1–R5 权重。  
> **规模：** ~360M、从零 ~7B tokens，**不是**产品级模型；事实错误与胡编很常见。  
> **结构：** 共 **5** 例 = **3** 英文 + **2** 中文；每例给出 **R1–R5** 完整输出。  
> - 英文例：模型英文原样；其下 *观感* 为人工点评（**不是**译文）。  
> - 中文例：模型中文续写原样；*观感* 同样为人工点评。  
> **采样：** `temperature=0.8`，`top_p=0.9`，`max_new_tokens=64`，`gen_seed=0`。  
> **复现：** `SEED=42 LANG=en|zh ROUTES=R1,R2,R3,R4,R5 bash scripts/run_generate_samples.sh`

### 例 1（英文）

- **输入：** The history of artificial intelligence began  
- **R1 输出：** The history of artificial intelligence began with the work of the French mathematician Joseph Fourier, who developed the first artificial neural networks in 1869. The first artificial neural network was built by John A. Watson in 1930, which had been developed by Albert Einstein.  
  - *观感：* 把 AI 史错接到傅里叶、1869 神经网络、爱因斯坦等——事实全错，英语尚连贯。  
- **R2 输出：** The history of artificial intelligence began in the early 1950s. It was in the early 1960s that artificial intelligence (AI) started to take on a new role. The word AI started to mean ‘Artificial Intelligence’. As AI’s biggest impact came from the beginning of the 2000  
  - *观感：* 年代像教科书口吻，后文空洞并截断。  
- **R3 输出：** The history of artificial intelligence began with the first artificial intelligence program, Turing's algorithm, which was invented in 1944 by British physicist Sir Henry Simpson. He was working on a system that could be used to answer a computer's problem. Watson, by this time, was creating "intelligent" systems.  
  - *观感：* 提到 Turing 但人物/年份胡编。  
- **R4 输出：** The history of artificial intelligence began with the work of the French mathematician Auguste Comte (1859-1941), who began working on artificial intelligence as a child. He began working on artificial intelligence on the basis of computational methods, the basis of which was known as CHOICE (Advanced Machine Learning).  
  - *观感：* 孔德被安上 AI 生涯与虚构缩写 CHOICE，风格接近 R1 的「名人乱点」。  
- **R5 输出：** The history of artificial intelligence began with the development of the first intelligent machines, the silicon-based computers of the 1950s. In the 1960s, computers were built to handle repetitive tasks like typing and typing. However, because of the complexity of computer algorithms, computers were slow to learn and were often unable to  
  - *观感：* 从 1950 年代计算机切入，略空但比 R2/R4 少胡编人名。  

### 例 2（英文）

- **输入：** Photosynthesis is the process by which  
- **R1 输出：** Photosynthesis is the process by which plants convert carbon dioxide, water, and sunlight into energy. Plants produce the energy they need through photosynthesis. The process of photosynthesis is the process by which plants use sunlight to make food. There are four types of photosynthesis: - Enzymes  
  - *观感：* 开头方向正确，后文「类型/酶」划分混乱。  
- **R2 输出：** Photosynthesis is the process by which plants convert food into energy and nitrogen. The process of photosynthesis involves the conversion of sunlight energy to chemical energy (see Organic Synthesis). Dissolved Oxygen (DO) The oxygen required for photosynthesis is derived from the breakdown of sugar to carbon dioxide.  
  - *观感：* 概念反了（食物→能量/氮），并混入词条腔。  
- **R3 输出：** Photosynthesis is the process by which plants convert carbon dioxide and water into sugars, a process that can be done with little water. In the absence of light, photosynthesis is reversed. In order for photosynthesis to occur, the sugar molecules that are the primary raw materials must be produced.  
  - *观感：* 有 CO₂+水→糖 骨架，「几乎不需要水/无光逆转」不严谨。  
- **R4 输出：** Photosynthesis is the process by which plants use carbon dioxide and water to make food. The process involves the release of water, carbon dioxide, and light energy. The process takes place in the green house, which is the outer atmosphere of the Earth.  
  - *观感：* 前半尚可，后半 greenhouse/大气层表述离谱。  
- **R5 输出：** Photosynthesis is the process by which plant cells use light energy to make water and carbon dioxide. It is a direct process in which plants convert carbon dioxide from the air into a form that can be used by other plants. Photosynthesis is also known as photosynthesis.  
  - *观感：* 方向写反（光造水与 CO₂），并同义反复。  

### 例 3（英文）

- **输入：** The capital of France is  
- **R1 输出：** The capital of France is Paris, the second largest city, and one of the most powerful cities in Europe. It is located on the southern end of the Alps. It is the second largest city in France after Paris. The capital of France is also the capital of France, the second-largest country in the world.  
  - *观感：* 巴黎正确；地理与「世界第二大国」错误，并自我重复。  
- **R2 输出：** The capital of France is that of the France of the World. The capital of France is Paris. It is also the capital of the French province of Aquitaine. It is also the capital of the province of Aquitaine. It is the largest city in France. The population of France is 60 million.  
  - *观感：* 能说到 Paris，夹杂空洞重复与行政区错误。  
- **R3 输出：** The capital of France is Paris. The capital of France is Paris. The capital of the United States is New York. The capital of the United Kingdom is Westminster. The capital of the United States is New York. The capital of the United States is Washington. The capital of the United Kingdom is St. Petersburg.  
  - *观感：* 巴黎正确后乱列他国「首都」，错误连篇。  
- **R4 输出：** The capital of France is now the capital of the province of La Loire (present-day Paris). It is also the capital of the world’s largest democracy, with more than 11 million people, and of the world’s 285 largest cities.  
  - *观感：* 硬拗行政区与「最大民主国家」等，事实混乱。  
- **R5 输出：** The capital of France is Paris. The capital of France is Paris. The capital of France is the USA. The capital of France is Paris. The capital of France is Paris.  
  - *观感：* 重复 Paris，并出现「法国首都是美国」的胡说。  

### 例 4（中文）

- **输入：** 人工智能的历史始于  
- **R1 输出：** 人工智能的历史始于1979年，并在十八年前开始走进北京，并在1988年创立了“北京人工智能大会”，  
  - *观感：* 中文通顺，带出北京/大会等实体，时间线不可信。  
- **R2 输出：** 人工智能的历史始于中国，在智能网联汽车领域拥有多年的技术实力和研发经验。在高科技、人工  
  - *观感：* 跳到智能网联宣传腔，与「历史」脱节。  
- **R3 输出：** 人工智能的历史始于传奇。一、关于人工智能的热点及其在内的问题 传统的智能制造技术与科技相结  
  - *观感：* 像网页目录拼接，信息空洞。  
- **R4 输出：** 人工智能的历史始于往年，但就现在来看，其中一个问题就是人工智能的存在。1、人工智能是什么？人工智  
  - *观感：* FAQ/小标题腔，未真正讲历史。  
- **R5 输出：** 人工智能的历史始于2000年，经过从1995年到2003年的日益延续，走过了从互联网到AI的过渡，成为了大  
  - *观感：* 像编年体起笔，年份自相矛盾，句子未完。  

### 例 5（中文）

- **输入：** 很久很久以前，在一个小村庄里，  
- **R1 输出：** 很久很久以前，在一个小村庄里，艾菲亚他们必须在每一集节目中做出令艾菲亚心灵深处无法自拔的艾  
  - *观感：* 有叙事口吻，但「每一集节目」像串台剧本。  
- **R2 输出：** 很久很久以前，在一个小村庄里，张嘉译说了很多遍，那些张嘉译不得不说，自己是哪一个国家的农民。但是这种  
  - *观感：* 突然插入现代人名，童话设定崩坏。  
- **R3 输出：** 很久很久以前，在一个小村庄里，那个人，它总是无法想象，它们的父母是一个孩子，在一个城市里，我做的事情，它总是  
  - *观感：* 人称/指代混乱，推进弱。  
- **R4 输出：** 很久很久以前，在一个小村庄里，被遗弃的善英一家将全部死在他手中。全讯网-大丰收娱乐城-澳门百  
  - *观感：* 暴力设定后直接撞进赌博广告垃圾串。  
- **R5 输出：** 很久很久以前，在一个小村庄里，他遇到了弗雷迪，看他的「栋笃笑」绝对是一种享受。国际奥委会主席罗格先生上  
  - *观感：* 跳跃到港式喜剧/奥运人物，叙事拼贴感强。  

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
