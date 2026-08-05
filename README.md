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

## 生成样例（seed=42，完整输入 / 输出 + 中文翻译）

> 下列生成样例来自**较早纯英文训练栈**的 R1–R3 权重，用于定性观感；**不要**与上表 mix73 PPL 直接对应。  
> 模型约 360M、从零 ~7B tokens，**不是**产品级模型。  
> 英文为模型原样输出；**中文为人工翻译**（不是中文续写）。  
> 采样：`temperature=0.8`，`top_p=0.9`，`max_new_tokens=64`，`gen_seed=0`。  
> 复现：`SEED=42 LANG=en bash scripts/run_generate_samples.sh`。

### 例 1

- **输入：** The history of artificial intelligence began  
  - *中文：* 人工智能的历史始于  
- **R1 输出：** The history of artificial intelligence began when the 1960s. The first artificial intelligence was the 1963 IBM 5480 computer. It was a single-machine algorithm, which was able to make predictions and then produce a plan. However, the 1960s saw a shift in the field  
  - *中文：* 人工智能的历史始于 1960 年代。第一批人工智能是 1963 年的 IBM 5480 计算机。它是一种单机算法，能够做出预测并生成计划。不过，1960 年代该领域发生了转变……  
- **R2 输出：** The history of artificial intelligence began in the late 1960s with the arrival of the computer scientists and the computer scientists themselves. In the early 1970s, computers were presented as a single unit and were very useful in certain situations. They were called the Nokia. The name was chosen because the computer scientists,  
  - *中文：* 人工智能的历史始于 1960 年代末，随着计算机科学家的出现……1970 年代初计算机被当作一个整体单元展示……它们被称为 Nokia……  
- **R3 输出：** The history of artificial intelligence began in the early 1990s, when there was a new form of machine learning that allowed machines to learn quickly. This process allowed computers to learn by using automatic processes, so that machines could learn quickly. These days, there are still a lot of applications in the computer industry that can use machine learning,  
  - *中文：* 人工智能的历史始于 1990 年代初，当时出现了一种新的机器学习形式，使机器能够快速学习……如今计算机行业仍有许多应用可以使用机器学习……  

### 例 2

- **输入：** In mathematics, a prime number is  
  - *中文：* 在数学中，质数是  
- **R1 输出：** In mathematics, a prime number is a set of numbers that can be divided by itself to determine the product of two numbers. - The first number is the prime number. - The second number is the sum of the two numbers. - The sum of the two numbers is the number of the prime numbers. - The second number is the  
  - *中文：* 质数被说成「可用自身相除来确定两数乘积的一组数」，随后条目自相矛盾、定义混乱。  
- **R2 输出：** In mathematics, a prime number is any number which is not a prime number. In English, a prime number is any number which is not a prime number. In mathematics, a prime number is any number which is not a prime number. In the modern world, a prime number is a number that is not a prime number. In  
  - *中文：* 质数是任何「不是质数」的数……反复自相矛盾。  
- **R3 输出：** In mathematics, a prime number is a number of numbers that have the same as the first or second and is therefore called a prime number. In the case of the prime number, a prime number is a number that has the same as the first or second and is thus called a prime number. The prime number is an important part of math. If  
  - *中文：* 循环、含糊的质数描述，并称质数是数学的重要部分。  

### 例 3

- **输入：** Photosynthesis is the process by which  
  - *中文：* 光合作用是……的过程  
- **R1 输出：** Photosynthesis is the process by which cells in the body break down carbohydrates. It is an important part of the body's energy production. The brain, liver and kidneys are the main sources of energy for the body. The body needs food to function well and function properly. The body needs carbohydrates to function properly. The brain is a very important  
  - *中文：* 错把光合作用说成体内细胞分解碳水化合物，并扯到脑、肝、肾产能。  
- **R2 输出：** Photosynthesis is the process by which a plant produces energy from sunlight. Do you want to learn more about photosynthesis? We have a great section on photosynthesis, the process of photosynthesis, and more on photosynthesis. What is photosynthesis? Hydrocyclorapr fixed in water Here are the different types of photosynthesis: 1. Soil-  
  - *中文：* 开头「植物利用阳光产能」尚可，后文变成重复广告腔与乱码词。  
- **R3 输出：** Photosynthesis is the process by which plants and other organisms use light to produce food. It is also the most energy-intensive of all the chemical reactions that happen in the Earth’s atmosphere. Plants use a variety of resources to produce energy, from nitrogen and oxygen to sunlight and carbon dioxide. But, because of a complex chemistry, the amount of light  
  - *中文：* 植物等用光制造食物……相对最贴题，后文仍有夸大与含糊。  

### 例 4

- **输入：** The capital of France is  
  - *中文：* 法国的首都是  
- **R1 输出：** The capital of France is Paris. It is the largest city in France and the sixth largest in the world. It is located on the southern end of the Alps. It is the capital of the Kingdom of France. The City of Paris is also known as the City of the Holy Roman Empire or the City of the Holy Roman Empire. It  
  - *中文：* 巴黎正确；后文把位置说成阿尔卑斯南端、并扯到神圣罗马帝国，错误。  
- **R2 输出：** The capital of France is Paris. The capital of the Italian and French nations is Paris. The city of Paris has the most beautiful buildings of any city in the world. It is the city of the French people. The French flag was chosen on 10 January 1802, at the height of the French Revolution. The  
  - *中文：* 巴黎正确；随后国旗日期与大革命时间线混乱。  
- **R3 输出：** The capital of France is Paris. The capital of the United States is New York. The capital of Switzerland is Berlin. The capital of Germany is Warsaw. The capital of Germany is Prague. The capital of Greece is Athens. The capital of Japan is Kyoto. The capital of Israel is Baghdad. The capital of Turkey is Cairo. The capital of  
  - *中文：* 除巴黎外，美/瑞/德/日/以/土等首都大多错误。  

### 例 5

- **输入：** Once upon a time, in a small village,  
  - *中文：* 很久很久以前，在一个小村庄里，  
- **R1 输出：** Once upon a time, in a small village, a little girl of the village was asked by a farmer to put a tree in the field. She told the farmer, “It is the first time that you have put an object in the field.” The farmer replied, “The tree is there, but you can see it by the way it is put in the field  
  - *中文：* 小女孩与农夫把树放进田里的对话故事，语句尚通顺。  
- **R2 输出：** Once upon a time, in a small village, in a town, in a community, or in a town, you would often find that the mother was a lady who had been married to a man and had married a woman. That’s the reason why we do have the following question: If the woman had been married to a man, who had the right to  
  - *中文：* 叙事跳到「母亲婚姻与权利」的问题，逻辑跳跃。  
- **R3 输出：** Once upon a time, in a small village, in a little village, a group of residents who had fled to the countryside as their families were already in their own houses, with their village’s houses, with their boys’ houses, with their boys’ houses, and their father’s houses, they gathered together in the same place, where there was a group  
  - *中文：* 逃到乡下的居民聚集……大量重复「房子」，叙事不清。  

---

## Quick start

```bash
# Smoke 135M
NPROC=1 bash scripts/06_run_all.sh --config configs/smoke_135m.yaml --seed 42 --nproc 1

# Mainline 360M
NPROC=4 SEED=42 bash scripts/run_full_r123.sh
NPROC=4 SEED=43 bash scripts/run_resume_r123.sh

# Generation samples (English)
SEED=42 LANG=en MAX_NEW=64 bash scripts/run_generate_samples.sh

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
