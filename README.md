# fp4-route

Compare **three train/infer routes** on a causal LM using **hardware TE NVFP4**.

| Route | Train | Infer |
|-------|-------|-------|
| **R1** | From-scratch BF16 | BF16 |
| **R2** | Same BF16 checkpoint | TE **NVFP4** (block Linears) |
| **R3** | TE **NVFP4** train (block Linears) | TE **NVFP4** |

**Quant scope:** transformer-block `Linear` only; `embed_tokens` + `lm_head` stay high precision.

Architecture: [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) via `from_config` (random init, ~362M).  
Train: FineWeb-Edu ~7B tokens (English). Eval: WikiText-2 **PPL** + throughput + generation samples.

**Requires Transformer Engine (NVFP4)** — use:

```text
nvcr.io/nvidia/pytorch:26.06-py3
```

**中文技术报告：** [`docs/TECHNICAL_REPORT_zh.md`](docs/TECHNICAL_REPORT_zh.md)（SVG 配图见 `docs/figures/`）。  
**结果摘要：** [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) · [`RUN_STATUS.md`](RUN_STATUS.md)。  
Seeds **42** / **43** 主线已跑完。

---

## 生成样例（seed=42，完整输入 / 输出 + 中文翻译）

模型约 360M、英文 FineWeb 从零训练约 7B tokens：**不是**产品级模型。  
下列为 **英文原样生成**；其下 **中文** 为人工翻译，便于阅读（**不是**中文续写模型输出）。

- 采样：`temperature=0.8`, `top_p=0.9`, `max_new_tokens=64`, `gen_seed=0`
- 复现：`SEED=42 LANG=en bash scripts/run_generate_samples.sh`
- 备份：[`docs/generation_samples_seed42.md`](docs/generation_samples_seed42.md)

### 例 1

**输入**

```text
The history of artificial intelligence began
```

> 中文：人工智能的历史始于

**R1 完整输出**

```text
The history of artificial intelligence began when the 1960s. The first artificial intelligence was the 1963 IBM 5480 computer. It was a single-machine algorithm, which was able to make predictions and then produce a plan.
However, the 1960s saw a shift in the field
```

> 中文翻译：人工智能的历史始于 1960 年代。第一批人工智能是 1963 年的 IBM 5480 计算机。它是一种单机算法，能够做出预测并生成计划。不过，1960 年代该领域发生了转变……

**R2 完整输出**

```text
The history of artificial intelligence began in the late 1960s with the arrival of the computer scientists and the computer scientists themselves. In the early 1970s, computers were presented as a single unit and were very useful in certain situations. They were called the Nokia. The name was chosen because the computer scientists,
```

> 中文翻译：人工智能的历史始于 1960 年代末，随着计算机科学家的出现……1970 年代初，计算机被当作一个整体单元展示，在某些情况下很有用。它们被称为 Nokia。这个名字之所以被选中，是因为计算机科学家……

**R3 完整输出**

```text
The history of artificial intelligence began in the early 1990s, when there was a new form of machine learning that allowed machines to learn quickly. This process allowed computers to learn by using automatic processes, so that machines could learn quickly. These days, there are still a lot of applications in the computer industry that can use machine learning,
```

> 中文翻译：人工智能的历史始于 1990 年代初，当时出现了一种新的机器学习形式，使机器能够快速学习。这一过程让计算机通过自动流程学习。如今，计算机行业仍有许多应用可以使用机器学习……

### 例 2

**输入**

```text
In mathematics, a prime number is
```

> 中文：在数学中，质数是

**R1 完整输出**

```text
In mathematics, a prime number is a set of numbers that can be divided by itself to determine the product of two numbers.
- The first number is the prime number.
- The second number is the sum of the two numbers.
- The sum of the two numbers is the number of the prime numbers.
- The second number is the
```

> 中文翻译：在数学中，质数是一组可以用自身相除来确定两个数乘积的数。——第一数是质数。——第二数是两数之和。——两数之和是质数的个数。——第二数是……（定义混乱）

**R2 完整输出**

```text
In mathematics, a prime number is any number which is not a prime number.
In English, a prime number is any number which is not a prime number.
In mathematics, a prime number is any number which is not a prime number.
In the modern world, a prime number is a number that is not a prime number.
In
```

> 中文翻译：在数学中，质数是任何“不是质数”的数。……（自相矛盾并重复）

**R3 完整输出**

```text
In mathematics, a prime number is a number of numbers that have the same as the first or second and is therefore called a prime number. In the case of the prime number, a prime number is a number that has the same as the first or second and is thus called a prime number. The prime number is an important part of math.
If
```

> 中文翻译：在数学中，质数是与第一或第二相同的一堆数，因此被称为质数。……质数是数学的重要部分。如果……（循环、含糊）

### 例 3

**输入**

```text
Photosynthesis is the process by which
```

> 中文：光合作用是……的过程

**R1 完整输出**

```text
Photosynthesis is the process by which cells in the body break down carbohydrates. It is an important part of the body's energy production. The brain, liver and kidneys are the main sources of energy for the body.
The body needs food to function well and function properly.
The body needs carbohydrates to function properly.
The brain is a very important
```

> 中文翻译：光合作用是体内细胞分解碳水化合物的过程。它是身体产能的重要部分。大脑、肝脏和肾脏是身体的主要能量来源……（事实错误：把光合作用说成人体内代谢）

**R2 完整输出**

```text
Photosynthesis is the process by which a plant produces energy from sunlight.
Do you want to learn more about photosynthesis? We have a great section on photosynthesis, the process of photosynthesis, and more on photosynthesis.
What is photosynthesis?
Hydrocyclorapr fixed in water
Here are the different types of photosynthesis:
1. Soil-
```

> 中文翻译：光合作用是植物利用阳光产生能量的过程。你想了解更多吗？……什么是光合作用？Hydrocyclorapr fixed in water……以下是不同类型的光合作用：1. 土壤-……（开头尚可，后文胡编）

**R3 完整输出**

```text
Photosynthesis is the process by which plants and other organisms use light to produce food. It is also the most energy-intensive of all the chemical reactions that happen in the Earth’s atmosphere. Plants use a variety of resources to produce energy, from nitrogen and oxygen to sunlight and carbon dioxide. But, because of a complex chemistry, the amount of light
```

> 中文翻译：光合作用是植物和其他生物利用光制造食物的过程。它也是地球大气中最耗能的化学反应之一。植物用多种资源产能，从氮、氧到阳光与二氧化碳。但由于复杂的化学过程，光的量……（相对最贴题）

### 例 4

**输入**

```text
The capital of France is
```

> 中文：法国的首都是

**R1 完整输出**

```text
The capital of France is Paris. It is the largest city in France and the sixth largest in the world. It is located on the southern end of the Alps. It is the capital of the Kingdom of France.
The City of Paris is also known as the City of the Holy Roman Empire or the City of the Holy Roman Empire. It
```

> 中文翻译：法国的首都是巴黎。它是法国最大、世界第六大城市。它位于阿尔卑斯山脉南端。它是法兰西王国的首都。巴黎市也被称为神圣罗马帝国之城……（前半对，后半地理/历史错误）

**R2 完整输出**

```text
The capital of France is Paris. The capital of the Italian and French nations is Paris. The city of Paris has the most beautiful buildings of any city in the world. It is the city of the French people.
The French flag was chosen on 10 January 1802, at the height of the French Revolution. The
```

> 中文翻译：法国的首都是巴黎。意大利与法国的首都是巴黎。……法国国旗于 1802 年 1 月 10 日选定，正值法国大革命高峰……（事实混乱）

**R3 完整输出**

```text
The capital of France is Paris. The capital of the United States is New York. The capital of Switzerland is Berlin. The capital of Germany is Warsaw. The capital of Germany is Prague. The capital of Greece is Athens. The capital of Japan is Kyoto. The capital of Israel is Baghdad. The capital of Turkey is Cairo. The capital of
```

> 中文翻译：法国的首都是巴黎。美国的首都是纽约。瑞士的首都是柏林。德国的首都是华沙/布拉格。……日本是京都，以色列是巴格达，土耳其是开罗……（除巴黎外多为错误首都）

### 例 5

**输入**

```text
Once upon a time, in a small village,
```

> 中文：很久很久以前，在一个小村庄里，

**R1 完整输出**

```text
Once upon a time, in a small village, a little girl of the village was asked by a farmer to put a tree in the field. She told the farmer, “It is the first time that you have put an object in the field.” The farmer replied, “The tree is there, but you can see it by the way it is put in the field
```

> 中文翻译：很久很久以前，在一个小村庄里，村里的一个小女孩被农夫要求把一棵树放到田里。她对农夫说：“这是你第一次把东西放到田里。”农夫回答：“树在那儿，但你可以从它被放进田里的方式看到……”

**R2 完整输出**

```text
Once upon a time, in a small village, in a town, in a community, or in a town, you would often find that the mother was a lady who had been married to a man and had married a woman. That’s the reason why we do have the following question: If the woman had been married to a man, who had the right to
```

> 中文翻译：很久很久以前，在一个小村庄里，在一个镇里、社区里或镇里，你常常会发现母亲是一位曾嫁给男人又……的女士。这就是我们会有如下问题的原因：如果这个女人曾嫁给一个男人，谁有权……

**R3 完整输出**

```text
Once upon a time, in a small village, in a little village, a group of residents who had fled to the countryside as their families were already in their own houses, with their village’s houses, with their boys’ houses, with their boys’ houses, and their father’s houses, they gathered together in the same place, where there was a group
```

> 中文翻译：很久很久以前，在一个小村庄里，在一个小村庄里，一群居民逃到乡下，因为他们的家人已经在自己的房子里……他们带着村里的房子、男孩们的房子……聚集在同一个地方，那里有一群……（重复、叙事不清）

---

## Quick start

```bash
# Inside NGC PyTorch container:
bash scripts/00_setup_remote.sh   # optional host venv for data prep
source venv/bin/activate          # optional

# Smoke 135M — full R1/R2/R3
NPROC=1 bash scripts/06_run_all.sh --config configs/smoke_135m.yaml --seed 42 --nproc 1

# Mainline 360M from-scratch (full R1/R2/R3 + PPL + bench)
NPROC=4 SEED=42 bash scripts/run_full_r123.sh

# Safe resume after machine reclaim
NPROC=4 SEED=43 bash scripts/run_resume_r123.sh

# Generation samples (English)
SEED=42 LANG=en MAX_NEW=64 bash scripts/run_generate_samples.sh

# R3 HF export only (if TE final save failed but resume exists)
python scripts/04_export_nvfp4_from_resume.py --seed 42

# Throughput
IMG=nvcr.io/nvidia/pytorch:26.06-py3 bash scripts/run_bench_docker.sh
IMG=nvcr.io/nvidia/pytorch:26.06-py3 NPROC=4 bash scripts/run_bench_max_ddp.sh
```

Configs: `configs/smoke_135m.yaml`, `configs/main_360m.yaml`, `configs/bench_360m.yaml`.

## Layout

```
configs/       # smoke / main / bench
mxfp4_lib/     # data, train_loop, te_linear (NVFP4), bench
docs/          # TECHNICAL_REPORT_zh.md, figures/*.svg, generation samples
scripts/
  02_train_bf16.py / 03_train_nvfp4.py / 05_eval_ppl.py
  13_generate_samples.py    # qualitative generation (English)
  run_full_r123.sh / run_resume_r123.sh / run_finish_r123.sh
  12_bench_throughput.py / run_bench_*.sh
```

## Checkpoints

```
checkpoints/seed_<N>/
  init_model/
  ckpt_bf16/      # R1 weights; also used for R2 infer
  ckpt_nvfp4/     # R3 weights (saved as nn.Linear after TE train)
```

## Remote

```bash
python3 scripts/remote_run.py --check
REMOTE_HOST=10.x.x.x bash scripts/stage_to_gpu.sh
```

Do not commit GPU IPs or bulky `data/` / `checkpoints/` / `results/`.
