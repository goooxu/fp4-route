# Generation samples (model seed=42, gen_seed=0, max_new_tokens=64)

Sampling: `do_sample=True`, temperature=0.8, top_p=0.9.  
English model outputs only; Chinese under each block is a **human translation** for reading (not model generation).

See also the homepage section in [`../README.md`](../README.md).

## 例 1

**输入 / Input**

```text
The history of artificial intelligence began
```

中文：人工智能的历史始于

**R1 完整输出**

```text
The history of artificial intelligence began when the 1960s. The first artificial intelligence was the 1963 IBM 5480 computer. It was a single-machine algorithm, which was able to make predictions and then produce a plan.
However, the 1960s saw a shift in the field
```

中文翻译：人工智能的历史始于 1960 年代。第一批人工智能是 1963 年的 IBM 5480 计算机。它是一种单机算法，能够做出预测并生成计划。不过，1960 年代该领域发生了转变……

**R2 完整输出**

```text
The history of artificial intelligence began in the late 1960s with the arrival of the computer scientists and the computer scientists themselves. In the early 1970s, computers were presented as a single unit and were very useful in certain situations. They were called the Nokia. The name was chosen because the computer scientists,
```

中文翻译：人工智能的历史始于 1960 年代末……它们被称为 Nokia……

**R3 完整输出**

```text
The history of artificial intelligence began in the early 1990s, when there was a new form of machine learning that allowed machines to learn quickly. This process allowed computers to learn by using automatic processes, so that machines could learn quickly. These days, there are still a lot of applications in the computer industry that can use machine learning,
```

中文翻译：人工智能的历史始于 1990 年代初……一种新的机器学习形式……

## 例 2

**输入 / Input**

```text
In mathematics, a prime number is
```

中文：在数学中，质数是

**R1 完整输出**

```text
In mathematics, a prime number is a set of numbers that can be divided by itself to determine the product of two numbers.
- The first number is the prime number.
- The second number is the sum of the two numbers.
- The sum of the two numbers is the number of the prime numbers.
- The second number is the
```

中文翻译：……定义混乱的“质数”说明……

**R2 完整输出**

```text
In mathematics, a prime number is any number which is not a prime number.
In English, a prime number is any number which is not a prime number.
In mathematics, a prime number is any number which is not a prime number.
In the modern world, a prime number is a number that is not a prime number.
In
```

中文翻译：质数是任何不是质数的数……（自相矛盾并重复）

**R3 完整输出**

```text
In mathematics, a prime number is a number of numbers that have the same as the first or second and is therefore called a prime number. In the case of the prime number, a prime number is a number that has the same as the first or second and is thus called a prime number. The prime number is an important part of math.
If
```

中文翻译：……循环、含糊的质数描述……

## 例 3

**输入 / Input**

```text
Photosynthesis is the process by which
```

中文：光合作用是……的过程

**R1 完整输出**

```text
Photosynthesis is the process by which cells in the body break down carbohydrates. It is an important part of the body's energy production. The brain, liver and kidneys are the main sources of energy for the body.
The body needs food to function well and function properly.
The body needs carbohydrates to function properly.
The brain is a very important
```

中文翻译：把光合作用说成体内分解碳水……（事实错误）

**R2 完整输出**

```text
Photosynthesis is the process by which a plant produces energy from sunlight.
Do you want to learn more about photosynthesis? We have a great section on photosynthesis, the process of photosynthesis, and more on photosynthesis.
What is photosynthesis?
Hydrocyclorapr fixed in water
Here are the different types of photosynthesis:
1. Soil-
```

中文翻译：开头“植物利用阳光产能”尚可，后文胡编。

**R3 完整输出**

```text
Photosynthesis is the process by which plants and other organisms use light to produce food. It is also the most energy-intensive of all the chemical reactions that happen in the Earth’s atmosphere. Plants use a variety of resources to produce energy, from nitrogen and oxygen to sunlight and carbon dioxide. But, because of a complex chemistry, the amount of light
```

中文翻译：植物等用光制造食物……（相对最贴题）

## 例 4

**输入 / Input**

```text
The capital of France is
```

中文：法国的首都是

**R1 完整输出**

```text
The capital of France is Paris. It is the largest city in France and the sixth largest in the world. It is located on the southern end of the Alps. It is the capital of the Kingdom of France.
The City of Paris is also known as the City of the Holy Roman Empire or the City of the Holy Roman Empire. It
```

中文翻译：巴黎正确；后文地理/历史错误。

**R2 完整输出**

```text
The capital of France is Paris. The capital of the Italian and French nations is Paris. The city of Paris has the most beautiful buildings of any city in the world. It is the city of the French people.
The French flag was chosen on 10 January 1802, at the height of the French Revolution. The
```

中文翻译：巴黎正确；国旗年份等混乱。

**R3 完整输出**

```text
The capital of France is Paris. The capital of the United States is New York. The capital of Switzerland is Berlin. The capital of Germany is Warsaw. The capital of Germany is Prague. The capital of Greece is Athens. The capital of Japan is Kyoto. The capital of Israel is Baghdad. The capital of Turkey is Cairo. The capital of
```

中文翻译：除巴黎外多为错误首都。

## 例 5

**输入 / Input**

```text
Once upon a time, in a small village,
```

中文：很久很久以前，在一个小村庄里，

**R1 完整输出**

```text
Once upon a time, in a small village, a little girl of the village was asked by a farmer to put a tree in the field. She told the farmer, “It is the first time that you have put an object in the field.” The farmer replied, “The tree is there, but you can see it by the way it is put in the field
```

中文翻译：小女孩与农夫把树放进田里的对话故事……

**R2 完整输出**

```text
Once upon a time, in a small village, in a town, in a community, or in a town, you would often find that the mother was a lady who had been married to a man and had married a woman. That’s the reason why we do have the following question: If the woman had been married to a man, who had the right to
```

中文翻译：叙事跳跃到婚姻权利问题……

**R3 完整输出**

```text
Once upon a time, in a small village, in a little village, a group of residents who had fled to the countryside as their families were already in their own houses, with their village’s houses, with their boys’ houses, with their boys’ houses, and their father’s houses, they gathered together in the same place, where there was a group
```

中文翻译：逃到乡下的居民聚集……（重复、叙事不清）
