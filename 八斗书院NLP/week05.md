# AI大模型应用：NLP与大模型

## 模型类型划分

Transformer 模型主要由两个核心部分构成：编码器（Encoder）和解码器（Decoder）。依托这两个关键组件的不同组合和应用，Transformer 模型发展出三种主流架构：编码（Encoder-Only）大语言模型（bert）、解码（Decoder-Only）大语言模型（GPT）以及编解码（Encoder-Decoder）大语言模型（使用较少）。

![2025-09-07_09-19.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/918929aa-7452-4809-a078-89867b9d4537.jpeg)

![2025-09-07_09-21.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/aa36cdb6-9c76-4eff-859f-440a88621a83.jpeg)

![2025-09-07_09-25.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/9a0eb787-2da2-47fa-b164-db61d4b40ba7.jpeg)

### Sentence-BERT

Sentence-BERT通过对整个句子进行建模，学习句子级别的嵌入表示。Sentence-BERT通过在预训练阶段学习句子级别的嵌入表示，能够更好地捕捉句子的语义信息，使得句子表示更为丰富和有意义。

Sentence-BERT的表示能够捕捉到句子中的语义信息，使得生成的句子嵌入在一些任务中更易于解释和理解。

![2025-09-07_09-32.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/b0348272-e686-4c11-8b16-a6e6f0772625.png)

### 无监督句子编码

mean poolong

max pooling

IDF-Weighted Mean Pooling(基于逆文档频率的均值池化)

SIF

### 解码器模型

![2025-09-07_09-52.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/803f142a-eb3e-47e7-9f24-2f0c5dedf7d9.jpeg)

解码器处理输入的上下文信息，采用自回归生成方式逐步生成输出序列

![2025-09-07_09-54.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/cc5435a1-9f89-4866-9e9c-e841bcdf18cf.jpeg)

![2025-09-07_09-57.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/0ea12569-4e0c-4026-b858-931925140283.jpeg)

### 大模型是新的解决范式

语言模型通常是指能够建模自然语言文本生成概率的模型
从语言建模到任务求解，这是科学思维的一次重要跃升

![图片1.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/94921a69-6ab0-4514-9a15-9786a65d7ee5.jpeg)

过去建模步骤：采集大量数据集、训练模型、对比模型、模型调参、模型部署
现在建模步骤：采集少量数据、对比提示词、大模型版本、部署应用

## 大语言模型

定义：通常是指具有超大规模参数的预训练语言模型
架构：主要为 Transformer解码器架构
训练：预训练（base model）、后训练（instruct model）

![图片2.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/a2c3cdb9-70e1-4a41-9571-c108c56fa247.png)


| 对比方面 | 预训练 (Pre-training)              | 后训练 (Post-training)             |
| -------- | ---------------------------------- | ---------------------------------- |
| 核心目标 | 建立模型基础能力                   | 将基座模型适配到具体应用场景       |
| 数据资源 | 数万亿词元的自然语言文本           | 数十万、数百万到数千万指令数据     |
| 所需算力 | 耗费百卡、千卡甚至万卡算力数月时间 | 耗费数十卡、数百卡数天到数十天时间 |
| 使用方式 | 通常为few-shot提示                 | 可以直接进行zero-shot使用          |

### ChatGPT原理

ChatGPT整体过程可以分为三个阶段：

> 第一个阶段是**基础大模型训练**，该阶段主要完成长距离语言模型的预训练，通过代码预训练使得模型具备代码生成的能力；

> 第二阶段是**指令微调（Instruct Tuning）**，通过给定指令进行微调的方式使得模型具备完成各类任务的能力；

> 第三个阶段是**类人对齐**，加入更多人工提示词，并利用有监督微调并结合基于强化学习的方式，使得模型输出更贴合人类需求。

![图片4.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/7958468e-ce17-4f96-a23e-061f47c4d3f6.jpeg)

#### 指令微调

以BERT为代表的预训练语言模型需要根据任务数据进行微调（Fine-tuning），这种范式可以应用于参数量在几百万到几亿规模的预训练模型。但是针对数十亿甚至是数百亿规模的大模型，针对每个任务都进行微调的计算开销和时间成本几乎都是不可接受的。因此，研究人员们提出了指令微调（Instruction Finetuning）方案，将大量各类型任务，统一为生成式自然语言理解框架，并构造训练语料进行微调。

> 例如，可以将情感倾向分析任务，通过如下指令，将贬义和褒义的分类问题转换到生成式自然语言理解框架：
> For each snippet of text, label the sentiment of the text as positive or negative.Text: this film seems thirsty for reflection, itself taking on adolescent qualities.Label: [positive / negative]

#### 类人对齐

![2025-09-07_10-33.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/9dd2eb8f-4e02-42e0-95a7-8f3ec8155d74.jpeg)

大模型预训练：基于Transformer解码器架构，进行下一个词预测
大模型后训练：
指令微调（Instruction Tuning） ：使用输入与输出配对的指令数据对于模型进行微调
人类对齐（Human Alignment） ：将大语言模型与人类的期望、需求以及价值观对齐

![图片5.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/4b9d55b8-4c6d-4eb1-affe-ed44d36279cf.jpeg)

![图片6.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/46791ad2-668c-46b4-9a4c-96dfe31d08d6.jpeg)

### 扩展定律（解码器模型）

通过扩展参数规模、数据规模和计算算力，大语言模型的能力会出现显著提升

参数规模（𝑁）、数据规模（𝐷） 和计算算力（𝐶）之间的幂律关系

![图片7.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/dadfce45-3c18-4e60-8bb0-8801779fe39a.jpeg)

$$
L(N)=\quad\left(\frac{N_{c}}{N}\right)^{\alpha_{N}},\quad\alpha_{N}\sim0.076,N_{c}\sim8.8\times10^{13}\\L(D)=\quad\left(\frac{D_{c}}{D}\right)^{\alpha_{D}},\:\alpha_{D}\sim0.095,D_{c}\sim5.4\times10^{13}\\L(C)=\quad\left(\frac{C_{c}}{C}\right)^{\alpha_{C}},\quad\alpha_{C}\sim0.050,C_{c}\sim3.1\times10^{8}
$$

![图片8.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/d86663bb-589e-4e3a-a67a-eb08633c6193.jpeg)


### GPT的发展里程

![图片9.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/ba902677-08bf-4428-b192-5dd959d748d3.jpeg)

GPT-1（1.1亿参数） Decode-only Transformer架构 ，预训练后针对特定任务微调
GPT-2 （15亿参数） 将任务形式统一为单词预测，预训练与下游任务一致
GPT-3 模型规模达到1750亿参数，涌现出上下文学习能力
InstructGPT 大语言模型与人类价值观对齐 ，提出RLHF算法
GPT-4 推理能力显著提升，建立可预测的训练框架，可支持多模态信息的大语言模型
o系列模型 推理任务上能力大幅提升 ，长思维链推理能力



### deepseek的发展里程

![图片10.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/9bbc972d-6349-40fb-a221-f2029282649f.jpeg)



语言大模型：DeepSeek LLM/V2/V3、Coder/Coder-V2、Math
多模态大模型：DeepSeek-VL
推理大模型：DeepSeek-R1

DeepSeek-V3
671B参数（37B激活），14.8T训练数据
基于V2的MoE架构，引入了MTP和新的复杂均衡损失
对于训练效率进行了极致优化，共使用 2.788M H800 GPU时



### LLaMA模型的发展历程

![图片11.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/3accc9f5-9358-4b0c-a027-58c6e8864728.jpeg)

LLaMA是Meta于2023年2月发布的模型集合，包含7 B、13 B、33 B和65 B共4个版本。其中LLaMA-13 B在多数数据集上的表现超越了GPT-3并且使用开源语料作为训练语料。而羊驼家族是指一些基于LLaMA模型结合2.1节中涉及方法构建的模型，以下针对Alpace、Vicuna、Koala和Baize 4个羊驼家族成员进行简要介绍。

Alpaca（2023年3月）基于Self-instruct （Wang等，2023b）方法自动构建调优数据集并基于构建的调优数据集监督微调LLaMA。它的优势在于其极低的微调成本以及极少的资源消耗。
Vicuna（2023年4月）其数据集从ShareGPT收集筛选得来，并在Alpaca的基础上，改进了训练损失函数以适应多轮对话场景，增加了最大上下文长度以提升长文本理解能力。
Baize（2023年4月）通过Self-Chat的ChatGPT对话数据自动收集的方法，批量生成高质量多轮对话数据集用于调优。同时，在训练阶段应用了低秩适配 （Kow-rank adaptation，LoRA）方法 （Hu等，2022）进一步降低了微调成本。


### 中文LLaMA & Alpaca大模型

中文LLaMA:通用语言模型能力，能够针对输入内容进行续写 中文Alpaca
在中文LLaMA的基础上，进一步通过指令精调提升模型的instruction following能力

![图片12.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/3c5924d5-8763-4b14-9d1e-52ede9a6a593.jpeg)



原版LLaMA没有显式使用中文语料进行训练，词表(32K tokens)中仅包含非常少的中文字符。此外由于LLaMA使用了sentencepiece分词器，对于不在词表中的文本会切分为byte-level字符



处理方法：
第一步:使用sentencepiece工具在中文预训练语料上训练出20K个单词的中文词表
第二步:删除上述20K中文词表中包含原版LLaMA词表(32K)的部分
第三步:在原版LLaMA词表上拼接去􏰁后的中文词表，得到最终词表
