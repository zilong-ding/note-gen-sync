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
