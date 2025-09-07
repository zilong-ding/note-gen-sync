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




## 大语言模型
