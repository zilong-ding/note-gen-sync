# AI大模型应用：NLP与大模型

## Transformer:为什么现在BERT和GPT采用这种结构

RNN(或者LSTM、GRU等)的计算限制为是顺序的，也就是RNN相关算法只能从左向右或从右向左依次计算：

时间片t的计算依赖于t-1时刻的计算结果，限制了模型的并行能力；

尽管LSTM等门机制的结构缓解了长期依赖的问题，但顺序计算的过程中信息会丢失；

![2025-09-02_14-24.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/906e11f5-ee9d-40aa-aa92-862cca75c2f1.png)

Transformer的提出解决了上面两个问题，它使用了Attention机制，将序列中的任意两个位置之间的距离缩小为一个常量；其次他不是类似于RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

### 组成原理

#### Seq2seq 模型

Seq2Seq(Sequence-to-Sequence)将自然语言处理中的任务（如文本摘要、机器翻译、对话系统等)看作从一个输入序列到另外一个输出序列的映射，然后通过一个端到端的神经网络来直接学习序列的映射关系。

Seq2Seq也是编码器-解码器结构的雏形。

![2025-09-02_14-32.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/ef6876d8-f362-44a0-a53e-38ac3b60ab79.png)

仅使用encoder的预训练模型：BERT

仅使用decoder的预训练模型：GPT

#### Transformer内部结构

Transformer是一种基于注意力机制的编码器-解码器结构，具有很好的并行性能，同时利用注意力机制很好地解决了长序列的长程依赖问题。

![2025-09-02_14-35.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/f1b3364c-49da-41f9-9f23-76f87508d9bb.png)![2025-09-02_14-35_1.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/6018e64f-e5fe-4beb-858e-456a11737722.png)

多头注意力层接收输入词嵌入与位置编码之和，并进行多头注意力的计算。注意力机制中的Q、K、V的来源不同。以机器翻译为例，Q来源于目标输出，而K和V来源于输入信息。与之相对，自注意力机制的Q、K、V均来源于同一个输入X。

$$
{\mathrm{Atention}}{\big(}Q,K,V{\big)}={\mathrm{Sofimax}}{\Bigg(}{\frac{Q K^{T}}{\sqrt{d_{k}}}}{\Bigg)}V
$$

$$
Q=XW_q,K=XW_k,V=XW_r
$$

多头注意力机制。多头注意力机制是多个自注意力机制的组合，目的是从多个不同角度提取交互信息，其中每个角度称为一个注意力头，多个注意力头可以独立并行计算。

$$
\mathrm{MultiHead}\big(Q,K,V\big)=\mathrm{Concat}\big(\mathrm{head}_{1},\cdots,\mathrm{head}_{b}\big)\boldsymbol{W}^{O}\\\mathrm{head}_{i}=\mathrm{Attention}\big(QW_{i}^{Q},KW_{i}^{\kappa},VW_{i}^{\nu}\big)
$$

Transformer编码器的输入是词嵌入与位置编码之和。将输入序列转化成词嵌入的方法是从一张查询表(Lookup Table)中获取每个词元(Token)对应的向量表示。但如果仅使用词嵌入作为Transformer的注意力机制的输入，则在计算词元之间的相关度时并未考虑它们的位置信息。原始的Transformer采用了正余弦位置编码。

$$
\mathrm{PE}_{(\mathrm{pos},2i)}=\sin\left(\mathrm{pos}/10000^{2i/d_{\mathrm{model}}}\right)\\\mathrm{PE}_{(\mathrm{pos},2i+1)}=\cos\left(\mathrm{pos}/10000^{2i/d_{\mathrm{model}}}\right)
$$

通过计算得出各个位置每个维度上的信息，而非通过训练学习到

输入长度不受最大长度的限制，可以计算到比训练数据更长的位置，具有一定的外推性；

每个分量都是正弦或者余弦函数，并且整体的位置编码具有远程衰减性质，具备位置信息；

![2025-09-02_14-47.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/5e0fa038-d4e5-49b1-97fe-910ea43f7d32.png)

## BERT：判别式模型 vs 生成式模型

### 语言模型

#### 语言模型是一种文字游戏

现阶段所有的NLP模型都不能理解这个世界，只是依赖已有的数据集进行概率计算。而在目前的"猜概率"游戏环境下，基于大型语言模型(LLM，LargeLanguageModel)演进出了最主流的两个方向：BERT和GPT。

BERT是之前最流行的方向，统治了所有NLP领域中的判别任务，并在自然语言理解类任务中发挥出色。而最初GPT则较为薄弱，在GPT3.0发布前，GPT方向一直是弱于BERT的。

![2025-09-02_14-55.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/3eb765af-d609-47bc-ac96-d3d050b580e5.png)


#### 生成式预训练语言模型GPT

OpenAl公司在2018年提出的GPT(Generative Pre-Training)模型是典型的生成式预训练语言模型之一。GPT-2由多层Transformer组成的单向语言模型，主要可以分为输入层，编码层和输出层三部分。

![2025-09-02_14-57.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/cd833afa-5b13-4ddb-b730-c57126c21d44.png)


#### 掩码预训练语言模型BERT

2018年Devlin等人提出了掩码预训练语言模型BERT(Bidirectional Encoder Representation from Transformers)。BERT利用掩码机制构造了基于上下文预测中间词的预训练任务，相较于传统的语言模型建模方法，BERT能进一步挖掘上下文所带来的丰富语义。

![2025-09-02_14-58.png](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/d69e3899-9a08-464d-b84f-f38f9c64aa6d.png)


### BERT模型组成

BERT由多层Transformer编码器组成，这意味着在编码过程中，每个位置都能获得所有位置的信息，而不仅仅是历史位置的信息。BERT同样由输入层，编码层和输出层三部分组成。编码层由多层Transformer编码器组成。

在预训练时，模型的最后有两个输出层MLM和NSP，分别对应了两个不同的预训练任务：掩码语言模型（MaskedLanguage Modeling，MLM）和下一句预测（NextSentencePrediction， NSP)

掩码语言模型的训练对于输入形式没有要求，可以是一句话也可以一段文本，甚至可以是整个篇章，但是下一句预测则需要输入为两个句子，因此BERT在预训练阶段的输入形式统一为两段文字的拼接，这与其他预训练模型相比有较大区别。

![2025-09-02_15-01.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/afa60005-f70f-4a98-8e45-72fcdad644de.jpeg)


在预训练时，模型的最后有两个输出层MLM和NSP,分别对应了两个不同的预训练任务：掩码语言模型(Masked Language Modeling,MLM)和下一句预测(Next Sentence Prediction,NSP)

![2025-09-02_15-03.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/db6db784-96fc-4181-aeaf-52f7b02e77a9.jpeg)


#### BERT微调

![2025-09-02_15-05.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/1a8f353b-db41-41aa-8ba2-e47e8f6d9d1a.jpeg)


![2025-09-02_15-07.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/8dc6975a-233d-488f-a4ce-962484c741f6.jpeg)

截断方法通常是将关键信息截断在起点和终点。我们使用三种不同的方法进行BERT微调。

1.仅限头部:保留前510token；

2.tail-only:保留最后510个令牌；

3.头+尾:凭经验选择第一个128和最后一个382音。

#### BERT分支模型

![2025-09-02_15-08.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/0f529534-c8f2-4d35-ac70-b8dfdd8cc218.jpeg)


![2025-09-02_15-09.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/6b9a0529-8f15-4632-929a-7daa2ab0a4b9.jpeg)


#### BERT模型总结

预训练阶段：BERT采用了无监督的预训练策略，通过在大规模语料库上进行预训练，学习通用的语言表示。模型在两个任务上进行预训川练： Masked Language Model(MLM):随机遮蔽输入文本中的一些词，然后预测这些被遮蔽的词。 Next Sentence Prediction(NSP):预测两个相邻句子是否是原文中相邻的句子。微调阶段：在具体任务上进行有监督的微调，例如文本分类、命名实体识别等。 

优点：模型精度高，且泛化性较好； 

缺点：模型复杂度较高； 





## HuggingFace：现在人工智能和大模型的关键社区
