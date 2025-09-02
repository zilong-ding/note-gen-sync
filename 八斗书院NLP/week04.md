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

![2025-09-02_15-10.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/c5ed5082-52ac-46aa-af0b-5185d0e5792f.jpeg)

## HuggingFace：现在人工智能和大模型的关键社区

## 压测工具ApacheBench

ApacheBench(ab)是一个命令行工具，用于测试Apache HTTP服务器的性能。尽管它最初是为Apache
服务器设计的，但实际上它可以测试任何HTTP服务器。它能够模拟对服务器的并发请求，并提供关于服务器响应时间、吞吐量和请求失败率等关键性能指标。

-n requests:指定要发送的请求总数。

-c concurrency:指定并发请求数。例如，-c10表示同时发起10个请求。

-t timelimit::指定测试的最大时间限制（秒）。

-p postfile:指定包含POST数据的文件。用于测试POST请求。

-T content--type:为POST数据指定Content-Type。

https://httpd.apache.orq/docs/current/programs/ab.html



## 专业名词

### 语言模型的发展

传统的语言模型，比如RNN(循环神经网络)，就像一个“短视”的学生。它在读一篇文章时，只能逐字逐句地读，并且每次只能记住前一个词的信息。当句子很长时，它就会“忘记”前面读过的内容，导致对整句话的理解不完整。

Transformer的出现，就像给这个学生配上了“全局视野”。它最大的创新是引入了自注意力机制(Self-Attention Mechanism)。简单来说，自注意力机制让模型在处理一个词时，能够同时关注到句子中的所有其他词，并根据它们之间的关系来确定每个词的重要性。

BERT的“B”代表双向(Bidirectional)。这意味着BERT在理解一个词时，不仅会看它前面的词，还会看它后面的词。

GPT(Generative Pre-trained Transformer)则更像一个“创造者”或“生成者”。它是一个单向的模型，从左到右地预测下一个词。它的训练方式是给定一句话的前半部分，然后让它预测并生成后面的内容。


### 注意力机制(Attention Mechanism)

人类阅读一段文字时，大脑并不是平等地处理每一个字。相反，你会根据上下文，有意识或无意识地去关注那些对理解当前信息更重要的词语。

注意力机制的工作过程分解成三个简单的步骤： 

1.打分(Scoring):对于句子中的每一个词，模型都会计算它与当前正在处理的词之间的“相关性分数”。分数越高，代表这两个词越相关。 

2.归一化(Normalizing):这些分数会被转换成权重，通常使用Softmax函数。这样做的好处是，所有权重加起来等于1，而且越重要的词权重越高，不重要的词权重越低，这就像给每个词分配了不同等级的关注度。 

3.加权求和(Weighted Sum):最后，模型会将所有词的原始信息（也叫“值”）与它们对应的权重相乘，然后相加。这样得到的最终表示，就包含了所有词的信息，但重点突出了那些权重高的、更重要的词。

注意力机制通过让模型能够“回头看”句子中的所有词，并给它们分配不同的权重，彻底解决了这个问题。它让模型在处理长句子时，不再“短视”，而是拥有了“全局视野”，能更好地理解整个句子的语境和含义。


### BERT 

Bidirectional Encoder Representations from Transformers,直译过来就是“来自 Transformer的双向编码表示”。 

双向(Bidirectional):这是BERT最大的特点。 

编码器(Encoder):表明它擅长理解文本。 

来自Transformer(from Transformers):说明它的底层架构是Transformer。 

BERT的创新之处在于，它采用了双向的训练方式。它在训练时，会随机遮盖(msk)掉句子中一部分词，然后让模型根据被遮盖词的前后所有词来预测它。除了“完形填空”(Masked Language Model))这个主要任务外，BERT还有另一个辅助任务：下一句预测(Next Sentence Prediction)

![2025-09-02_15-52.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/843a2cb4-1086-4c6f-871f-6a4fdf5be046.jpeg)


BERT首先在海量的文本数据上进行“预训练”，学习通用的语言知识。之后，我们可以用少量的特定任务数据对它进行“微调”，让它胜任具体的任务，如**情感分析**、**问答系统**、**命名实体识别**等。

### 分词器 

tokenizer将原始文本转换为模型能够处理的数字序列。像BERT和GPT这样的Transformer模型，都只能处理数字，模型根本不认识这些汉字。分词器的作用，就是把这个句子拆解并转换成一串数字，比如[101,23,45,67,102]。然后，这些数字才能被输入到模型中进行计算。

![2025-09-02_15-56.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/a54aae54-b487-4e2f-94d6-5c0d6cf5527b.jpeg)


1.分词（Tokenization）分词器会把一句话分解成一个个独立的“词元”（Token)。

按词分词：比如，句子“Ilovecats”会被分成['I', 'love'， 'cats']

按字分词：对于中文，句子“我爱中国”会被分成［'我'，‘爱'，‘中'，‘国']

2.特殊标记（SpecialTokens）为了让模型更好地理解句子的结构，分词器会添加一些特殊的标记：

> [CLS]（Classification)：通常放在句子的开头，它代表了整个句子的信息，常用于文本分类任务。

> [SEP](Separation)：通常放在句子的结尾，用于分隔不同的句子。如果想让模型同时处理两个句子（比如问答任务)，会在它们之间放一个[SEP]

> [PAD](Padding)：当批次中的句子长度不一时，分词器会用[PAD]来填充短句子，让所有句子的长度都相同，方便模型并行处理。

3.映射到ID（Token toID）分词器内部有一个巨大的“词汇表”（Vocabulary)，它将每个词元映射到一个唯一的整数ID。

4.生成注意力掩码（AttentionMask）在上面的步骤2中，我们用[PAD]填充了句子。但模型需要知道哪些是真正的词，哪些是填充物。


### ·位置编码 

Transformer的位置编码不是简单的数字。它是一种与词嵌入(word embedding)相加的编码方式。在将词语输入模型之前，会将它的词嵌入向量和位置编码向量相加。位置编码有多种实现方式，但Transformer论文中介绍的是一种基于正弦(sine)和余弦(cosine)函数的编码方法。
