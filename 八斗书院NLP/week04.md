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







### 组成的层

### 使用案例


## BERT：判别式模型 vs 生成式模型

## HuggingFace：现在人工智能和大模型的关键社区
