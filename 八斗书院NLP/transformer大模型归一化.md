# transformer大模型归一化

当前主流大模型使用的Normalization主要有三类，分别是Layer Norm，[RMS Norm](https://zhida.zhihu.com/search?content_id=225975397&content_type=Article&match_order=1&q=RMS+Norm&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NTc1NzE4NDUsInEiOiJSTVMgTm9ybSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIyNTk3NTM5NywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0._QqipVJi_Llfnq61FfFthxIJaqUvoeUGvsMWMD_tsyg&zhida_source=entity)，以及Deep Norm，这里依次介绍他们的异同

在随机优化理论中，学习率往往设置为常数或者逐渐衰减 (decay)，从而保证算法收敛，这种学习率的设置方法也与机器学习里很多任务上的实际经验类似。然而，不管是设置学习率为常数还是使学习率逐渐衰减都不能让Transformer很好地收敛。

在优化Transformer结构时，除了设置初始学习率与它的衰减策略，往往还需要在训练的初始阶段设置一个非常小（接近0）的学习率，让它经过一定的迭代轮数后逐渐增长到初始的学习率，这个过程称作warm-up阶段（学习率预热）。

Warm-up是原始Transformer结构优化时的一个必备学习率调整策略。Transformer结构对于warm-up的超参数（持续轮数、增长方式、初始学习率等）非常敏感，若调整不慎，往往会使得模型无法正常收敛。

Transformer结构的优化非常困难，其具体表现在：

> warm-up阶段超参数敏感；
> 优化过程收敛速度慢。

## Post-LN&Pre-LN

针对以上问题，论文《On Layer Normalization in the Transformer Architecture》提出了两种Layer Normalization方式并进行了对比。

把Transformer架构中传统的**Add&Norm**做layer normalization的方式叫做Post-LN，并针对Post-LN，模型提出了Pre-LN，即把layer normalization加在残差连接之前，如下图所示：

![2025-09-09_14-36.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/2987a49e-e8a4-4d29-b1ce-b375dd621638.jpeg)

![2025-09-09_14-36_1.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/01d370cd-f9b3-4ba2-8883-7f03ba612dc3.jpeg)


### post-ln特点

**原始形式**： 这是Vaswani等人（2017）论文中描述的设置。

**潜在的不稳定性**： 后归一化面临的主要问题是，残差分支的输出（和$x+SubLayer(x)$）在传递给下一层 之前 未经归一化。在深度网络中，激活值的幅度可能在层与层之间显著变化，这可能导致训练初期出现梯度爆炸或梯度消失。

**预热要求**： 后归一化配置通常需要仔细的学习率预热阶段（从较小的学习率开始并逐渐增加）。如果没有预热，由于未经归一化的加法，初始梯度可能过大，导致训练发散。

### pre-ln特点

**提高稳定性**： 通过对每个子层的 输入 进行归一化，预归一化防止了传递到这些可能复杂的函数中的激活值发生爆炸。通过残差连接 (𝑥) 的输出梯度路径保持清晰，使得梯度能够更顺畅地流经深度网络，而不会被归一化层过度缩放。

**对预热的敏感性降低**： 预归一化配置通常对学习率调度不那么敏感，即使没有特定的预热阶段，或者预热阶段很短，也能稳定训练。这简化了超参数的调整。

**常见实践**： 由于其稳定性优势，预归一化已成为许多现代大规模Transformer实现（例如GPT-2、GPT-3、ViT）中的事实标准。与后归一化相比，它使得训练更深的模型成为可能。


| 特点     | 后归一化(Post-LN)                    | 预归一化(Pre-LN)                           |
| -------- | ------------------------------------ | ------------------------------------------ |
| 放置位置 | LayerNorm(x + SubLayer(x))           | X + SubLayer(LayerNorm(x))                 |
| 稳定性   | 稳定性较差，尤其在深层模型中         | 稳定性更好，有助于训练更深的模型           |
| 预热     | 通常需要仔细的学习率预热             | 对学习率预热不那么敏感，常无需预热也能训练 |
| 梯度流动 | 梯度在相加后通过归一化层             | 梯度通过残差路径绕过归一化层               |
| 原始论文 | 是                                   | 否 (后续改进)                              |
| 现代应用 | 在非常大的模型中较少见               | 被广泛采用，尤其对于大型模型               |
| 最佳表现 | 经过大量调整有时能达到路好的最佳结果 | 通常更容易调整以获得良好、稳定的结果       |


## deepnorm

### 📌 背景与动机

* **问题**：标准 Transformer 在层数 > 32 时容易训练崩溃（梯度爆炸/消失、输出爆炸）。
* **LayerNorm + Residual 的缺陷**：随着层数加深，残差分支不断叠加 → 输出值指数级增长 → 不稳定。
* **目标**：设计一种新的归一化 + 初始化组合，让 1000 层 Transformer 也能稳定训练。

→ **DeepNorm 诞生**：出自 2022 年论文《DeepNet: Scaling Transformers to 1,000 Layers》（微软研究院）

### 🧮 公式与结构

DeepNorm 不是一个“独立归一化层”，而是一个**残差连接 + 归一化 + 参数缩放**的组合方案。

#### 标准 Post-LN Transformer（不稳定）：

x = x + F(x)          **# 残差连接**

x = LayerNorm(x)      **# 后归一化**

#### DeepNorm 改进版：

x = x + α \* F(x)      **# 残差连接加缩放因子 α**

x = LayerNorm(x)      **# 仍用 LayerNorm**

同时，**对 F(x) 内部的参数（如 Attention、FFN 的权重）进行特殊初始化**：

* 缩放因子 β 乘在初始化权重上 → 控制每层输出幅度

#### 缩放因子设计（根据架构选择）：


|                 |     |     |          |
| --------------- | --- | --- | -------- |
| Encoder-only    | 1.0 | —  | β = 0.2 |
| Decoder-only    | —  | 0.8 | β = 0.1 |
| Encoder-Decoder | 1.0 | 0.8 | β = 0.1 |

> α 控制残差幅度，β 控制函数 F(x) 输出幅度 → 二者协同防止输出爆炸。

### ✅ 为什么有效？

1. **控制残差增长**：α < 1（尤其在 decoder）抑制了残差叠加的爆炸趋势。
2. **初始化配合**：β 缩放让每层输出方差稳定 → 梯度传播更平稳。
3. **兼容 LayerNorm**：不改变归一化方式，工程改动小。

### 📈 效果

* 成功训练 1000 层 Transformer（标准结构只能到 \~32 层）。
* 在机器翻译、长文本建模任务中显著优于 Post-LN 和 Pre-LN。
* 成为“极深 Transformer”的标配方案之一。

### 🧠 直观理解：

> “每加一层，就给残差‘踩一脚刹车’（α），同时让新增模块‘轻手轻脚’（β），避免累积爆炸。”






## RMSNorm




### 背景与动机

* **LayerNorm 开销大**：需要计算均值和方差 → 减均值操作在 GPU 上效率低。
* **均值是否必要？** 实验发现：在很多任务中，**去掉减均值，只除以标准差，效果几乎不降，甚至更好！**
* **目标**：简化 LayerNorm，加速训练和推理，降低内存访问。

→ **RMSNorm 诞生**：出自 2019 年论文《Root Mean Square Layer Normalization》

### 🧮 公式对比

#### LayerNorm：

对于输入向量 `x ∈ ℝᴰ`：

μ = mean(x)                    # 计算均值

σ² = mean((x - μ)²)            # 计算方差

x̂ = (x - μ) / √(σ² + ε)        # 减均值 + 除标准差

y = γ ⊙ x̂ + β                  # 仿射变换

#### RMSNorm：

RMS = √( mean(x²) + ε )        # 只计算均方根，不减均值！

x̂ = x / RMS

y = γ ⊙ x̂                      # 通常省略 β（偏移项）

> 注意：RMSNorm **没有减均值步骤**，也没有 β 偏移项（实践中常省略）。

### ✅ 为什么有效？

1. **简化计算**：省去减均值 → 减少一次内存读写 → 更快（尤其在 GPU 上）。
2. **实验鲁棒**：在语言建模、机器翻译等任务中，效果与 LayerNorm 相当或略优。
3. **对异常值更鲁棒**？有研究认为：不减均值可能保留更多“token 重要性”信息。

### 📈 效果与采用情况

* **LLaMA 系列（Meta）**：全部使用 RMSNorm
* **Mistral、Mixtral（Mistral AI）**：使用 RMSNorm
* **Falcon（TII）**：使用 LayerNorm → 但很多复现版改用 RMSNorm
* 推理速度提升 \~5\~10%，训练速度也有提升
