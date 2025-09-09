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

提高稳定性： 通过对每个子层的 输入 进行归一化，预归一化防止了传递到这些可能复杂的函数中的激活值发生爆炸。通过残差连接 (
𝑥
x) 的输出梯度路径保持清晰，使得梯度能够更顺畅地流经深度网络，而不会被归一化层过度缩放。
对预热的敏感性降低： 预归一化配置通常对学习率调度不那么敏感，即使没有特定的预热阶段，或者预热阶段很短，也能稳定训练。这简化了超参数的调整。
常见实践： 由于其稳定性优势，预归一化已成为许多现代大规模Transformer实现（例如GPT-2、GPT-3、ViT）中的事实标准。与后归一化相比，它使得训练更深的模型成为可能。
