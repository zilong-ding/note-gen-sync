# RNN,LSTM,GRU计算方式

## RNN计算方式

`torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)`

$$
h_t=\tanh(x_tW_{ih}^T+b_{ih}+h_{t-1}W_{hh}^T+b_{hh})
$$

```python
# Efficient implementation equivalent to the following with bidirectional=False
def forward(x, h_0=None):
    if batch_first:
        x = x.transpose(0, 1)
    seq_len, batch_size, _ = x.size()
    if h_0 is None:
        h_0 = torch.zeros(num_layers, batch_size, hidden_size)
    h_t_minus_1 = h_0
    h_t = h_0
    output = []
    for t in range(seq_len):
        for layer in range(num_layers):
            h_t[layer] = torch.tanh(
                x[t] @ weight_ih[layer].T
                + bias_ih[layer]
                + h_t_minus_1[layer] @ weight_hh[layer].T
                + bias_hh[layer]
            )
        output.append(h_t[-1])
        h_t_minus_1 = h_t
    output = torch.stack(output)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h_t
```

* 参数：
  ·input_size-输入X中预期特征的数量
  ·hidden size-隐藏状态h中的特征数量
  ·num_layers-循环层数。例如，设置num_layers:=2表示将两个RNN堆叠在一起形成一个堆叠RNW,第二
  个RNN接收第一个RNN的输出并计算最终结果。默认为：1
  ·nonlinearity-使用的非线性函数。可以是"tanh或"relu'。默认为："tanh
  ·bias-如果为False,则该层不使用偏置权重bh和bhh。默认为：True
  ·batch_first-如果为True,则输入和输出张量提供为(batch,seq,feature),而不是(seg,batch,feature)。请
  注意，这不适用于隐藏状态或单元状态。有关详细信息，请参阅下面的输入/输出部分。默认为：False
  ·dropout-如果非零，则在除最后一层外的每个RNN层的输出上引入Dropout层，dropout概率等于
  dropout。默认为：0
  ·bidirectional-如果为True,则成为双向RNN。默认为：False


输入：input,,hx
·input:形状为(L,Hin)的张量（用于无批次输入），形状为(L,N,Hn)当batch_first=:False时，或形状为(N,L,Him)当batch_first=True时，包含输入序列的特征。输入也可以是填充的可变长度序列。有关详细信息，请参阅torch.nn.utils.rnn,pack padded sequence()或torch.nn.utils.rnn.pack_sequence()

·hx:形状为(D*num_layers,Hout)的张量（用于无批次输入），或形状为(D*num layers,.N,Hout)的张量（用于输入序列批次）包含初始隐藏状态。如果未提供，则默认为零。

其中

$$
\begin{aligned}&N=\text{batch size}\\&L=\text{sequence length}\\&D=2\text{ if bidirectional=True otherwise 1}\\&H_{in}=\text{input size}\\&H_{out}=\text{hidden size}\end{aligned}
$$

输出：output,hn

output:形状为(L,D*Hot)的张量（用于无批次输入），或形状为(L,N,D*Hout)当batch_first:=False时，或形状为(W,L,D*Hout)当batch_first=True时，包含从RNN最后一层在每个t的输出特征ht)。如果输入是torch.nn.utils.rnn.PackedSequence,则输出也将是打包序列。

hn:形状为(D*num _layers,Hot)的张量（用于无批次输入），或形状为D*num_layers,N,Hout)的张量（用于批次中的每个元素）包含最终隐藏状态。

变量：

weight ih[k]-第k层的可学习输入-隐藏权重，对于k=O,形状为hidden size,input size)。否则，形状(hidden size,num_directions hidden_size)

weight_hh[k-第k层的可学习隐藏-隐藏权重，形状为hidden_size,.hidden_size)

bias ih I[k-第k层的可学习输入-隐藏偏置，形状为hidden_size)

bias_hh[k-第k层的可学习隐藏-隐藏偏置，形状为hidden_size)



## LSTM计算方式

`torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)`

$$
\begin{aligned}&i_{t}=\sigma(W_{ii}x_{t}+b_{ii}+W_{hi}h_{t-1}+b_{hi})\\&f_{t}=\sigma(W_{if}x_{t}+b_{if}+W_{hf}h_{t-1}+b_{hf})\\&g_{t}=\tanh(W_{ig}x_{t}+b_{ig}+W_{hg}h_{t-1}+b_{hg})\\&o_{t}=\sigma(W_{io}x_{t}+b_{io}+W_{ho}h_{t-1}+b_{ho})\\&c_{t}=f_{t}\odot c_{t-1}+i_{t}\odot g_{t}\\&h_{t}=o_{t}\odot\tanh(c_{t})\end{aligned}
$$













## GRU计算方式
