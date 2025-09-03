# RNN,LSTM,GRU计算方式

## RNN计算方式

```python
torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
```

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

参数
·input_size-输入x中的预期特征数
·hidden_size-隐藏状态h中的特征数量
·num_layers-循环层数。例如，设置意味着将两个RNN堆叠在一起以形成堆叠rNN,第二个RNN接收第一个RNN的输出，并且计算最终结果。默认值：1 num_layers=2
·非线性-要使用的非线性。可以是或。违约：'tanh''relu''tanh
·bias-如果，则该层不使用偏置权重b_h和b_hh。违约：False True
·batch_first-如果，则提供输入和输出张量as (batch,seq,feature),而不是(seq,batch,feature),。请注意，这不适用于隐藏状态或单元格状态。请参阅有关详细信息，请参阅下面的输入/
输出部分。违约：True False
·dropout-如果不为零，则在每个RNN层除最后一层外，丢弃概率等于。默认值：0 dropout
·双向-如果成为双向RNN。违约：True False




## LSTM计算方式



## GRU计算方式
