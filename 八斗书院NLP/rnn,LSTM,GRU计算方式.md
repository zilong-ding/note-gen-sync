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
input_size—输入x中预期特征的数量
hidden_size-隐藏状态h的特征数量
num_layers-循环层数。，设置意味着将两个rnn堆叠在一起形成一个堆叠RNN，第二个RNN接受第一个RNN的输出并计算最终结果。默认值:1 num_layers = 2
非线性-要使用的非线性。两者皆可。默认值:“双曲正切“relu双曲正切”
bias-如果，则该层不使用bias权重b_ih和b_hh。默认值:虚假的真
batch_first-If，则输入和输出张量提供为（batch,seq,feature），而不是(seq，
批处理功能)。请注意，这不适用于隐藏状态或单元状态。请参阅输入/输出部分
详情如下。默认值:真的假的
drop -如果非零，则在除最后一层之外的每个RNN层的输出上引入一个Dropout层；
退学概率等于。默认值:0辍学
变成了一个双向RNN。默认值:真的假的




## LSTM计算方式



## GRU计算方式
