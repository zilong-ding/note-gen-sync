# pytorch损失函数详解

## 一、损失函数的基本概念

损失函数（Loss Function）又称代价函数（Cost Function），是衡量模型预测结果与真实标签之间差异的指标。在模型训练过程中，通过优化算法（如梯度下降）最小化损失函数，使模型逐渐逼近最优解。

损失函数的选择取决于具体任务类型：

回归任务：预测连续值（如房价、温度）
分类任务：预测离散类别（如图片分类、垃圾邮件识别）
其他任务：如生成任务、序列标注等



## 二、常用损失函数及实现

### 均方误差损失（MSELoss）

均方误差损失是回归任务中最常用的损失函数，计算预测值与真实值之间平方差的平均值。

数学公式：

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}
$$


其中，![y_i](https://latex.csdn.net/eq?y_i)

为真实值，![\hat{y}_i](https://latex.csdn.net/eq?%5Chat%7By%7D_i)

为预测值，n为样本数量。

```python
import torch
import torch.nn as nn
 
# 初始化MSE损失函数
mse_loss = nn.MSELoss()
 
# 示例数据
y_true = torch.tensor([3.0, 5.0, 2.5])  # 真实值
y_pred = torch.tensor([2.5, 5.0, 3.0])  # 预测值
 
# 计算损失
loss = mse_loss(y_pred, y_true)
print(f'MSE Loss: {loss.item()}')  # 输出：MSE Loss: 0.0833333358168602
```


**特点**：

* 对异常值敏感，因为会对误差进行平方
* 是凸函数，存在唯一全局最小值
* 适用于大多数回归任务


### 平均绝对误差损失(L1Loss/MAELoss)

平均绝对误差计算预测值与真实值之间绝对差的平均值，对异常值的敏感性低于 MSE。

数学公式：

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_{i}-\hat{y}_{i}|
$$

```python
# 初始化L1损失函数
l1_loss = nn.L1Loss()
 
# 计算损失
loss = l1_loss(y_pred, y_true)
print(f'L1 Loss: {loss.item()}')  # 输出：L1 Loss: 0.25
```

**特点**：

* 对异常值更稳健
* 梯度在零点处不连续，可能影响收敛速度
* 适用于存在异常值的回归场景




### 交叉熵损失（CrossEntropyLoss）

交叉熵损失是多分类任务的标准损失函数，在 **PyTorch** 中内置了 Softmax 操作，直接作用于模型输出的 logits。

数学公式：

$$
CrossEntropyLoss=-\sum_{i=1}^{C}y_{i}\log(\hat{y}_{i})
$$


其中，C为类别数，![y_i](https://latex.csdn.net/eq?y_i)

为真实标签的 one-hot 编码，![\hat{y}_i](https://latex.csdn.net/eq?%5Chat%7By%7D_i)

为经过 Softmax 处理的预测概率。

```python
def test_cross_entropy():
    # 模型输出的logits（未经过softmax）
    logits = torch.tensor([[1.5, 2.0, 0.5], [0.5, 1.0, 1.5]])
    # 真实标签（类别索引）
    labels = torch.tensor([1, 2])  # 第一个样本属于类别1，第二个样本属于类别2
  
    # 初始化交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f'Cross Entropy Loss: {loss.item()}')  # 输出：Cross Entropy Loss: 0.6422222256660461
 
test_cross_entropy()
```


**计算过程解析**：

1. 对 logits 应用 Softmax 得到概率分布
2. 计算真实类别对应的负对数概率
3. 取平均值作为最终损失

**特点**：

* 自动包含 Softmax 操作，无需手动添加
* 适用于多分类任务（类别互斥）
* 标签格式为类别索引（非 one-hot 编码）



### 二元交叉熵损失（BCELoss）

二元交叉熵损失用于二分类任务，需要配合 Sigmoid 激活函数使用，确保输入值在 (0,1) 范围内。

数学公式：









## 损失函数选择指南
