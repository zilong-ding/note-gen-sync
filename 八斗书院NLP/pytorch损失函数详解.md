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

### 平滑L1损失（Smooth L1 Loss）

Smooth L1损失结合了L1和L2损失的优点，当误差小于阈值时使用平方项，否则使用绝对项。其公式如下：

$$
\text{SmoothLLoss}=\begin{cases}0.5\times(y_i-\hat{y}_i)^2/\beta,&\text{if }|y_i-\hat{y}_i|<\beta\\|y_i-\hat{y}_i|-0.5\times\beta,&\text{otherwise}\end{cases}
$$

其中，β 是控制阈值的参数。

**应用场景**：

* 回归任务，尤其是特征值较大时。
* 对异常值的鲁棒性较好。

```python
loss_fn = nn.SmoothL1Loss(beta=1.0)
output = loss_fn(input, target)
print(output)
```


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

$$
BCELoss=-\frac{1}{n}\sum_{i=1}^{n}[y_{i}\log(\hat{y}_{i})+(1-y_{i})\log(1-\hat{y}_{i})]
$$

```python
def test_bce_loss():
    # 模型输出（已通过sigmoid处理）
    y_pred = torch.tensor([[0.7], [0.2], [0.9], [0.7]])
    # 真实标签（0或1）
    y_true = torch.tensor([[1], [0], [1], [0]], dtype=torch.float)
  
    # 方法1：使用BCELoss
    bce_loss = nn.BCELoss()
    loss1 = bce_loss(y_pred, y_true)
  
    # 方法2：使用functional接口
    loss2 = nn.functional.binary_cross_entropy(y_pred, y_true)
  
    print(f'BCELoss: {loss1.item()}')  # 输出：BCELoss: 0.47234177589416504
    print(f'Functional BCELoss: {loss2.item()}')  # 输出：Functional BCELoss: 0.47234177589416504
 
test_bce_loss()
```

**变种：BCEWithLogitsLoss**
对于未经过 Sigmoid 处理的 logits，推荐使用`BCEWithLogitsLoss`，它内部会自动应用 Sigmoid，数值稳定性更好：

```python
# 对于logits输入（未经过sigmoid）
logits = torch.tensor([[0.8], [-0.5], [1.2], [0.6]])
bce_with_logits_loss = nn.BCEWithLogitsLoss()
loss = bce_with_logits_loss(logits, y_true)
```

### 负对数似然损失（Negative Log-Likelihood Loss, NLLLoss）

NLLLoss适用于分类任务，直接计算对数似然的负值。其公式如下：

$$
\text{NLLLoss}=-\sum_{i=1}^Ny_i\log\hat{y}_i
$$


**应用场景**：

* 分类任务，尤其是简单任务或训练速度较快时。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建一个 3x5 的张量，表示 3 个样本，每个样本有 5 个特征
input = torch.randn(3, 5, requires_grad=True)

# 创建一个大小为 3 的张量，表示每个样本的类别索引（0 到 4）
target = torch.tensor([1, 0, 4])

# 定义负对数似然损失函数
loss_fn = nn.NLLLoss()

# 应用 LogSoftmax 函数到输入张量
log_softmax_output = F.log_softmax(input, dim=1)

# 计算损失
output = loss_fn(log_softmax_output, target)

print(output)
```

### 三元组损失（Triplet Margin Loss）

三元组损失用于嵌入学习，通过最小化锚点和正样本之间的距离，同时最大化锚点和负样本之间的距离。其公式如下：

$$
\text{TripletMarginLoss}=\max\left(d_{ap}-d_{an}+\operatorname*{margin},0\right)
$$

其中，dap 是锚点和正样本之间的距离，dan 是锚点和负样本之间的距离。

**应用场景**：

* 排序任务，如人脸验证和搜索检索。


嵌入学习（Embedding Learning）是一种将复杂、高维数据（如文本、图像或声音）转换为低维、稠密向量表示的技术。这些向量能够捕捉数据之间的内在联系，如相似性，使得相似的数据点在向量空间中彼此接近。嵌入学习通常涉及无监督或半监督的学习过程，模型在大量未标记的数据上进行预训练，以学习数据的基本特征和结构。预训练的嵌入可以被进一步微调，以适应特定的下游任务，如分类、聚类或推荐系统。

排序任务（Learning to Rank，LTR）是机器学习中的一个重要任务，尤其在信息检索、推荐系统和广告排序等领域有着广泛的应用。排序任务的目标是将数据按照一定的标准进行排序，以满足特定的需求，如将最相关的文档排在搜索结果的前面，或者将最符合用户兴趣的商品推荐给用户。

<iframe id="aswift_5" name="aswift_5" browsingtopics="true" sandbox="allow-forms allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-top-navigation-by-user-activation" width="916" height="0" frameborder="0" marginwidth="0" marginheight="0" vspace="0" hspace="0" allowtransparency="true" scrolling="no" allow="attribution-reporting; run-ad-auction" src="https://googleads.g.doubleclick.net/pagead/ads?client=ca-pub-4340231068438843&output=html&h=280&adk=637260312&adf=4274663820&pi=t.aa~a.2347235066~i.170~rp.4&w=916&fwrn=4&fwrnh=100&lmt=1756970179&num_ads=1&rafmt=1&armr=3&sem=mc&pwprc=2244349787&ad_type=text_image&format=916x280&url=https%3A%2F%2Fjishuzhan.net%2Farticle%2F1909306269229449217&fwr=0&pra=3&rh=200&rw=915&rpe=1&resp_fmts=3&wgl=1&fa=27&uach=WyJMaW51eCIsIjYuMTQuMCIsIng4NiIsIiIsIjEzOS4wLjM0MDUuMTI1IixudWxsLDAsbnVsbCwiNjQiLFtbIk5vdDtBPUJyYW5kIiwiOTkuMC4wLjAiXSxbIk1pY3Jvc29mdCBFZGdlIiwiMTM5LjAuMzQwNS4xMjUiXSxbIkNocm9taXVtIiwiMTM5LjAuNzI1OC4xNTUiXV0sMF0.&abgtt=6&dt=1756970140504&bpp=1&bdt=3526&idt=1&shv=r20250903&mjsv=m202509020101&ptt=9&saldr=aa&abxe=1&cookie=ID%3D3fa32f74630b17f4%3AT%3D1756970139%3ART%3D1756970139%3AS%3DALNI_MZHGGg7FsRS9SBfy1wI9s6Hr0L02g&gpic=UID%3D000011127e4535ee%3AT%3D1756970139%3ART%3D1756970139%3AS%3DALNI_MYvzD3nxcwi1Omrt7YFKE7tEE4H6A&eo_id_str=ID%3Df67e67839b20849f%3AT%3D1756970139%3ART%3D1756970139%3AS%3DAA-AfjZjV72LFSYVikRHrHycIgI8&prev_fmts=0x0%2C308x250%2C308x250%2C674x280%2C605x280&nras=4&correlator=5099568063629&frm=20&pv=1&u_tz=480&u_his=2&u_h=1080&u_w=1920&u_ah=1052&u_aw=1920&u_cd=24&u_sd=1&dmc=8&adx=491&ady=6347&biw=1897&bih=938&scr_x=0&scr_y=2602&eid=31094367%2C42532524%2C95362655%2C95369636%2C95369802%2C95370331%2C95370343%2C31094473&oid=2&psts=AOrYGskMjdXy0JxxO5VvhVyQujyUy2gYvXdnyMS8H5gnx6G35ncM9OJJPni95gpRVN9LSNWB1Pc_YlCKKOK2X7suAN-BBy8e6tPFW4okQgd8M2vKrak&pvsid=5980779720325596&tmod=1226655176&uas=0&nvt=1&ref=https%3A%2F%2Fwww.bing.com%2F&fc=1408&brdim=0%2C28%2C0%2C28%2C1920%2C28%2C1920%2C1052%2C1912%2C938&vis=1&rsz=%7C%7Cs%7C&abl=NS&fu=128&bc=31&bz=1&td=1&tdf=2&psd=W251bGwsbnVsbCxudWxsLDNd&nt=1&ifi=6&uci=a!6&btvi=3&fsb=1&dtd=38916" data-google-container-id="a!6" tabindex="0" title="Advertisement" aria-label="Advertisement" data-google-query-id="CJeruPvHvo8DFVFRwgUd0XgO7w" data-load-complete="true"></iframe>

`TripletMarginLoss` 是一种用于嵌入学习和排序任务的损失函数。它通过优化锚点和正样本之间的距离与锚点和负样本之间的距离的差值，使得相似样本的嵌入更接近，不相似样本的嵌入更远离。这种损失函数在人脸识别、图像检索、推荐系统等任务中被广泛应用，能够有效提高模型的嵌入质量，使得模型在处理相似性匹配和排序问题时表现更优。




## 损失函数选择指南


| 任务类型            | 推荐损失函数              | 特点                                 |
| ------------------- | ------------------------- | ------------------------------------ |
| 回归任务            | MSELoSS                   | 对异常值敏感，适用于大多数回归场景   |
| 回归任务 (含异常值) | L1Loss                    | 对异常值稳健，梯度不连续             |
| 多分类任务          | CrossEntropyLoss          | 内置 Softmax，处理互斥类别           |
| 二分类任务          | BCELoss/BCEWithLogitsLoss | 配合 Sigmoid 使用，输出概率值        |
| 多标签分类          | BCEWithLogitsLoss         | 每个类别独立判断，可同时属于多个类别 |
