# 使用bert进行文本分类

## 外卖评论数据集

```
label,review
1,很快，好吃，味道足，量大
1,没有送水没有送水没有送水
1,非常快，态度好。
1,方便，快捷，味道可口，快递给力
1,菜味道很棒！送餐很及时！
1,今天师傅是不是手抖了，微辣格外辣！
1,"送餐快,态度也特别好,辛苦啦谢谢"
```

## bert微调

```python
# 导入必要的库
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
```

```python
# 加载数据集
dataset_path = "waimai_10k.csv"

dataset_df = pd.read_csv(dataset_path, sep=",", header=None)[1:]

print(dataset_df.shape)
print(dataset_df.head())
```

```python
# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df[0].values[:])
texts = list(dataset_df[1].values[:])

print(len(texts))

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)
```

```python
# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=2)
```

```python
# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)
```

```python
# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

print(train_dataset.shape)
```

```python
# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}
```

```python
# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=8,                  # 训练的总轮数
    per_device_train_batch_size=32,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=32,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="best",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

# 实例化 Trainer
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()
trainer.save_model("best")
print("Done")
```

## fastapi部署

### 数据接口定义

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional
```

BaseModel: pydantic 的核心类，所有数据模型都继承自它，用于定义结构化数据。
Field: 用于对字段添加额外信息，如默认值、描述、约束等。
Dict, List, Any, Union, Optional: 来自 typing 模块，用于类型注解：

* Optional[str] 等价于 Union[str, None]，表示该字段可为空（即可以是 str 或 None）。
  Union[A, B] 表示字段可以是 A 类型或 B 类型。
  List[str] 表示字符串列表。
  Any 表示任意类型（不推荐过度使用，会失去类型安全）。

```python
# 请求模型
class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
```

`request_id: Optional[str] = Field(..., description="请求id, 方便调试")`

* 类型：`Optional[str]` → 可以是字符串，也可以是 `None`（即这个字段不是必须传的）。
* `Field(..., ...)`：
  * 第一个 `...` 表示这个字段是**必填项**（即使类型是 `Optional`，但如果不传值，也会报错）。
  * `description`：字段描述，会在 API 文档（如 Swagger）中显示。
* 用途：客户端传一个请求 ID，便于服务端日志追踪和调试。

> ⚠️ 注意：`Optional[str]` + `...` 意味着：**可以传 null，但不能不传字段**。
> 如果你想让字段完全可选（可不传），应写成：`request_id: Optional[str] = None`

`request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")`

* 类型：可以是一个字符串，也可以是一个字符串列表。
  * 比如：`"今天天气真好"` 或 `["今天天气真好", "我很开心"]`
* 必填字段（因为用了 `...`）
* 用途：表示要进行分类的文本内容，支持单条或批量输入。

```python
# 响应模型
class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")
```

`request_id: Optional[str] = Field(...)`

* 返回客户端传入的 `request_id`，方便对应请求与响应。
* 仍是可为空的字符串，且为必填字段（必须返回，但值可以是 `null`）。

`request_text: Union[str, List[str]] = Field(...)`

* 回显客户端传入的原始文本，便于核对。
* 类型与请求一致。

`classify_result: Union[str, List[str]] = Field(...)`

* 分类结果，如果输入是字符串，输出就是单个分类标签（如 `"负面"`）；
* 如果输入是列表，输出也应是对应的标签列表（如 `["负面", "正面"]`）。
* 类型与 `request_text` 对应。

`classify_time: float = Field(...)`

* 类型：浮点数（单位：秒）
* 表示模型完成分类所花费的时间，用于性能监控或前端展示。
* 例如：`0.123` 表示耗时 123 毫秒。

`error_msg: str = Field(...)`

* 错误信息字段。
* 即使成功，也建议返回空字符串 `""` 或 `"success"`。
* 如果出错，填充错误描述，如 `"模型加载失败"`。

### 推理函数

```python
# 导入依赖和初始化配置
from typing import Union, List
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification

from config import BERT_MODEL_PERTRAINED_PATH, BERT_MODEL_PKL_PATH, CATEGORY_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH, num_labels=2)
model.to(device)
```

自定义datasets类

```python
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)
```

这是 PyTorch 的标准数据集封装类，用于将文本编码和标签打包供 `DataLoader` 使用。

方法详解：

* `__init__`: 接收 `encodings`（tokenize 后的结果）和 `labels`（标签列表）。
* `__getitem__(idx)`:
  * 把每个样本的 `input_ids`, `attention_mask` 等转为 `torch.tensor`。
  * 同时把标签也转为 tensor（虽然是测试集，但为了统一输入格式仍需提供 `labels`）。
* `__len__()`: 返回样本数量。

> 💡 注意：测试时 `labels` 设为 `[0]*len(request_text)` 是合理的，因为我们只关心预测结果，不参与损失计算。

核心函数

```python
# 定义一个函数，输入可以是单个字符串或字符串列表，输出是对应的分类结果（字符串或字符串列表）。
def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None
#   输入格式统一化
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")
# 分词与编码
# 参数说明：
#    truncation=True: 超过最大长度时截断。
#    padding=True: 自动补全到 batch 中最长序列长度（用于 batch 推理）。
#    max_length=30: 最多保留 30 个 token（包含 [CLS], [SEP]）。
    test_encoding = tokenizer(list(request_text), truncation=True, padding=True, max_length=30)
  
#   构建测试数据集和数据加载器
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#   模型推理
    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())
#   映射类别名称
    classify_result = [CATEGORY_NAME[x] for x in pred]
    return classify_result
```

模型推理部分

* `model.eval()`: 切换到评估模式（关闭 dropout、batch norm 使用固定统计量）。
* `with torch.no_grad()`: 停止梯度计算，节省内存，加快推理速度。
* `input_ids`, `attention_mask`, `labels` 移动到设备上（GPU/CPU）。
* `outputs = model(...)`:
  * 返回一个元组，`outputs[0]` 是 loss（因为提供了 labels），`outputs[1]` 是 logits。
  * 我们只需要 `logits`（未归一化的分类得分）。
* `logits.detach().cpu().numpy()`:
  * 从计算图中分离 → 移动到 CPU → 转为 NumPy 数组。
* `np.argmax(..., axis=1)`: 找出每条样本得分最高的类别索引。
* `pred += list(...)`：将当前 batch 的预测结果加入总列表。

### ab压测

ApacheBench(ab)是一个命令行工具，用于测试Apache HTTP服务器的性能。尽管它最初是为Apache服务器设计的，但实际上它可以测试任何HTTP服务器。它能够模拟对服务器的并发请求，并提供关于服务器响应时间、吞吐量和请求失败率等关键性能指标。

> -n requests:指定要发送的请求总数。
> -c concurrency:指定并发请求数。例如，-c10表示同
> 时发起10个请求。
> -t timelimit:指定测试的最大时间限制（秒）。
> p postfile:指定包含POST数据的文件。用于测试
> POST请求。
> -T content-type:为POST数据指定Content-Type。

```bash
echo '{"request_id": "string","request_text": "很快，好吃，味道足，量大"}' > test.json
```

```bash
ab -n 100 -c 1 -T "application/json" -p test.json http://localhost:8000/v1/text-cls/bert
```


| 参数             | 含义                         |
| ---------------- | ---------------------------- |
| ab               | Apache 压力测试工具          |
| -n 100           | 总共发 100个请求             |
| -c 1             | 每次并发1个 (串行)           |
| -T"apliationson" | 告诉服务器请求体是JSON       |
| -p estson        | 从文件读取JSON数据作为请求体 |
| http/.           | 目标API地址                  |





## 这里为什么要构建测试数据集和数据加载器？

**构建 `Dataset` + `DataLoader` 是为了：**

1. **统一训练与推理的数据流程**
2. **支持批量推理（batch inference），提升效率**
3. **利用 PyTorch 的自动批处理和设备搬运机制**
4. **保证代码结构清晰、可维护、可扩展**

### ✅ 总结：为什么要构建测试数据集和数据加载器？


| 原因            | 说明                      |
| --------------- | ------------------------- |
| ✅ 批量推理     | 提升 GPU 利用率，加快速度 |
| ✅ 内存友好     | 避免一次性加载全部数据    |
| ✅ 流程统一     | 训练、验证、推理保持一致  |
| ✅ 自动 padding | 不用手动对齐序列长度      |
| ✅ 易于维护     | 结构清晰，便于扩展和调试  |
| ✅ 工程最佳实践 | 工业级项目的标准做法      |
