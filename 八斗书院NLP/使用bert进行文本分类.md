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







### 推理函数
