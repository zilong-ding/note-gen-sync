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


```
# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}
```
