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
