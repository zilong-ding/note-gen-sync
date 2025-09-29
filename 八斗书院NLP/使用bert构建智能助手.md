# 使用bert构建智能助手

## 背景

智能对话系统（如聊天机器人、智能语音助手）、搜索引擎和垂直领域问答系统的核心。

不是闲聊助手，理解查询意图，并将其转化为计算机可以精确处理和执行的结构化数据。

这里构建导航助手

## 输入和输出

原始输入（text）: “查询许昌到中山的汽车。”

语义解析输出（Structured Data）:

```python
intent： 意图 （用户提问的类型）
QUERY（查询）、BOOK（预订）、CANCEL（取消）、COMPARE（对比）
domain： 领域
slots：槽位（实体）
{ “Dest”: “中山”, “Src”: “许昌” }
```

结构化查询

```sql
SELECT * FROM bus_schedule WHERE src = '许昌' AND dest = '中山';
```

组织为自然语言 -》 输出

## 项目运行方案

![2025-09-27_10-03.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/f88f016d-2215-4b43-a90f-9a28878703cc.jpeg)

该方案是早期的一个方案，这里槽位识别与意图识别分别使用两个bert模型。但是两个bert一个是训练要分两次，部署的时候也会占用较大的显存，于是改为自定义bert模型，使其既能识别意图也能识别槽位信息。

本项目要做的范围从输入文本信息开始到输出结构化JSON为止

### 数据集构建

```python
CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "天津",
    "重庆", "苏州", "郑州", "长沙", "青岛", "沈阳", "大连", "宁波", "厦门", "许昌",
    "中山", "佛山", "东莞", "珠海", "惠州", "江门", "肇庆", "汕头", "潮州", "揭阳"
]

VEHICLES = {
    "汽车": "bus",
    "大巴": "bus",
    "班车": "bus",
    "火车": "train",
    "高铁": "train",
    "动车": "train",
    "飞机": "plane",
    "航班": "plane"
}

DATES = ["今天", "明天", "后天", "周一", "周二", "周三", "周四", "周五", "周六", "周日", "下周一"]
TIMES = ["上午", "下午", "晚上", "早上", "中午", "9点", "10点", "15点", "18点"]
SEAT_TYPES = ["一等座", "二等座", "商务座", "硬座", "软座", "卧铺"]
```

```python
# templates.py
TEMPLATES = {
    "QUERY": [
        "查一下{src}到{dest}的{vehicle}。",
        "我想看看{src}去{dest}有什么{vehicle}。",
        "从{src}到{dest}的{vehicle}有哪些？",
        "有没有{src}到{dest}的{vehicle}？",
        "查询{date}{src}到{dest}的{vehicle}。",
        "{src}到{dest}的{vehicle}什么时候发车？",
        "帮我查{src}到{dest}的{vehicle}票。",
        "看看{src}去{dest}的{vehicle}。",
        "我想知道{src}到{dest}的{vehicle}信息。",
        "查{src}到{dest}的{vehicle}班次。"
    ],
    "BOOK": [
        "我想订一张{date}{src}到{dest}的{vehicle}票。",
        "帮我预订{date}{time}{src}到{dest}的{vehicle}。",
        "订票：{src}到{dest}，{date}，{vehicle}。",
        "我要买{date}从{src}到{dest}的{vehicle}票，{passenger_count}个人。",
        "预订{src}到{dest}的{vehicle}，{date}出发，{seat_type}。",
        "帮我订{date}{src}去{dest}的{vehicle}，{passenger_count}张。",
        "我想订{date}{time}从{src}到{dest}的{vehicle}，{seat_type}。",
        "订一张{src}到{dest}的{vehicle}票，{date}，{passenger_count}人。",
        "我要预订{date}从{src}到{dest}的{vehicle}，{seat_type}。",
        "帮我订{src}到{dest}的{vehicle}，{date}，{passenger_count}个人。"
    ],
    "COMPARE": [
        "比较一下{src}到{dest}坐{vehicle1}和{vehicle2}哪个快？",
        "{src}到{dest}，{vehicle1}和{vehicle2}的价格对比。",
        "帮我对比{src}到{dest}的{vehicle1}和{vehicle2}。",
        "{src}去{dest}，{vehicle1}跟{vehicle2}哪个便宜？",
        "比较{src}到{dest}的{vehicle1}、{vehicle2}和{vehicle3}的时间。",
        "我想知道{src}到{dest}坐{vehicle1}和{vehicle2}有什么区别。",
        "对比一下{src}到{dest}的{vehicle1}和{vehicle2}的票价。",
        "{src}到{dest}，{vehicle1}和{vehicle2}哪个更方便？",
        "帮我看看{src}到{dest}的{vehicle1}和{vehicle2}怎么选。",
        "比较{date}{src}到{dest}的{vehicle1}和{vehicle2}。"
    ],
    "CANCEL": [
        "取消订单{booking_id}。",
        "我想取消{src}到{dest}的订单，{date}的。",
        "帮我取消预订：{src}到{dest}，{date}。",
        "取消我的{vehicle}票，{src}到{dest}，{date}。",
        "我要退掉{date}从{src}到{dest}的{vehicle}票。",
        "取消订单，{src}到{dest}，{date}。",
        "请帮我取消{booking_id}这个订单。",
        "我想退订{src}到{dest}的{vehicle}，{date}。",
        "取消我的行程：{src}到{dest}，{date}。",
        "退掉{date}{src}到{dest}的{vehicle}票。"
    ]
}
```

这里利用上述两个人工构建的数据来随机组合生成数据集

```python
# generate_data.py
import random
import json
from itertools import product
from entities import CITIES, VEHICLES, DATES, TIMES, SEAT_TYPES

# 加载模板
from templates import TEMPLATES


def get_random_booking_id():
    return f"BK{random.randint(100000, 999999)}"


def bio_tag(text, entities):
    """为文本生成 BIO 标签"""
    tokens = list(text)
    labels = ["O"] * len(tokens)

    for slot_name, value in entities.items():
        if not value:
            continue
        # 处理多值槽位（如 VehicleTypes）
        values = value if isinstance(value, list) else [value]
        for v in values:
            start = text.find(v)
            while start != -1:
                labels[start] = f"B-{slot_name}"
                for i in range(1, len(v)):
                    if start + i < len(labels):
                        labels[start + i] = f"I-{slot_name}"
                start = text.find(v, start + 1)
    return labels


def generate_samples():
    samples = []
    intent_counts = {"QUERY": 600, "BOOK": 600, "COMPARE": 400, "CANCEL": 400}

    for intent, count in intent_counts.items():
        for _ in range(count):
            # 随机选实体
            src = random.choice(CITIES)
            dest = random.choice([c for c in CITIES if c != src])
            vehicle = random.choice(list(VEHICLES.keys()))
            date = random.choice(DATES)
            time = random.choice(TIMES)
            passenger = random.randint(1, 5)
            seat = random.choice(SEAT_TYPES)
            booking_id = get_random_booking_id()

            # 为 COMPARE 选多个交通工具
            if intent == "COMPARE":
                vehicles = random.sample(list(VEHICLES.keys()), k=min(2, len(VEHICLES)))
                template = random.choice(TEMPLATES[intent])
                # 替换 {vehicle1}, {vehicle2}
                text = template.format(
                    src=src, dest=dest, date=date,
                    vehicle1=vehicles[0],
                    vehicle2=vehicles[1] if len(vehicles) > 1 else vehicles[0],
                    vehicle3=random.choice(list(VEHICLES.keys()))
                )
                slots = {
                    "Src": src,
                    "Dest": dest,
                    "Date": date if "date" in template else None,
                    "VehicleTypes": vehicles
                }
            elif intent == "CANCEL":
                text = random.choice(TEMPLATES[intent]).format(
                    src=src, dest=dest, date=date, vehicle=vehicle,booking_id=booking_id
                )
                slots = {"Src": src, "Dest": dest, "Date": date,"booking_id":booking_id}
            else:
                template = random.choice(TEMPLATES[intent])
                text = template.format(
                    src=src, dest=dest, vehicle=vehicle,
                    date=date, time=time,
                    passenger_count=passenger, seat_type=seat
                )
                slots = {
                    "Src": src,
                    "Dest": dest,
                    "VehicleType": vehicle,
                    "Date": date if "date" in template.lower() else None,
                    "Time": time if "time" in template.lower() else None,
                }
                if intent == "BOOK":
                    slots.update({
                        "PassengerCount": str(passenger),
                        "SeatType": seat
                    })

            # 清理 None 值
            slots = {k: v for k, v in slots.items() if v is not None}

            # 生成 BIO 标签
            bio_labels = bio_tag(text, slots)

            samples.append({
                "text": text,
                "intent": intent,
                "slots_bio": bio_labels
            })

    return samples


if __name__ == "__main__":
    samples = generate_samples()
    print(f"生成 {len(samples)} 条样本")

    # 保存为 JSONL
    with open("train_data.jsonl", "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 打印示例
    print("\n示例：")
    for i in range(3):
        print(samples[i])
```

生成的数据集如下：

```json
{"text": "武汉到肇庆的汽车什么时候发车？", "intent": "QUERY", "slots_bio": ["B-Src", "I-Src", "O", "B-Dest", "I-Dest", "O", "B-VehicleType", "I-VehicleType", "O", "O", "O", "O", "O", "O", "O"]}
{"text": "从肇庆到上海的高铁有哪些？", "intent": "QUERY", "slots_bio": ["O", "B-Src", "I-Src", "O", "B-Dest", "I-Dest", "O", "B-VehicleType", "I-VehicleType", "O", "O", "O", "O"]}
{"text": "查宁波到大连的动车班次。", "intent": "QUERY", "slots_bio": ["O", "B-Src", "I-Src", "O", "B-Dest", "I-Dest", "O", "B-VehicleType", "I-VehicleType", "O", "O", "O"]}

```

### bert模型构建

这里主要是模仿官方huggingface中bert不同下游任务模型的实现

```python
from dataclasses import dataclass
from typing import Optional,Union,Tuple,Dict
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers import (
    BertPreTrainedModel,
    BertModel
)
@dataclass
class SequenceAndTokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    intent_logits: Optional[torch.FloatTensor] = None
    slot_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Bert4TextAndTokenClassification(BertPreTrainedModel):
    def __init__(self, config, seq_num_labels, token_num_labels):
        super().__init__(config)
        self.seq_num_labels = seq_num_labels
        self.token_num_labels = token_num_labels
        self.config = config

        self.bert = BertModel(config)
        self.sequence_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, seq_num_labels),
        )
        self.token_classification = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, token_num_labels),
        )
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            intent_labels: Optional[torch.Tensor] = None,
            slot_labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceAndTokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]  # [CLS]
        sequence_output = outputs[0]  # [batch, seq_len, hidden]

        intent_logits = self.sequence_classification(pooled_output)
        slot_logits = self.token_classification(sequence_output)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            # 意图：单标签分类 → CrossEntropyLoss
            intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)
            # 槽位：序列标注 → CrossEntropyLoss (ignore_index=-100)
            slot_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                slot_logits.view(-1, self.token_num_labels), slot_labels.view(-1)
            )
            loss = intent_loss + 10*slot_loss

        if not return_dict:
            output = (intent_logits, slot_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceAndTokenClassifierOutput(
            loss=loss,
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```


### 加载数据集

```python
intents = ['QUERY', 'BOOK', 'COMPARE', 'CANCEL']
intents2id = {intent: id for id, intent in enumerate(intents)}
id2intents = {id: intent for intent, id in intents2id.items()}
slots = ['O','B-Time','I-Time', 'B-SeatType','I-SeatType', 'B-VehicleTypes','I-VehicleTypes', 'B-Dest','I-Dest',
         'B-booking_id','I-booking_id', 'B-VehicleType','I-VehicleType',
         'B-Date','I-Date', 'B-PassengerCount', 'B-Src','I-Src',
            ]
slots2id = {slot: id for id, slot in enumerate(slots)}
id2slots = {id: slot for slot, id in slots2id.items()}
tokenizer = BertTokenizerFast.from_pretrained("../../bert-base-chinese")
def load_data(
    file_path: str = "train_data.jsonl",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 JSONL 文件加载数据，划分为训练集和测试集，并进行数据校验。

    Args:
        file_path: JSONL 文件路径
        test_size: 测试集比例
        random_state: 随机种子，确保可复现

    Returns:
        (train_df, test_df): 划分后的训练集和测试集 DataFrame

    Raises:
        AssertionError: 如果数据中的意图或槽位标签与预定义不一致
        FileNotFoundError: 如果文件不存在
        json.JSONDecodeError: 如果 JSON 格式错误
    """
    # 1. 加载数据
    samples = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    sample = json.loads(line)
                    # 可选：校验必要字段
                    if not all(k in sample for k in ["text", "intent", "slots_bio"]):
                        raise ValueError(f"Missing keys in line {line_num}")
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if not samples:
        raise ValueError("No valid samples loaded from the file.")

    # 2. 转为 DataFrame 并校验标签
    df = pd.DataFrame(samples)

    # 校验意图标签
    data_intents: Set[str] = set(df["intent"].unique())
    expected_intents: Set[str] = set(intents)
    if data_intents != expected_intents:
        missing = expected_intents - data_intents
        extra = data_intents - expected_intents
        msg = []
        if missing:
            msg.append(f"Missing intents in data: {missing}")
        if extra:
            msg.append(f"Unexpected intents in data: {extra}")
        raise AssertionError("Intent label mismatch!\n" + "\n".join(msg))

    # 校验槽位标签
    all_slot_tags: List[str] = [tag for tags in df["slots_bio"] for tag in tags]
    data_slots: Set[str] = set(all_slot_tags)
    expected_slots: Set[str] = set(slots)
    if data_slots != expected_slots:
        missing = expected_slots - data_slots
        extra = data_slots - expected_slots
        msg = []
        if missing:
            msg.append(f"Missing slot tags in data: {missing}")
        if extra:
            msg.append(f"Unexpected slot tags in data: {extra}")
        raise AssertionError("Slot label mismatch!\n" + "\n".join(msg))

    # 3. 划分数据集
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["intent"]  # 按意图分层抽样，保证分布一致
    )

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    return train_df, test_df
```

这里是从json文件中读取并转为了pandas的格式，一个是pandas在pycharm中调试观察很方便，在再一个是pandas转其他格式也很方便。


### pandas数据集--> huggingface数据集

```python
    train_dataset,test_dataset = load_data()

    model = Bert4TextAndTokenClassification.from_pretrained("../../bert-base-chinese", seq_num_labels=len(intents),
                                              token_num_labels=len(slots))

    train_dataset = Dataset.from_pandas(train_dataset)
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names  # 删除原始列
    )

    test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=test_dataset.column_names
    )
```

这里运用到了对齐函数，主要是槽位识别这里需要将token和label进行对齐

```python
def tokenize_and_align_labels(examples):
    # 告诉 tokenizer 输入已是字符列表
    tokenized_inputs = tokenizer(
        [list(text) for text in examples["text"]],  # 按字切分
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128
    )

    intent_labels = [intents2id[intent] for intent in examples["intent"]]

    slot_labels = []
    for i, label in enumerate(examples["slots_bio"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], padding → -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 首次出现的 token → 使用原始标签
                label_ids.append(slots2id[label[word_idx]])
            else:
                # subword → -100（忽略）
                label_ids.append(-100)
            previous_word_idx = word_idx
        slot_labels.append(label_ids)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "intent_labels": intent_labels,
        "slot_labels": slot_labels,
    }
```

这里我们已经成功将数据集转为huggingface中trainer可以用来训练的格式

### 设置训练参数

```python
    training_args = TrainingArguments(
        output_dir='./Results',
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=20,
        eval_strategy="epoch",  # 修正
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="wandb",
        run_name="my-Results-run",
        logging_strategy="steps",
        remove_unused_columns=False,  # 👈 关键！保留自定义列
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

这里主要是需要注意`training_args`的最后一个参数，必须设置为False，不然huggingface会默认删除掉不认识的列。

然后这里在训练过程中有评估函数

```python

```
