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
