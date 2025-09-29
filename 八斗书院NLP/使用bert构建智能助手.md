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

```
