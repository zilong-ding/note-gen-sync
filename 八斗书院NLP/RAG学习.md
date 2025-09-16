# RAG学习

## RAG定义

在自然语言处理领域，大型语言模型(LLM)如GPT-3、BERT等已经取得了显著的进展，它们能够生成连贯、自然的文本，回答问题，并执行其他复杂的语言任务。然而，这些模型存在一些固有的局限性，如“模型幻觉问题”、“时效性问题”和“数据安全问题”。为了克服这些限制，检索增强生成(RAG)技术应运而生。

![2025-09-16_08-15.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/01ee858c-5483-44ac-b119-8673cc24b5fb.jpeg)

RAG技术结合了大型语言模型的强大生成能力和检索系统的精确性。它允许模型在生成文本时，从外部知识库中检索相关信息，从而提高生成内容的准确性、相关性和时效性。这种方法不仅增强了模型的回答能力，还减少了生成错误信息的风险。

## RAG常见流程

![政企问答项目RAG流程](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/6caead23-9836-469a-b35b-9f0f42f05802.jpeg)

![个人知识库RAG流程](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/f4cdcab4-499a-465b-8a73-a064d3f94d4a.jpeg)

从上述流程中我们可以看到RAG中常见功能部分有文件解析功能、切片划分功能、向量嵌入功能、查询改写功能、路由功能、数据库、多路召回功能、排序功能、大模型回答功能。


## RAG中功能模块

### 文件解析功能

现实生活中我们常见的文件格式有文本(txt)，word文档，pdf文档，md文档。文本解析就是将文档中的内容（文字，图片，表格等信息）提取出来。这里比较困难的是pdf文档解析。

为了解析文件我们可以使用如下技术：

* pdfplumber：pdf文件文本提取和图片提取
* 深度学习的方法：ocr等技术，paddle-ocr，pdf2md
* 多模态大模型的方法：qwen-vl等

![2025-09-16_08-41.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/347f5f4b-67fd-47ed-8c83-a92cba387a00.jpeg)

### 切片划分功能

参考文章：https://zhuanlan.zhihu.com/p/1912878600853623201

RAG技术中划分chunk是为了更好地适应大模型的处理能力，提高检索和生成的效率和准确性，以及优化内容的相关性。

**从大模型输入角度模型**

> 在预训练过程中通常有上下文长度限制，如果输入的文本超过这个长度，超出部分会被丢弃，从而影响模型的性能表现。因此，需要将长文档分割成小块，以适应模型的上下文窗口。

**从语义表示的差异性角度**

> 长文档中各个片段的语义可能存在较大差异，如果将整个文档作为一个整体进行知识检索，会导致语义杂揉，影响检索效果。将长文档切分成多个小块，可以使得每个小块内部表意一致，块之间表意存在多样性，从而更充分地发挥知识检索的作用。

**划分Chunk的注意事项**

在进行chunk划分时，需要保留每个chunk的原始内容来源信息，这包括但不限于：

> 页面编号：记录每个chuk来自文档的哪一页，有助于在需要时快速定位原始信息。
> 文档标识：为每个文档分配一个唯一的标识符，以便在检索时能够准确引用。
> 版本控制：如果文档会更新，记录cunk对应的文档版本，确保内容的一致性和准确性。

随着时间的推移，原始文档可能会更新或修改（可能使用新的文档处理方法重新划分chuk)。因此，在划分chunk时需要考虑：

> 更新策略：制定一个清晰的策略，以确定何时以及如何更新chunk,以保持信息的最新性。
> 版本兼容性：确保新l旧版本的chunk能够兼容，或者能够明确区分不同版本的chunk。

此外chunk之间的顺序可能对理解整个文档至关重要：

> 顺序标记：为每个chuk分配顺序编号或时间戳，以保持它们的逻辑顺序。
> 顺序检索：在检索时，确保能够按照正确的顺序检索和展示chunk。

![v2-72acb98d3c9ec6806011f9ca3b802e17_b.webp](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/c7c2610c-fc5c-41d9-94b0-b7dc2d60a4a4.webp)

**常用切片划分的策略如下**：

1. 固定大小：设置100个字符为一个chunk
2. 滑动窗口：在固定大小的情况下加入overlap
3. 按照文档结构：先划分为段落，再划分为句子
4. 递归分块：先划分章节、再划分段落、句子
5. 语义分块：将含义相近的句子划分到一个chunk
6. 大模型分块：写一个提示词，让大模型分块；
7. Late chunking:提取token特征，对chunk tokensi进行pooling;


#### 固定大小切分

将文档按照预设的字符数、词数或句子数进行等间隔划分。例如每段包含500个字符或5个句子。该方法实现简单，但容易打断语义边界，可能导致上下文缺失或内容重复。

![v2-4defad599bbfd1773c082c815aa1c2df_1440w.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/7461faf6-59e8-4cf6-b263-85c791a2b265.jpeg)





#### 文档结构分块

利用原始文档的结构信息（如HTML标签、Markdown标题、PDF书签、Word段落等）进行切分。比如以章节、小标题、列表项为边界进行分块。

![v2-bc06c016ad0f247e5757bbe1c133356d_b.webp](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/74cc9fba-c350-4b30-9641-4c8294737190.webp)

这种方式在处理格式规范的文档（如手册、报告）时效果尤为突出。

![v2-01afd8700d9a93b5c56a878b4f484247_1440w.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/85224810-5181-400d-b5c2-f3497ab03bae.jpeg)


#### 层次递归分块

![2025-09-16_08-50.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/79b08918-afc9-4e22-a7ed-13f45e3aeadd.jpeg)

在保持固定长度的同时，尝试以语义结构（如段落、句子、标点）为边界递归地切分文本。若段落太长无法容纳于块中，则再递归切分为句子，直到满足长度要求。

![v2-080935082e37fdabb9fee9db9279fba5_b.webp](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/0d84a105-9815-4f58-8cf5-a5c4dce10fcd.webp)

![v2-d7ba9e86e42de2764170cce7ce518e62_1440w.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/72eae67c-873b-4eaf-a02d-628311001069.jpeg)


#### 语义分块

通过自然语言处理技术（如句向量相似度、话题建模等）判断文本语义的边界，在语义上自然断句。

![2025-09-16_09-00.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/ca3254e6-8192-46a4-8e5a-7369372b3444.jpeg)

![v2-e0f65749b7f6a139182a0c6209b5ead7_b.webp](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/1498a369-3ba9-47e6-a0b9-52841036cd9b.webp)

以向量相似度为例，将句子或段落转换为向量，通过计算相邻句段的余弦相似度，如果判断两个段落语义上属于同一单元，那么就进行合并。

![v2-0c546f112ac1bb72b9ad5a7bf6f0f292_1440w.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/89969f67-e858-4a2d-b2af-5685471d3dde.jpeg)

这种方式能提升分块的语义连贯性，适用于逻辑紧密的文章，但计算代价较高，依赖模型质量。

#### 大模型分块

构建提示词，借助大模型输入文本长的特点对长文本进行切分。这里类似是先对长文本进行滑动窗口切块，然后借助大模型进行分块。

![v2-de87a53022dbabb18b7598cae91f0c78_b.webp](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/324665ef-e2a1-43b5-9365-9e1dfcbed970.webp)

借助大语言模型来“理解”文档内容并主动划定分块边界。例如，提示模型判断哪些段落构成完整的语义单元，或根据任务需求生成最佳的分块方案。这种方式智能程度高，但计算成本也相对较大，适合高精度应用场景。

#### Late chunking

![2025-09-16_08-56.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/d3420337-732c-40aa-a4cd-2b0f6ab0b564.jpeg)

#### 评价Chunk划分方法

**分块归因(Chunk Attribution)**

> 分块归因用于评估每个检索到的分块是否对模型的响应产生了影响。它使用一个二元指标，将每个分块归类为“有影响”（(Attributed)或“无影响”(Not Attributed)。分块归因与分块利用率(Chunk Utilization)密切相关，其中归因确定分块是否对响应产生了影响，而利用率衡量分块文本在影响中所占的程度。只有被归为“有影响”的分块才能有大于零的利用率得分。

**分块利用率(Chunk Utilization)**

> 分块利用率衡量每个检索到的分块中有多少文本对模型的响应产生了影响。这个指标的范围是 0到1，其中1表示整个分块影响了响应，而较低的值，如0.5，则表示存在“无关”文本，这部分文本并未对响应产生影响。分块利用率与分块归因紧密相关，归因确定分块是否影响了响应，而利用率衡量分块文本在影响中所占的部分。只有被归为“有影响”的分块才能有大于零的利用率得分。



#### 分块可视化工具

https://chunkviz.up.railway.app/

### 向量嵌入功能

参考文章：https://zhuanlan.zhihu.com/p/1912910452339484544

**Embedding（嵌入向量）** 是将文字、图片、语音等“人类语言”**转换为**“计算机语言”的关键一步。它的作用，是把一句话或者一个词，变成一串可以进行数学运算的数字向量，让模型能“理解”我们在说什么。

![v2-240153b1807c7f36748478856aec8365_1440w.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/97aad5ad-8142-45c9-8553-648d9f0ce662.jpeg)


#### 选择嵌入模型

在 RAG 系统中，嵌入模型（Embedding Model）就像是用户与知识库之间的翻译官——它决定了“你在说什么”和“它能不能听懂”。

选择一个合适的嵌入模型，能大幅提升检索质量与上下文匹配度。选得好，模型如虎添翼，问啥答啥；选不好，可能“查到不对题，答得更离谱”。

以下是选型时需要重点考虑的几个维度


| 考量维度       | 说明                                                               |
| -------------- | ------------------------------------------------------------------ |
| 语义表现力     | 能否正确捕捉句子的含义？是否支持中文、多 语言？                    |
| 模型大小/效率  | 越大越准？不一定！推理速度、GPU/CPU占 用也是关键                   |
| 训练目标       | 是面向“检索”训练的（如BGE+)，还是面 向“生成”或“通用”训练的？ |
| 向量归一化     | 是否适合FAISS+等向量库索引（部分模型需 显式归一化)                 |
| 开源/闭源      | 是否可部署本地？是否支持商用？                                     |
| 社区支持与文档 | 模型活跃度越高，调试与优化越方便                                   |


#### 主流嵌入模型

以下是一些主流且表现优秀的嵌入模型，涵盖中英双语、轻量级部署、本地化支持等需求。

**中文 & 多语言方向**


| 模型名称       | 简介与特点                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------- |
| BGE (BAAI)     | 北京智源开源的检索导向模型，支持中文/英 文，<br />带bge-base-zh,bge-m3等版本，性能与 速度兼顾。 |
| E5 系列+       | 多语言嵌入模型（包括e5-base,e5-large)， 适用于检索任务，<br />广泛支持中英文句子匹配。          |
| GTE系列+       | 百度提出的 GTE 模型（如 gte-base)，表现稳 定、部署友好，<br />适合中文问答和文档检索。          |
| text2vec 系列+ | 来自 HuggingFace 的中文句向量模型，<br />如 shibing624/text2vec-base-multilingual, 易 用性高。  |

**英文或通用方向**


| 模型名称         | 简介与适用场景                                                                      |
| ---------------- | ----------------------------------------------------------------------------------- |
| MiniLM+ / MPNet+ | HuggingFace SentenceTransformers 库的经典嵌入模型，<br />轻量快速、适合低资源场景。 |
| Instructort      | 支持带任务说明的嵌入（如"Representthe query for retrieval: xx")，效果优秀。         |
| OpenAl Adat      | GPT体系内置嵌入模型（如text- embedding-ada-002)，<br />闭源但商用表现稳定 强劲。    |
| Cohere Embed+    | 专注于“可控语义检索”的服务型模型，API 提供简单，商用接口友好。                    |

如果不知道选哪个，建议：

* 小模型部署快，适合原型验证（如 `bge-small-zh`）
* 大模型更准，适合上线产品（如 `bge-large-zh-v1.5`）
* 想本地部署？就用 BGE、E5、GTE
* 要省心云服务？那就试试 OpenAI Ada、Cohere



### 查询增强功能

#### Query预处理方法

1. 引导进行二次提问、引导到已知的提问；
2. Query Rewriting(查询重写)：将用户的提问改写一下，转换多个提问；
3. Step-Back Strategy(后退一步策略)：让大模型生成一个更加抽象、更加基础的问题
4. Hypothetical Document Embeddings(HyDE)
5. 拒绝回答、特定格式回答；

#### 二次提问


#### 查询重写


#### 后退一步策略



#### HYDE


#### 拒绝回答



### 路由功能


### 数据库


### 多路召回功能


### 重排序


### 大模型功能
