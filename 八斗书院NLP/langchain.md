# langchain

LangChain是一个用于开发由语言模型区动的应用程序的框架。我们相信，最强大和不同的应用程序不仅将通过API调用语言模型：

组件：LangChain为处理语言模型所需的组件提供模块化的抽象。LangChain还为所有这些抽象提供了实现的集合。这些组件旨在易于使用，无论您是否使用LangChain框架的其余部分。

用例特定链：链可以被看作是以特定方式组装这些组件，以便最好地完成特定用例。这旨在成为一个更高级别的接口，使人们可以轻松地开始特定的用例。这些链也旨在可定制化。

## langchain中的组件结构

![2025-10-13_10-01.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/b8e6951b-cec3-4d84-bda7-c3ec6932aaac.jpeg)

#### 1️⃣ Models（模型层） —— 黄色区域

这是所有 LLM 应用的基础，负责与语言模型交互。

LLMs：基础大语言模型接口（如 OpenAI, HuggingFaceHub），输入文本 → 输出文本。
Chat Models：专为对话设计的模型接口（如 ChatOpenAI），支持角色消息（system/user/assistant）。
Embeddings：用于文本向量化，是检索系统（RAG）的核心，如 OpenAIEmbeddings, HuggingFaceEmbeddings。
✅ 注意：虽然 Embeddings 不是“生成模型”，但它是 LangChain 中与模型紧密关联的重要组件，常用于检索增强。

#### 2️⃣ Prompts（提示工程） —— 粉红色区域

用于构造、管理和优化发送给模型的提示词。

templates：PromptTemplate，支持变量插值（如 "Hello {name}"）。
few-shot examples：少样本学习示例，提高模型表现。
example selector：动态选择最相关的 few-shot 示例（如基于语义相似度）。
（省略号表示还有其他提示相关组件，如 ChatPromptTemplate、OutputParser 等）
💡 Prompt 是控制 LLM 行为的关键，好的 prompt 能显著提升输出质量。

#### 3️⃣ Indexes（索引与检索） —— 蓝色区域

用于构建和管理外部知识库，支撑 RAG（Retrieval-Augmented Generation）应用。

document loaders：加载各种格式文档（PDF、网页、CSV 等）。
text splitters：切分长文本为 chunk（如 RecursiveCharacterTextSplitter）。
vectorstores：存储向量的数据库（FAISS, Chroma, Pinecone, Weaviate）。
retrievers：封装检索逻辑，根据 query 返回最相关的文档块。
（省略号表示更多检索策略或工具）
🔍 这部分是 LangChain 实现“知识增强”的核心，让 LLM 可以引用外部信息回答问题。

#### 4️⃣ Memory（记忆） —— 紫色区域

用于在链或代理执行过程中保存状态（如对话历史、中间结果）。

ConversationBufferMemory：缓存全部对话历史。
ConversationBufferWindowMemory：只保留最近 N 条对话。
ConversationTokenBufferMemory：按 token 数限制历史长度。
ConversationSummaryMemory：用 LLM 总结历史，节省上下文。
VectorStoreBackedMemory：结合向量数据库实现长期记忆检索。
（其他如 EntityMemory 等未列出）
🧠 Memory 让应用具备“上下文感知”能力，是构建多轮对话系统的必备组件。

#### 5️⃣ Chains（链） —— 灰色区域

将多个组件串联成可复用的工作流。

LLMChain：最基础链，组合 Prompt + LLM。
RouterChain：根据输入路由到不同子链。
SimpleSequentialChain / SequentialChain：按顺序执行多个链。
TransformChain：对输入/输出做转换处理。
（省略号表示更多高级链类型，如 RetrievalQAChain、ConversationalRetrievalChain 等）
🔄 Chain 是 LangChain 的“粘合剂”，让开发者可以像搭积木一样组合功能。

#### 6️⃣ Agents（智能体） —— 绿色虚线框区域

让 LLM 具备“自主决策 + 工具调用”能力，实现复杂任务自动化。

主要 Agent 类型：
Conversational Agents：支持多轮对话 + 工具调用。
OpenAI Functions：调用 OpenAI 的 function calling 功能。
Self Ask with Search：先自我提问，再搜索答案。
Action Agents：通用术语，指能执行动作的 agent（如 ReAct）。
Plan-and-execute agents：先规划步骤，再逐步执行（适合复杂任务）。
🤖 Agent = LLM + Tools + Reasoning Loop。它让模型从“被动响应”变成“主动解决问题”。

#### 7️⃣ 隐含组件：Callbacks & Utilities（未在图中显式列出）

虽然图中没有单独列出，但在实际开发中非常重要：

Callbacks：用于日志、监控、流式输出、调试（如 LangSmith 集成）。
Tools：Agent 调用的具体函数（如搜索、计算、数据库查询）。
Integrations：第三方服务集成（Wikipedia, Google Search, SQL, Zapier 等）。

## 概念

Chat models（聊天模型）：通过聊天 API 暴露的大语言模型（LLMs），以消息序列作为输入，并输出一条消息。

Messages（消息）：聊天模型中通信的基本单元，用于表示模型的输入和输出。

Chat history（聊天历史）：一段对话，表示为消息序列，用户消息和模型回复交替出现。

Tools（工具）：一个带有相关 schema 的函数，该 schema 定义了函数的名称、描述以及所接受的参数。

Tool calling（工具调用）：一种聊天模型 API，它除了接收消息外，还接受工具 schema 作为输入，并在输出消息中返回这些工具的调用结果。

Structured output（结构化输出）：一种使聊天模型以结构化格式（例如符合给定 schema 的 JSON）进行响应的技术。

Memory（记忆）：关于对话的信息，被持久化存储，以便在未来的对话中使用。

Multimodality（多模态）：处理不同形式数据的能力，例如文本、音频、图像和视频。

Runnable interface（可运行接口）：LangChain 组件和 LangChain 表达式语言（LCEL）所基于的基础抽象。

Streaming（流式处理）：LangChain 的流式 API，用于在结果生成过程中实时返回。

LangChain Expression Language (LCEL)（LangChain 表达式语言）：一种用于编排 LangChain 组件的语法，尤其适用于较简单的应用程序。

Document loaders（文档加载器）：将一个数据源加载为文档列表。

Retrieval（检索）：信息检索系统根据查询从数据源中检索结构化或非结构化数据。

Text splitters（文本分割器）：将长文本分割为更小的块，以便单独索引，从而实现细粒度检索。

Embedding models（嵌入模型）：将文本或图像等数据表示为向量空间中的向量的模型。

Vector stores（向量存储）：用于存储向量及其关联元数据，并支持高效搜索。

Retriever（检索器）：一个组件，用于根据查询从知识库中返回相关文档。

Retrieval Augmented Generation (RAG)（检索增强生成）：一种通过将语言模型与外部知识库结合来增强其能力的技术。

Agents（智能体）：使用语言模型来选择要执行的一系列动作。智能体可通过工具与外部资源交互。

Prompt templates（提示模板）：用于将模型“提示”（通常是一系列消息）中的静态部分提取出来的组件。有助于序列化、版本控制和复用这些静态部分。

Output parsers（输出解析器）：负责将模型的输出转换为更适合下游任务的格式。在工具调用和结构化输出广泛可用之前，输出解析器尤为重要。

Few-shot prompting（少样本提示）：一种通过在提示中提供少量任务示例来提升模型性能的技术。

Example selectors（示例选择器）：用于根据给定输入从数据集中选择最相关的示例。在少样本提示中，示例选择器用于为提示选择示例。

Async programming（异步编程）：使用 LangChain 进行异步上下文编程所需掌握的基础知识。

Callbacks（回调）：在内置组件中执行自定义辅助代码的机制。LangChain 中的回调用于流式输出 LLM 结果、追踪应用程序的中间步骤等。

Tracing（追踪）：记录应用程序从输入到输出所执行步骤的过程。对于调试和诊断复杂应用中的问题至关重要。

Evaluation（评估）：评估 AI 应用性能和有效性的过程。包括根据一组预定义标准或基准测试模型响应，以确保其达到所需质量标准并实现预期目标。该过程对构建可靠应用至关重要。

Testing（测试）：验证集成或应用程序的某个组件是否按预期工作的过程。测试对于确保应用行为正确、以及确保代码变更不会引入新 bug 至关重要。
