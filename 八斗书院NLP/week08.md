# AI大模型：NEO4J,NebulaGraph,langchain,AI agents

## neo4j

Neo4是一种图数据库管理系统，它专注于处理图形数据模型。图数据库是一种用于存储和查询图形结构的数据库，其中数据以节点和边的形式表示，节点表示实体，边表示实体之间的关系。

Neo4j使用一种名为Cypher的声明性查询语言，这种语专门设计用于图数据库。Cypheri语法简洁，易于理解，许用户以自然的方式查询和操作图数据。

Neo4j提供ACID(原子性、一致性、隔离性和持久性)务支持，确保数据的完整性和一致性。

![2025-10-13_09-55.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/1b98a6c2-5418-4af0-9856-2f12f77fba0c.jpeg)

## NebulaGraph

NebulaGraph:是一个分布式、可扩展、高性能的图数据库，旨在有效地存储和检索庞大的图网络中的信息。图数据库，如NebulaGraph,专注于通过在标记的属性图中将数据表示为顶点（节点）和边缘（关系）来管理实体之间的关系。顶点和边缘都可以附有属性，并且顶点可以带有一个或多个标签。

NebulaGraph支持流行的编程语言，如Java、Python、 C++和G0,旨在为开发人员提供友好的使用体验。其他客户端库正在开发中，以扩展语言支持。 

NebulaGraph查询语言(nGQL)是一种声明性的、兼容 OpenCypher的文本查询语言，易于理解和使用，用于查询图数据。

![2025-10-13_09-54.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/821c8c71-56d0-42a5-af46-402ec9a7afa6.jpeg)

## langchain

LangChain是一个用于开发由语言模型区动的应用程序的框架。我们相信，最强大和不同的应用程序不仅将通过API调用语言模型：

组件：LangChain为处理语言模型所需的组件提供模块化的抽象。LangChain还为所有这些抽象提供了实现的集合。这些组件旨在易于使用，无论您是否使用LangChain框架的其余部分。

用例特定链：链可以被看作是以特定方式组装这些组件，以便最好地完成特定用例。这旨在成为一个更高级别的接口，使人们可以轻松地开始特定的用例。这些链也旨在可定制化。

### langchain中的组件结构

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

## 提示词工程

![2025-10-13_11-03.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/9615c922-0382-4026-8153-c11d7da6648f.jpeg)


| Zero-shot Prompting              | Few-shot Prompting            | Chain-of-Thought L Prompting   |
| -------------------------------- | ----------------------------- | ------------------------------ |
| Meta Prompting                   | Self-Consistency              | Generate Knowledge Prompting   |
| Prompt Chaining                  | Tree of Thoughts              | Retrieval Augmented Generation |
| Automatic Reasoning and Tool-use | Automatic Prompt Engineer     | Active-Prompt                  |
| Directional Stimulus Prompting   | Program-Aided Language Models | ReAct                          |
| Reflexion                        | Multimodal CoT                | Graph Prompting                |

## AI agents

A!Agent(或简称为Agent)是建立在大语言模型之上的智能应用，是将人工智能与特定场景深度结合的重要方式。Agent模仿人类“思考-行动-观察”的规划模式，具备自主思考和自主决策的能力，能够适应环境的变化，自主学习和改进，完成用户设定的目标。

与大语言模型的对话应用不同，Agent的突出特点是主动性，在行为上表现为多步操作、多角色会话、多轮迭代、反复修正答案以及调用外部资源的能力。
