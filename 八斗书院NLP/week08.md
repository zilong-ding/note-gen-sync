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

AI Agent(或简称为Agent)是建立在大语言模型之上的智能应用，是将人工智能与特定场景深度结合的重要方式。Agent模仿人类“思考-行动-观察”的规划模式，具备自主思考和自主决策的能力，能够适应环境的变化，自主学习和改进，完成用户设定的目标。

与大语言模型的对话应用不同，Agent的突出特点是主动性，在行为上表现为多步操作、多角色会话、多轮迭代、反复修正答案以及调用外部资源的能力。

![2025-10-13_11-09.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/28da65f8-9b1b-45f0-8122-e124acc7d45a.jpeg)

LLM 智能体系统划分为三大核心组件：

1. 规划（Planning）
   任务分解：通过 Chain of Thought（CoT）、Tree of Thoughts（ToT）等方法，将复杂任务拆解为可管理的子目标。
   外部规划器集成：如 LLM+P 方法，利用经典规划语言（PDDL）将长期规划任务外包给外部规划器。
   自我反思（Self-Reflection）：
   ReAct：结合推理（Thought）与行动（Action），通过与环境交互（如调用 API）提升决策质量。
   Reflexion：引入动态记忆与反思机制，在失败后生成改进策略并重试。
   Chain of Hindsight（CoH） 与 Algorithm Distillation（AD）：利用历史反馈或强化学习轨迹训练模型，使其具备从过去经验中学习并持续改进的能力。
2. 记忆（Memory）
   短时记忆：对应 Transformer 的上下文窗口，用于 in-context learning。
   长时记忆：通过外部向量数据库（如 FAISS、HNSW）实现无限容量的信息存储与快速检索。
   检索机制：采用最大内积搜索（MIPS）和近似最近邻（ANN）算法（如 LSH、ANNOY、ScaNN）高效访问相关记忆。
3. 工具使用（Tool Use）
   LLM 可调用外部工具（如计算器、搜索引擎、代码执行环境、API）以弥补其知识或能力局限。
   典型框架包括：
   MRKL：模块化架构，LLM 作为路由器调用专家模块。
   Toolformer / TALM：通过微调让模型学会插入 API 调用。
   HuggingGPT：利用 ChatGPT 规划任务，并调度 HuggingFace 上的专用模型执行。
   API-Bank：提供包含 53 个 API 的评测基准，评估智能体在“是否调用”“如何检索”“如何规划多步调用”三个层级的能力。

主要挑战
上下文长度有限：限制历史信息、指令细节和工具交互的完整表达。
长期规划能力弱：难以动态调整计划以应对意外错误。
自然语言接口不可靠：LLM 输出格式不稳定，需大量后处理与解析逻辑。



### 一、核心定义与分类

Agentic Systems（智能体系统）：泛指利用 LLM 动态决策、调用工具、完成任务的系统。
区分为两类：
Workflows（工作流）：由预定义代码路径编排 LLM 与工具，确定性强、可预测。
Agents（智能体）：由 LLM 自主规划、动态调用工具、自我调整流程，灵活性高但成本与风险也更高。

### 二、是否使用智能体？——关键权衡

优先尝试简单方案：单次 LLM 调用 + 检索（RAG）+ 少样本提示，往往已足够。
仅当任务复杂、步骤不可预知、需多轮交互时，才考虑引入智能体。
智能体的代价：更高延迟、更高成本、错误可能累积。

### 三、六大核心构建模式（Building Blocks）

1. Augmented LLM（增强型 LLM）
   基础单元：LLM + 工具 + 检索 + 记忆。
   关键：为 LLM 提供清晰、易用、文档完善的工具接口（即“Agent-Computer Interface, ACI”）。
2. Prompt Chaining（提示链）
   将任务分解为固定顺序的子任务，逐个 LLM 调用。
   适用：任务结构清晰、可线性分解（如“写大纲 → 审核 → 写全文”）。
3. Routing（路由）
   根据输入类型，动态选择下游处理路径或专用模型。
   适用：多类别任务（如客服：退款 / 技术问题 / 一般咨询）。
4. Parallelization（并行化）
   Sectioning：并行处理独立子任务（如多文件代码修改）。
   Voting：多次生成 + 投票/聚合，提升鲁棒性（如多角度安全审查）。
5. Orchestrator-Workers（协调者-工作者）
   一个“主 LLM”动态拆解任务、分配给“工作 LLM”，再汇总结果。
   适用：子任务数量与类型无法预知（如复杂代码变更）。
6. Evaluator-Optimizer（评估-优化循环）
   一个 LLM 生成，另一个 LLM 评估并反馈，迭代优化输出。
   适用：有明确评价标准、且人类反馈能提升质量的任务（如文学翻译、深度搜索）。


### 四、真正的“Agent”：自主性与环境交互

Agent 的核心特征：

自主规划（plan）
调用工具（act）
基于环境反馈调整（observe → reflect）
必要时请求人类介入
成功关键：
工具设计要“对 LLM 友好”（避免复杂格式、提供示例、使用绝对路径等）
充分测试（在沙盒中验证，防止错误累积）
设置停止条件（如最大迭代次数）
Agent 本质就是一个带工具调用和反馈循环的 LLM，实现可以非常简洁。

### 五、实用建议与原则

从简单开始：不要过早引入框架或复杂架构。
慎用框架：LangGraph、Rivet 等虽方便，但会增加调试难度；建议先直接调用 LLM API。
重视 ACI（Agent-Computer Interface）：
工具参数命名清晰
提供使用示例和边界说明
避免要求 LLM 处理易错格式（如 JSON 内嵌代码、diff 行号计算）
“像为初级工程师写文档一样写工具说明”
透明化决策过程：让 Agent 显式输出其推理和计划，便于调试与信任建立。


### agents 框架

#### Devika

Devika是一个AI智体软件，用于软件辅助开发场景，可以理解为人类输入指令，Agent将指令分解为操作步骤，并自主制订计划、编写代码以实现给定的目标。Devika利用大语言模型、规划和推理算法以及Wb浏览能力来智能化地开发软件。它改变传统构建软件的方式，可以在最少的人工指导下承担复杂的开发任务，其能力包括创建新功能、修复错误，甚至从头开始开发整个项目。

![2025-10-13_16-23.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/da8270aa-02f6-4169-bb5f-14809d40495f.jpeg)

#### TaskWeaver

TaskWeaver是一个以代码为中心的智能代理框架，用于无缝规划和执行数据分析任务。这个创新框架通过代码片段解释用户请求，并高效协调各种插件（以函数形式存在）来执行数据分析或工作流自动化任务。√有状态的对话-TaskWeaver设计为支持有状态的对话，这意味着你可以在多个聊天回合中与内存中的数据进行交互。√代码验证-TaskWeaver设计为在执行前验证生成的代码。它可以检测生成代码中的潜在问题并自动修复它们。

![2025-10-13_16-24.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/a6a3d8b4-0a97-464c-8777-8f8e0df85d43.jpeg)



### OpenAI Agents SDK
