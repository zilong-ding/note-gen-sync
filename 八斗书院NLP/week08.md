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

## AI agents
