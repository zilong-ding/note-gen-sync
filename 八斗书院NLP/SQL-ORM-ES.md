# SQL-ORM-ES学习

## SQL

SQL（Structured Query Language）是一种用于管理和操作关系数据库的标准语言，包括数据查询、数据插入、数据更新、数据删除、数据库结构创建和修改等功能。

![](https://www.runoob.com/wp-content/uploads/2013/09/SQL.png)

### 数据库表

一个数据库通常包含一个或多个表，每个表有一个名字标识（例如:"Websites"），表包含带有数据的记录（行）。

在本教程中，我们在 MySQL 的 RUNOOB 数据库中创建了 Websites 表，用于存储网站记录。

我们可以通过以下命令查看 "Websites" 表的数据

```sql
mysql> use RUNOOB;
Database changed

mysql> set names utf8;
Query OK, 0 rows affected (0.00 sec)

mysql> SELECT * FROM Websites;
+----+--------------+---------------------------+-------+---------+
| id | name         | url                       | alexa | country |
+----+--------------+---------------------------+-------+---------+
| 1  | Google       | https://www.google.cm/    | 1     | USA     |
| 2  | 淘宝          | https://www.taobao.com/   | 13    | CN      |
| 3  | 菜鸟教程      | http://www.runoob.com/    | 4689  | CN      |
| 4  | 微博          | http://weibo.com/         | 20    | CN      |
| 5  | Facebook     | https://www.facebook.com/ | 3     | USA     |
+----+--------------+---------------------------+-------+---------+
5 rows in set (0.01 sec)
```

### SQL 语句

SQL 对大小写不敏感：SELECT 与 select 是相同的

SQL 语句后面的分号？
某些数据库系统要求在每条 SQL 语句的末端使用分号。

分号是在数据库系统中分隔每条 SQL 语句的标准方法，这样就可以在对服务器的相同请求中执行一条以上的 SQL 语句。

在本教程中，我们将在每条 SQL 语句的末端使用分号。

下面的 SQL 语句从 "Websites" 表中选取所有记录：

```sql
SELECT * FROM Websites;
```

### 一些最重要的 SQL 命令

#### SELECT - 从数据库中提取数据

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
ORDER BY column_name [ASC|DESC]
```

* `column_name(s)`: 要查询的列。
* `table_name`: 要查询的表。
* `condition`: 查询条件（可选）。
* `ORDER BY`: 排序方式，`ASC` 表示升序，`DESC` 表示降序（可选）

#### UPDATE - 更新数据库中的数据

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition
```

* `table_name`: 要更新数据的表。
* `column1 = value1, column2 = value2, ...`: 要更新的列及其新值。
* `condition`: 更新条件。

#### DELETE - 从数据库中删除数据

```sql
DELETE FROM table_name
WHERE condition
```

* `table_name`: 要删除数据的表。
* `condition`: 删除条件。

#### INSERT INTO - 向数据库中插入新数据

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
```

* `table_name`: 要插入数据的表。
* `column1, column2, ...`: 要插入数据的列。
* `value1, value2, ...`: 对应列的值。

#### CREATE DATABASE - 创建新数据库

#### ALTER DATABASE - 修改数据库

```sql
ALTER TABLE table_name
ADD column_name data_type
```

* `table_name`: 要修改的表。
* `column_name`: 要添加的列。
* `data_type`: 列的数据类型。

#### CREATE TABLE - 创建新表

```sql
CREATE TABLE table_name (
    column1 data_type constraint,
    column2 data_type constraint,
    ...
)
```

* `table_name`: 要创建的表名。
* `column1, column2, ...`: 表的列。
* `data_type`: 列的数据类型（如 `INT`、`VARCHAR` 等）。
* `constraint`: 列的约束（如 `PRIMARY KEY`、`NOT NULL` 等）。

#### ALTER TABLE - 变更（改变）数据库表

```sql
ALTER TABLE table_name
ADD column_name data_type
```

* `table_name`: 要修改的表。
* `column_name`: 要添加的列。
* `data_type`: 列的数据类型。

#### DROP TABLE - 删除表

```sql
DROP TABLE table_name
```

* `table_name`: 要删除的表。

#### CREATE INDEX - 创建索引（搜索键）用于创建索引，以加快查询速度。

```sql
CREATE INDEX index_name
ON table_name (column_name)
```

* `index_name`: 索引的名称。
* `column_name`: 要索引的列。

#### DROP INDEX - 删除索引

```sql
DROP INDEX index_name
ON table_name
```

* `index_name`: 要删除的索引名称。
* `table_name`: 索引所在的表。

#### WHERE：用于指定筛选条件。

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
```

* `condition`: 筛选条件。

#### ORDER BY：用于对结果集进行排序。

```sql
SELECT column_name(s)
FROM table_name
ORDER BY column_name [ASC|DESC]
```

* `column_name`: 用于排序的列。
* `ASC`: 升序（默认）。
* `DESC`: 降序。

#### GROUP BY：用于将结果集按一列或多列进行分组。

```sql
SELECT column_name(s), aggregate_function(column_name)
FROM table_name
WHERE condition
GROUP BY column_name(s)
```

* `aggregate_function`: 聚合函数（如 COUNT、SUM、AVG 等）。

#### HAVING：用于对分组后的结果集进行筛选。

```sql
SELECT column_name(s), aggregate_function(column_name)
FROM table_name
GROUP BY column_name(s)
HAVING condition
```

* `condition`: 筛选条件。

#### JOIN：用于将两个或多个表的记录结合起来。

```sql
SELECT column_name(s)
FROM table_name1
JOIN table_name2
ON table_name1.column_name = table_name2.column_name
```

* `JOIN`: 可以是 INNER JOIN、LEFT JOIN、RIGHT JOIN 或 FULL JOIN。

#### DISTINCT：用于返回唯一不同的值。

```sql
SELECT DISTINCT column_name(s)
FROM table_name
```

* `column_name(s)`: 要查询的列。

## sqlite3

### 连接数据库

下面的 Python 代码显示了如何连接到一个现有的数据库。如果数据库不存在，那么它就会被创建，最后将返回一个数据库对象。

#### 实例

```python
#!/usr/bin/python
import sqlite3
conn = sqlite3.connect('test.db')
print ("数据库打开成功")
```

在这里，您也可以把数据库名称复制为特定的名称 **:memory:**，这样就会在 RAM 中创建一个数据库。现在，让我们来运行上面的程序，在当前目录中创建我们的数据库 **test.db**。您可以根据需要改变路径。

### 创建表

下面的 Python 代码段将用于在先前创建的数据库中创建一个表：

#### 实例

```python
#!/usr/bin/python
import sqlite3
conn = sqlite3.connect('test.db')
print ("数据库打开成功")
c = conn.cursor()
c.execute('''CREATE TABLE COMPANY
(ID INT PRIMARY KEY     NOT NULL,
NAME           TEXT    NOT NULL,
AGE            INT     NOT NULL,
ADDRESS        CHAR(50),
SALARY         REAL);''')
print ("数据表创建成功")
conn.commit()
conn.close()
```

上述程序执行时，它会在 **test.db** 中创建 COMPANY 表

### INSERT 操作

下面的 Python 程序显示了如何在上面创建的 COMPANY 表中创建记录：

#### 实例

```python
#!/usr/bin/python

import sqlite3

conn = sqlite3.connect('test.db')
c = conn.cursor()
print ("数据库打开成功")

c.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
      VALUES (1, 'Paul', 32, 'California', 20000.00 )")

c.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
      VALUES (2, 'Allen', 25, 'Texas', 15000.00 )")

c.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
      VALUES (3, 'Teddy', 23, 'Norway', 20000.00 )")

c.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
      VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00 )")

conn.commit()
print ("数据插入成功")
conn.close()
```

上述程序执行时，它会在 COMPANY 表中创建给定记录

### SELECT 操作

下面的 Python 程序显示了如何从前面创建的 COMPANY 表中获取并显示记录：

```python
import sqlite3

conn = sqlite3.connect('test.db')
c = conn.cursor()
print ("数据库打开成功")

cursor = c.execute("SELECT id, name, address, salary  from COMPANY")
for row in cursor:
   print "ID = ", row[0]
   print "NAME = ", row[1]
   print "ADDRESS = ", row[2]
   print "SALARY = ", row[3], "\n"

print ("数据操作成功")
conn.close()
```

### UPDATE 操作

下面的 Python 代码显示了如何使用 UPDATE 语句来更新任何记录，然后从 COMPANY 表中获取并显示更新的记录：

```python
#!/usr/bin/python

import sqlite3

conn = sqlite3.connect('test.db')
c = conn.cursor()
print ("数据库打开成功")

c.execute("UPDATE COMPANY set SALARY = 25000.00 where ID=1")
conn.commit()
print "Total number of rows updated :", conn.total_changes

cursor = conn.execute("SELECT id, name, address, salary  from COMPANY")
for row in cursor:
   print "ID = ", row[0]
   print "NAME = ", row[1]
   print "ADDRESS = ", row[2]
   print "SALARY = ", row[3], "\n"

print ("数据操作成功")
conn.close()
```

### DELETE 操作

下面的 Python 代码显示了如何使用 DELETE 语句删除任何记录，然后从 COMPANY 表中获取并显示剩余的记录：

```python
#!/usr/bin/python

import sqlite3

conn = sqlite3.connect('test.db')
c = conn.cursor()
print ("数据库打开成功")

c.execute("DELETE from COMPANY where ID=2;")
conn.commit()
print "Total number of rows deleted :", conn.total_changes

cursor = conn.execute("SELECT id, name, address, salary  from COMPANY")
for row in cursor:
   print "ID = ", row[0]
   print "NAME = ", row[1]
   print "ADDRESS = ", row[2]
   print "SALARY = ", row[3], "\n"

print ("数据操作成功")
conn.close()
```

## ORM

### ORM 基础概念

ORM（Object-Relational Mapping，对象关系映射）是一种编程技术，用于在面向对象编程语言中实现与关系型数据库的交互。ORM 将数据库表映射为编程语言中的类，将表中的行映射为对象实例，将表中的列映射为对象属性。

为什么使用 ORM？
提高开发效率：减少编写重复的 SQL 语句

数据库无关性：可以轻松切换数据库后端

安全性：自动防止 SQL 注入攻击

面向对象：使用熟悉的面向对象范式操作数据库

### 连接数据库

以下是一个连接 SQLite 数据库的示例：

<pre id="__code_1"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_1 > code"></button><code class="language-python">from sqlalchemy import create_engine

# 创建数据库引擎
engine = create_engine('sqlite:///test.db', echo=True)
</code></pre>

### 定义模型

在 SQLAlchemy 中，模型是一个 Python 类，它继承自 `declarative_base` 类。以下是一个定义用户模型的示例：

<pre id="__code_2"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_2 > code"></button><code class="language-python">from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

# 创建基类
Base = declarative_base()

# 定义用户模型
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

    def __repr__(self):
        return f"<User(name='{self.name}', age={self.age})>"
</code></pre>

### 创建表

使用 `Base.metadata.create_all()` 方法可以创建数据库表：

<pre id="__code_3"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_3 > code"></button><code class="language-python"># 创建表
Base.metadata.create_all(engine)
</code></pre>

### 插入数据

以下是一个插入数据的示例：

<pre id="__code_4"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_4 > code"></button><code class="language-python">from sqlalchemy.orm import sessionmaker

# 创建会话工厂
Session = sessionmaker(bind=engine)

# 创建会话
session = Session()

# 创建用户对象
user = User(name='John', age=25)

# 添加用户到会话
session.add(user)

# 提交会话
session.commit()
</code></pre>

### 查询数据

以下是一个查询数据的示例：

<pre id="__code_5"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_5 > code"></button><code class="language-python"># 查询所有用户
users = session.query(User).all()
for user in users:
    print(user)

# 根据条件查询用户
user = session.query(User).filter_by(name='John').first()
print(user)
</code></pre>

### 更新数据

以下是一个更新数据的示例：

<pre id="__code_6"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_6 > code"></button><code class="language-python"># 查询用户
user = session.query(User).filter_by(name='John').first()

# 更新用户信息
user.age = 26

# 提交会话
session.commit()
</code></pre>

### 删除数据

以下是一个删除数据的示例：

<pre id="__code_7"><button class="md-clipboard md-icon" title="复制" data-clipboard-target="#__code_7 > code"></button><code class="language-python"># 查询用户
user = session.query(User).filter_by(name='John').first()

# 删除用户
session.delete(user)

# 提交会话
session.commit()
</code></pre>

## ES

### 测试连接

```python
def test_connection():
    """测试 Elasticsearch 连接和 Ping"""
    print("--- 正在测试 Elasticsearch 连接 ---")
    response = make_request('GET', '')
    if response:
        print("连接成功！")
        print(json.dumps(response, indent=2, ensure_ascii=False))
```

### 分词器

#### 中文分词器

```python
def test_ik_analyzers():
    """测试 IK 分词器"""
    print("\n--- 正在测试 IK 分词器 ---")
    test_text_zh = "我在使用Elasticsearch，这是我的测试。"
    ik_analyzers = ["ik_smart", "ik_max_word"]

    for analyzer in ik_analyzers:
        print(f"\n使用 IK 分词器：{analyzer}")
        data = {
            "analyzer": analyzer,
            "text": test_text_zh
        }
        response = make_request('POST', '_analyze', data=data)
        if response and 'tokens' in response:
            tokens = [token['token'] for token in response['tokens']]
            print(f"原始文本: '{test_text_zh}'")
            print(f"分词结果: {tokens}")
```

#### 英文分词器

```python
def test_common_analyzers():
    """测试常见的 Elasticsearch 内置分词器"""
    print("\n--- 正在测试常见的 Elasticsearch 内置分词器 ---")
    test_text = "Hello, world! This is a test."
    analyzers = ["standard", "simple", "whitespace", "english"]

    for analyzer in analyzers:
        print(f"\n使用分词器：{analyzer}")
        data = {
            "analyzer": analyzer,
            "text": test_text
        }
        response = make_request('POST', '_analyze', data=data)
        if response and 'tokens' in response:
            tokens = [token['token'] for token in response['tokens']]
            print(f"原始文本: '{test_text}'")
            print(f"分词结果: {tokens}")
```

### 创建索引

```python
# 定义索引名称：用于标识 Elasticsearch 中的数据集合（相当于数据库中的“表”）
# 命名规则：小写字母、可含下划线，不能有大写或特殊字符
index_name = "blog_posts_py"

# 定义索引的 settings 和 mappings 配置
# 这个字典将作为创建索引时的 body 参数传入
mapping = {
    # ========================
    # 1. 索引级别设置（Settings）
    # 控制索引的底层存储和性能参数
    # ========================
    "settings": {
        # 主分片数量（primary shards）
        # - 数据会被分割成多个 shard 存储，提升分布式处理能力
        # - 单节点开发环境建议设为 1；生产环境通常设为 3 或以上
        # - ⚠️ 创建后无法修改，必须重建索引才能更改
        "number_of_shards": 1,

        # 副本分片数量（replica shards）
        # - 每个主分片的副本数，用于数据冗余和读请求负载均衡
        # - 设为 0 表示无副本（适用于测试环境）
        # - 生产环境应至少设为 1，防止节点宕机导致数据不可用
        # - ✅ 可在运行时动态调整（无需重建索引）
        "number_of_replicas": 0
    },

    # ========================
    # 2. 字段映射定义（Mappings）
    # 定义文档中各个字段的数据类型及处理方式
    # 相当于数据库的 Schema
    # ========================
    "mappings": {
        "properties": {
            # ------------------------
            # 字段 1: title（文章标题）
            # ------------------------
            "title": {
                "type": "text",                    # 类型为文本，支持全文检索
                "analyzer": "ik_max_word",         # 写入时使用的分词器：细粒度中文分词
                                                  # 将句子尽可能拆分为多个词语，提高召回率
                "search_analyzer": "ik_smart"      # 查询时使用的分词器：智能粗粒度分词
                                                  # 减少碎片词，提升搜索准确性
                                                  # 需提前安装 IK 中文分词插件
            },

            # ------------------------
            # 字段 2: content（文章正文）
            # ------------------------
            "content": {
                "type": "text",                    # 同样是可分词的全文字段
                "analyzer": "ik_max_word",         # 写入时使用最细粒度分词
                "search_analyzer": "ik_smart"      # 查询时使用智能分词，避免过度匹配
            },

            # ------------------------
            # 字段 3: tags（标签）
            # ------------------------
            "tags": {
                "type": "keyword",                 # 关键字类型，不分词，完整匹配
                                                  # 支持数组形式，如 ["Elasticsearch", "教程"]
                                                  # 适用于精确过滤、聚合统计、排序等场景
                                                  # 性能高，常用于 filter 条件
            },

            # ------------------------
            # 字段 4: author（作者）
            # ------------------------
            "author": {
                "type": "keyword",                 # 不分词，用于精确匹配作者名
                                                  # 如 term 查询 {"author": "张三"}
                                                  # 支持按作者进行聚合、筛选和排序
            },

            # ------------------------
            # 字段 5: created_at（发布时间）
            # ------------------------
            "created_at": {
                "type": "date",                    # 日期类型，支持多种格式（ISO8601、时间戳等）
                                                  # 在 Python 中可直接传入 datetime 对象
                                                  # 支持 range 查询、排序、直方图聚合等
                                                  # 示例：查找某时间段内的文章
            }
        }
    }
}

# ========================
# 3. 检查并创建索引逻辑
# ========================

# 先检查指定名称的索引是否已存在
if not es_client.indices.exists(index=index_name):
    # 如果不存在，则根据上面定义的 mapping 创建新索引
    es_client.indices.create(index=index_name, body=mapping)
    print(f"索引 '{index_name}' 创建成功。")
else:
    # 如果已存在，跳过创建，避免重复报错
    print(f"索引 '{index_name}' 已经存在。")
```

### 简单搜索

```python
# 简单搜索
    query = {
        "query": {
            "match": {
                "title": "测试"
            }
        }
    }
    result = es.search(index="my_index", body=query)
```


### 布尔查询

```python
    # 复杂搜索（bool查询）
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": "测试"}},
                    {"range": {"timestamp": {"gte": "2024-01-01"}}}
                ]
            }
        }
    }
    result = es.search(index="my_index", body=query)
```

### 分页查询

```python
  # 分页查询
    query = {
        "query": {"match_all": {}},
        "from": 0,  # 从第几条开始
        "size": 10  # 返回多少条
    }
    result = es.search(index="my_index", body=query)
```

### 聚合查询

```python
# 聚合查询示例  
query = {  
    "aggs": {  
        "popular_titles": {  
            "terms": {  
                "field": "title.keyword",  
                "size": 10  
            }  
        }  
    }  
}  
result = es.search(index="my_index", body=query)
```

### 向量检索

### 混合检索
