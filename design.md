# Elasticsearch RAG 教学 Demo：设计说明

## 目标

创建一个完全本地化的 RAG 检索教学项目。学习者将亲自完成：

1. 文本切分与本地 Embedding；
2. Elasticsearch `dense_vector` 向量检索；
3. Elasticsearch BM25 关键词检索；
4. 基于结构化字段的元数据过滤；
5. 在应用层用 RRF 融合向量与 BM25 结果。

本阶段只覆盖“检索增强”中的索引与召回，不接入大语言模型生成答案，避免把注意力从 Elasticsearch 核心概念上移开。

## 技术选择

- Python 3.11+
- Elasticsearch 8.17.4，Docker Compose 单节点部署
- `sentence-transformers` 加载本地模型目录
- 推荐模型：`BAAI/bge-small-zh-v1.5`（512 维，适合中文教学与本地 CPU）
- Elasticsearch Python Client 8.x
- 应用层 RRF，不依赖 Elasticsearch 商业许可中的高级混合检索能力

## 数据模型

每个文档块保存以下字段：

- `chunk_id`：块唯一标识
- `doc_id`：原文档标识
- `title`：标题，参与 BM25
- `content`：正文，参与 BM25 和向量化
- `content_vector`：`dense_vector`，512 维、cosine 相似度
- `source`：来源
- `category`：分类，可精确过滤
- `tags`：标签，可精确过滤
- `created_at`：时间，可范围过滤

## 查询路径

- 向量查询：问题 → 本地 Embedding → Elasticsearch kNN
- BM25 查询：问题 → `multi_match(title^2, content)` 
- 元数据查询：`bool.filter` 组合 `term`、`terms`、`range`
- 混合查询：分别取得向量与 BM25 排名 → 应用层 RRF → 返回统一结果

## 本地化边界

Elasticsearch 和应用均在本机运行。模型通过 ModelScope 下载一次并保存到 `models/`，运行阶段只读取本地目录。若部署环境从不联网，可在另一台机器下载后复制模型目录。

## 教学策略

仓库提供完整部署、数据、接口形状和测试，核心函数以 `TODO` 留给学习者实现。每章先阅读教程，再实现一个 TODO，最后运行对应测试获得反馈。
