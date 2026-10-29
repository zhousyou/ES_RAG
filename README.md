# 本地 Elasticsearch RAG 教学 Demo

这是一个“边学边实现”的项目：部署、数据和命令行骨架已经准备好，7 个核心练习由你补全。最终系统支持本地 Embedding、BM25、元数据过滤和混合检索。

## 0. 学习路线与目录

```text
docker-compose.yml       # 本地 Elasticsearch
data/documents.jsonl     # 教学数据
rag_demo/
  schema.py              # TODO 1：索引 mapping
  embedding.py           # TODO 2：本地向量化
  ingest.py              # TODO 3：切分与入库数据准备
  search.py              # TODO 4~7：过滤、BM25、kNN、RRF
  cli.py                 # 已完成：串联各模块
tests/                   # 每个 TODO 的验收测试
```

建议按 TODO 编号依次完成。搜索 TODO：

```bash
grep -R "TODO(student)" rag_demo
```

## 1. 本地部署 Elasticsearch

前置条件：Docker Desktop（建议给 Docker 至少 3 GB 内存）。

```bash
docker compose up -d
docker compose ps
curl http://localhost:9200
```

本项目固定使用 Elasticsearch `8.17.4`，关闭了认证，只监听宿主机回环地址 `127.0.0.1`。该配置适合本机教学，不能直接用于公网或生产环境。

Apple Silicon 上的 Docker 会声明支持 SVE2，但实际执行相关指令会触发 `SIGILL`。因此 Compose 同时设置了 `CLI_JAVA_OPTS` 和 `ES_JAVA_OPTS` 中的 `-XX:UseSVE=0`：前者作用于启动前的 CLI 进程，后者作用于 Elasticsearch 主进程。

数据保存在 Docker volume `es_data`。停止服务不会删除数据：

```bash
docker compose down
```

只有需要清空全部 ES 数据时才执行：

```bash
docker compose down -v
```

### 真正离线的部署方式

“本地运行”不等于“安装时不联网”。首次 `docker compose up` 会下载镜像。完全隔离网络时，在联网机器执行：

```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.17.4
docker save docker.elastic.co/elasticsearch/elasticsearch:8.17.4 \
  -o elasticsearch-8.17.4.tar
```

复制 tar 文件到离线机器后：

```bash
docker load -i elasticsearch-8.17.4.tar
docker compose up -d
```

## 2. Python 环境与本地 Embedding 模型

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在可联网机器从 ModelScope 下载 `BAAI/bge-small-zh-v1.5`：

```bash
python scripts/download_model.py
```

模型会保存到 `models/bge-small-zh-v1.5`。推理阶段只读取这个本地目录：`.env` 中的 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和代码里的 `local_files_only=True` 会阻止再次访问模型仓库。离线机器可以直接复制整个 `models/` 目录和预先下载的 Python wheel 包。

验证 ES：

```bash
python -m rag_demo.cli health
```

## 3. TODO 1：设计索引 mapping

打开 `rag_demo/schema.py`。返回值应具有以下轮廓：

```python
{
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            # 元数据字段……
            "content_vector": {
                "type": "dense_vector",
                "dims": embedding_dim,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}
```

为什么这样设计：

- `text` 会分词并参与 BM25 评分；
- `keyword` 不分词，适合 category、tags、source 等精确过滤；
- `date` 支持时间范围过滤；
- `dense_vector` 建立 HNSW 近似最近邻索引；
- mapping 中维度必须与模型输出维度完全一致。

完成后运行：

```bash
pytest tests/test_schema.py -q
python -m rag_demo.cli init-index
curl http://localhost:9200/rag_chunks/_mapping?pretty
```

教学版直接使用 ES 内置 analyzer。它能运行中文 BM25，但中文分词质量有限；生产项目可评估与 ES 版本严格匹配的 IK/SmartCN 插件。插件及词典也必须提前打包，才能保持完全离线。

## 4. TODO 2：实现本地 Embedding

打开 `rag_demo/embedding.py`，实现 `embed`：

```text
输入 texts
  → model.encode(normalize_embeddings=True, convert_to_numpy=True)
  → 检查二维结果中每个向量的维度
  → 转为 Python list 后返回
```

归一化可让 cosine 检索更稳定。向量必须转换为普通 list，否则 JSON 序列化可能无法处理 NumPy 类型。

可临时验证：

```bash
python - <<'PY'
from rag_demo.config import load_settings
from rag_demo.embedding import LocalEmbedder
s = load_settings()
e = LocalEmbedder(s.embedding_model_path, s.embedding_dim)
vectors = e.embed(["什么是混合检索？"])
print(len(vectors), len(vectors[0]), vectors[0][:5])
PY
```

## 5. TODO 3：文本切分与入库

打开 `rag_demo/ingest.py`。`prepare_chunks` 的推荐流程：

```text
for 每篇文档:
    chunks = chunk_text(content)
    vectors = embed(chunks)
    for position, (chunk, vector):
        复制文档元数据
        添加稳定 chunk_id、content、content_vector
        yield 结果
```

不要把原始文档的整段正文保留在每个 chunk 中。稳定 ID 可使重复入库覆盖旧块，而不是制造副本。

```bash
pytest tests/test_ingest.py -q
python -m rag_demo.cli ingest --file data/documents.jsonl
curl "http://localhost:9200/rag_chunks/_count?pretty"
```

## 6. TODO 4：元数据过滤

在 `rag_demo/search.py` 中实现 `build_filters`：

```text
category      → term
tags          → terms
created_after → range(gte)
```

过滤条件应放在 `bool.filter` 或 kNN 的 `filter` 中。它们只决定文档是否入选，不参与 `_score`，并且更容易被 ES 缓存。应拒绝未知字段，避免把任意用户输入直接拼入查询 DSL。

## 7. TODO 5：BM25 查询

目标 DSL：

```json
{
  "size": 5,
  "query": {
    "bool": {
      "must": [{
        "multi_match": {
          "query": "混合检索",
          "fields": ["title^2", "content"]
        }
      }],
      "filter": []
    }
  }
}
```

`title^2` 表示标题匹配贡献是普通字段的两倍。完成后：

```bash
pytest tests/test_search.py -q
python -m rag_demo.cli search "关键词相关性算法" --mode bm25
python -m rag_demo.cli search "搜索" --mode bm25 --category search
```

## 8. TODO 6：向量 kNN 查询

Elasticsearch 8.x 顶层 kNN 请求轮廓：

```python
{
    "size": size,
    "knn": {
        "field": "content_vector",
        "query_vector": list(query_vector),
        "k": size,
        "num_candidates": num_candidates,
        "filter": list(filters),
    },
}
```

`num_candidates` 越大，召回率通常越好，但延迟和计算量也越高，并且必须不小于 `k`。

```bash
python -m rag_demo.cli search "如何按语义找到相似内容" --mode vector
python -m rag_demo.cli search "相关性" --mode vector --tags elasticsearch
```

## 9. TODO 7：RRF 混合检索

BM25 与 cosine 的原始分数不在同一尺度，不应直接相加。RRF 只使用排名：

```text
score(document) = Σ 1 / (rank_constant + rank)
```

对每一路结果从 rank=1 开始遍历，以 `_id` 聚合分数；同一文档出现多次时累加；最后降序排列并截取 `limit`。把融合得分保存为 `_rrf_score`。

```bash
pytest tests/test_search.py -q
python -m rag_demo.cli search "关键词和语义一起检索" --mode hybrid
```

## 10. 完整验收与实验

```bash
pytest -q
python -m rag_demo.cli init-index
python -m rag_demo.cli ingest
python -m rag_demo.cli search "如何组合语义与关键词召回" --mode hybrid
```

建议对同一个问题分别使用 `bm25`、`vector`、`hybrid`，观察排名变化；再增加 `--category search`、`--tags elasticsearch` 或 `--created-after 2026-02-01` 观察过滤效果。

## 11. 从 Demo 演进到完整 RAG

本教程完成后，下一阶段可以按以下顺序扩展：

1. 增加 PDF/Markdown 加载器与 token 级切分；
2. 增加离线 reranker；
3. 把前 N 个 chunk 组装成提示词上下文；
4. 接入 Ollama、llama.cpp 等本地 LLM；
5. 增加引用、召回评估集和 Recall@K/NDCG 指标；
6. 生产化时恢复 ES TLS/认证、容量规划、备份与监控。
