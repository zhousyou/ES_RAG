# 教学 Demo 交付报告

已交付：

- Elasticsearch 8.17.4 本地 Docker Compose 部署；
- 完全离线运行说明、镜像迁移与模型下载脚本；
- 本地中文 Embedding 骨架；
- 文本切分、批量入库与稳定 chunk ID 骨架；
- BM25、向量检索、元数据过滤和应用层 RRF 骨架；
- 示例数据、统一 CLI、7 个核心练习和验收测试；
- 分章节教程、设计文档、实施计划与代码审查记录。

校验：

- `docker compose config --quiet`：通过；
- Python 语法检查：通过；
- IDE lint：无错误；
- 教学测试：1 个基础测试通过，7 个 TODO 测试按设计失败。
