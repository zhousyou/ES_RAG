from typing import Any


def build_index_mapping(embedding_dim: int) -> dict[str, Any]:
    """练习 1：构造 RAG 文档块的 Elasticsearch mapping。

    提示：
    - title/content 使用 text；
    - doc_id/source/category/tags 使用 keyword；
    - created_at 使用 date；
    - content_vector 使用 dense_vector，并配置 index、dims 和 cosine。
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
    """
    # TODO(student): 按 README 第 3 章补全。
    return {
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
    # raise NotImplementedError("请实现 build_index_mapping")

