from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_filters(metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """练习 4：把 category/tags/created_after 转成 bool.filter 子句。"""
    # category -> {"term": {"category": value}}
    # tags -> {"terms": {"tags": values}}
    # created_after -> {"range": {"created_at": {"gte": value}}}
    # 忽略值为 None 或空列表的字段；未知字段应抛出 ValueError。
    # TODO(student): 按 README 第 6 章补全。
    # raise NotImplementedError("请实现 build_filters")
    filters = []
    for k, v in metadata.items():
        if v is None or v == []:
            continue
        elif k == "category":
            filters.append({"term": {"category": v}})
        elif k == "tags":
            filters.append({"terms": {"tags": v}})
        elif k == "created_after":
            filters.append({"range": {"created_at": {"gte": v}}})
        else:
            raise ValueError(f"未知字段{k}")
    return filters


def build_bm25_query(
    query: str, filters: Sequence[dict[str, Any]], size: int = 10
) -> dict[str, Any]:
    """练习 5：构造带元数据过滤的 BM25 multi_match 请求体。"""
    # 将 multi_match 放到 bool.must，将 filters 放到 bool.filter。
    # title 权重应高于 content，例如字段写成 "title^2"。
    # TODO(student): 按 README 第 7 章补全。
    # raise NotImplementedError("请实现 build_bm25_query")
    dsl = {
        "size": size,
        "query":{
            "bool":{
                "must":[{
                    "multi_match":{
                        "query": query,
                        "fields": ["title^2", "content"]
                    }
                }],
                "filter":filters
            }
        }
    }
    
    return dsl


def build_knn_query(
    query_vector: Sequence[float],
    filters: Sequence[dict[str, Any]],
    size: int = 10,
    num_candidates: int = 100,
) -> dict[str, Any]:
    """练习 6：构造 Elasticsearch 顶层 knn 请求体。"""
    # 关键字段：field、query_vector、k、num_candidates、filter。
    # k 通常等于 size；num_candidates 应不小于 k。
    # TODO(student): 按 README 第 8 章补全。
    # raise NotImplementedError("请实现 build_knn_query")
    return {
        "size": size,
        "knn": {
            "field": "content_vector",
            "query_vector": list(query_vector),
            "k": size,
            "num_candidates": num_candidates,
            "filter": list(filters)
        }
    }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Mapping[str, Any]]],
    rank_constant: int = 60,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """练习 7：按 _id 合并多组排名结果。

    每个文档在某一路排名 rank（从 1 开始）的得分贡献为：
    1 / (rank_constant + rank)
    """
    # TODO(student): 按 README 第 9 章补全。
    raise NotImplementedError("请实现 reciprocal_rank_fusion")
