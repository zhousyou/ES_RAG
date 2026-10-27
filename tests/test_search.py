import pytest

from rag_demo.search import (
    build_bm25_query,
    build_filters,
    build_knn_query,
    reciprocal_rank_fusion,
)


def test_build_filters() -> None:
    assert build_filters(
        {
            "category": "search",
            "tags": ["elasticsearch", "bm25"],
            "created_after": "2026-01-01",
        }
    ) == [
        {"term": {"category": "search"}},
        {"terms": {"tags": ["elasticsearch", "bm25"]}},
        {"range": {"created_at": {"gte": "2026-01-01"}}},
    ]


def test_build_filters_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="未知"):
        build_filters({"tenant": "a"})


def test_bm25_query() -> None:
    filters = [{"term": {"category": "rag"}}]
    body = build_bm25_query("混合检索", filters, size=5)

    assert body["size"] == 5
    bool_query = body["query"]["bool"]
    assert bool_query["filter"] == filters
    assert bool_query["must"][0]["multi_match"]["fields"] == ["title^2", "content"]


def test_knn_query() -> None:
    filters = [{"term": {"category": "vector"}}]
    body = build_knn_query([0.1, 0.2], filters, size=3, num_candidates=20)

    assert body["size"] == 3
    assert body["knn"] == {
        "field": "content_vector",
        "query_vector": [0.1, 0.2],
        "k": 3,
        "num_candidates": 20,
        "filter": filters,
    }


def test_rrf_rewards_documents_found_by_both_rankings() -> None:
    bm25 = [
        {"_id": "a", "_source": {"title": "A"}},
        {"_id": "b", "_source": {"title": "B"}},
    ]
    vector = [
        {"_id": "b", "_source": {"title": "B"}},
        {"_id": "c", "_source": {"title": "C"}},
    ]

    results = reciprocal_rank_fusion([bm25, vector], rank_constant=60, limit=3)

    assert [hit["_id"] for hit in results] == ["b", "a", "c"]
    assert results[0]["_rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
