import argparse
import json
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch, helpers

from rag_demo.config import load_settings
from rag_demo.embedding import LocalEmbedder
from rag_demo.ingest import prepare_chunks
from rag_demo.schema import build_index_mapping
from rag_demo.search import (
    build_bm25_query,
    build_filters,
    build_knn_query,
    reciprocal_rank_fusion,
)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def print_hits(hits: list[dict[str, Any]]) -> None:
    for rank, hit in enumerate(hits, 1):
        source = hit["_source"]
        score = hit.get("_rrf_score", hit.get("_score", 0.0))
        print(f"{rank:>2}. score={score:.6f} [{source['category']}] {source['title']}")
        print(f"    {source['content'][:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Elasticsearch RAG 教学 Demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health")
    subparsers.add_parser("init-index")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--file", default="data/documents.jsonl")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    search_parser.add_argument("--category")
    search_parser.add_argument("--tags", nargs="*")
    search_parser.add_argument("--created-after")
    search_parser.add_argument("--size", type=int, default=5)

    args = parser.parse_args()
    settings = load_settings()
    client = Elasticsearch(settings.es_url)

    if args.command == "health":
        print(json.dumps(dict(client.cluster.health()), ensure_ascii=False, indent=2))
        return

    if args.command == "init-index":
        if client.indices.exists(index=settings.es_index):
            client.indices.delete(index=settings.es_index)
        client.indices.create(
            index=settings.es_index,
            **build_index_mapping(settings.embedding_dim),
        )
        print(f"已创建索引：{settings.es_index}")
        return

    embedder = LocalEmbedder(settings.embedding_model_path, settings.embedding_dim)

    if args.command == "ingest":
        chunks = prepare_chunks(read_jsonl(args.file), embedder)
        actions = (
            {
                "_index": settings.es_index,
                "_id": chunk["chunk_id"],
                "_source": chunk,
            }
            for chunk in chunks
        )
        success, errors = helpers.bulk(client, actions, raise_on_error=False)
        client.indices.refresh(index=settings.es_index)
        print(f"写入成功：{success}；失败：{len(errors)}")
        return

    metadata = {
        "category": args.category,
        "tags": args.tags,
        "created_after": args.created_after,
    }
    filters = build_filters(metadata)
    source_fields = [
        "chunk_id",
        "doc_id",
        "title",
        "content",
        "source",
        "category",
        "tags",
        "created_at",
    ]

    bm25_hits: list[dict[str, Any]] = []
    vector_hits: list[dict[str, Any]] = []
    if args.mode in {"bm25", "hybrid"}:
        body = build_bm25_query(args.query, filters, args.size)
        response = client.search(index=settings.es_index, source=source_fields, **body)
        bm25_hits = list(response["hits"]["hits"])

    if args.mode in {"vector", "hybrid"}:
        query_vector = embedder.embed([args.query])[0]
        body = build_knn_query(query_vector, filters, args.size)
        response = client.search(index=settings.es_index, source=source_fields, **body)
        vector_hits = list(response["hits"]["hits"])

    if args.mode == "bm25":
        print_hits(bm25_hits)
    elif args.mode == "vector":
        print_hits(vector_hits)
    else:
        print_hits(reciprocal_rank_fusion([bm25_hits, vector_hits], limit=args.size))


if __name__ == "__main__":
    main()
