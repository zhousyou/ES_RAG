from rag_demo.schema import build_index_mapping


def test_mapping_supports_text_vector_and_metadata() -> None:
    mapping = build_index_mapping(512)
    properties = mapping["mappings"]["properties"]

    assert properties["content"]["type"] == "text"
    assert properties["category"]["type"] == "keyword"
    assert properties["created_at"]["type"] == "date"
    assert properties["content_vector"] == {
        "type": "dense_vector",
        "dims": 512,
        "index": True,
        "similarity": "cosine",
    }
