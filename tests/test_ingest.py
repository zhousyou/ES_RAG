from rag_demo.ingest import chunk_text, prepare_chunks, stable_chunk_id


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_chunk_text_uses_overlap() -> None:
    assert chunk_text("abcdefghij", chunk_size=6, overlap=2) == [
        "abcdef",
        "efghij",
        "ij",
    ]


def test_prepare_chunks_preserves_metadata() -> None:
    documents = [
        {
            "doc_id": "d1",
            "title": "标题",
            "content": "abcdefgh",
            "source": "test",
            "category": "rag",
            "tags": ["demo"],
            "created_at": "2026-01-01",
        }
    ]

    chunks = list(prepare_chunks(documents, FakeEmbedder(), chunk_size=5, overlap=1))

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == stable_chunk_id("d1", 0)
    assert chunks[0]["content"] == "abcde"
    assert chunks[0]["content_vector"] == [5.0, 1.0]
    assert chunks[0]["category"] == "rag"
