import hashlib
from collections.abc import Iterable, Iterator
from typing import Any


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """按字符切分中文教学文本；生产环境可替换为 token/语义切分。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("要求 chunk_size > 0 且 0 <= overlap < chunk_size")
    clean = " ".join(text.split())
    step = chunk_size - overlap
    return [clean[start : start + chunk_size] for start in range(0, len(clean), step)]


def prepare_chunks(
    documents: Iterable[dict[str, Any]],
    embedder: Any,
    chunk_size: int = 300,
    overlap: int = 50,
) -> Iterator[dict[str, Any]]:
    """练习 3：切分文档、批量向量化，并生成可 bulk 写入的数据。"""
    # 建议步骤：
    # 1. 为每篇文档调用 chunk_text。
    # 2. 将所有 chunk 文本一次或分批传给 embedder.embed。
    # 3. 复制元数据并添加 content/content_vector。
    # 4. 用 doc_id + chunk 序号生成稳定 chunk_id。
    # 可使用下方 stable_chunk_id。
    # TODO(student): 按 README 第 5 章补全。
    # raise NotImplementedError("请实现 prepare_chunks")
    for document in documents:
        content = document["content"]
        chunks = chunk_text(content, chunk_size, overlap)
        vectors = embedder.embed(chunks)
        for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
            new_chunk = document.copy()
            new_chunk["chunk_id"] = stable_chunk_id(document["doc_id"], position)
            new_chunk["content"] = chunk
            new_chunk["content_vector"] = vector

            yield new_chunk


def stable_chunk_id(doc_id: str, position: int) -> str:
    raw = f"{doc_id}:{position}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]
