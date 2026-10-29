from pathlib import Path
from typing import Protocol, Sequence
# from config import load_settings


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


class LocalEmbedder:
    """仅从本地目录加载 SentenceTransformer 模型。"""

    def __init__(self, model_path: str, expected_dim: int) -> None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"本地模型不存在：{path}。请先执行 README 第 2 章的下载步骤。"
            )

        # 延迟导入，单元测试无需加载 PyTorch。
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import normalize_embeddings

        self._model: Encoder = SentenceTransformer(str(path), local_files_only=True)
        self._expected_dim = expected_dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """练习 2：批量生成归一化向量，并校验向量维度。"""
        # 伪代码：
        # vectors = self._model.encode(
        #     texts, normalize_embeddings=True, convert_to_numpy=True
        # )
        # 检查每个向量长度是否等于 self._expected_dim
        # 将 numpy 数组转换成 Elasticsearch 可序列化的 list[list[float]]
        # TODO(student): 按 README 第 4 章补全。
        # raise NotImplementedError("请实现 LocalEmbedder.embed")
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy = True)
        # print(f"vectors的维度是：{len(vectors)}")
        for vector in vectors:
            if self._expected_dim != len(vector):
                raise ValueError("向量长度不匹配")
        return vectors.tolist()
    
# if __name__ == "__main__":
#     # from rag_demo.config import load_settings
    
#     s = load_settings()
#     e = LocalEmbedder(s.embedding_model_path, s.embedding_dim)
#     vectors = e.embed(["什么是混合检索"])
#     print(len(vectors))
#     print(type(vectors))
#     # print(vectors[0])
        
