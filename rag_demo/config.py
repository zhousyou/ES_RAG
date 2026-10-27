from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    es_url: str
    es_index: str
    embedding_model_path: str
    embedding_dim: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        es_url=os.getenv("ES_URL", "http://localhost:9200"),
        es_index=os.getenv("ES_INDEX", "rag_chunks"),
        embedding_model_path=os.getenv(
            "EMBEDDING_MODEL_PATH", "./models/bge-small-zh-v1.5"
        ),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "512")),
    )
