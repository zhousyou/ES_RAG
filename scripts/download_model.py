"""从 ModelScope 下载一次；运行 RAG 时只读取本地目录。"""

from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


MODEL_ID = "BAAI/bge-small-zh-v1.5"
TARGET = Path("models/bge-small-zh-v1.5")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=str(TARGET))
    print(f"模型已保存到：{TARGET.resolve()}")


if __name__ == "__main__":
    main()
