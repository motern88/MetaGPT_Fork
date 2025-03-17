"""RAG Interfaces."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class RAGObject(Protocol):
    """支持 RAG 添加对象。"""

    def rag_key(self) -> str:
        """用于 RAG 搜索的唯一标识符。"""

    def model_dump_json(self) -> str:
        """用于 RAG 持久化存储的函数。

        Pydantic 模型无需实现此方法，因为 Pydantic 已经内置了一个名为 model_dump_json 的函数。
        """


@runtime_checkable
class NoEmbedding(Protocol):
    """某些检索器不需要嵌入，例如 BM25。"""

    _no_embedding: bool