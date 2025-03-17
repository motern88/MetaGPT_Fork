"""Milvus retriever."""

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode


class MilvusRetriever(VectorIndexRetriever):
    """Milvus 检索器。"""

    def add_nodes(self, nodes: list[BaseNode], **kwargs) -> None:
        """支持添加节点。

        将节点添加到 Milvus 索引中。
        """
        self._index.insert_nodes(nodes, **kwargs)

    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化存储。

        Milvus 会自动保存，因此不需要额外实现持久化功能。
        """
        # Milvus 会自动保存，所以无需实现具体的持久化逻辑
