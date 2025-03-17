"""FAISS retriever."""

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode


class FAISSRetriever(VectorIndexRetriever):
    """FAISS 检索器，继承自 VectorIndexRetriever，支持基于 FAISS 的向量检索。"""

    def add_nodes(self, nodes: list[BaseNode], **kwargs) -> None:
        """支持添加节点。

        将节点插入到 FAISS 索引中。
        """
        self._index.insert_nodes(nodes, **kwargs)

    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化存储。

        将 FAISS 索引的存储上下文持久化到指定目录。
        """
        self._index.storage_context.persist(persist_dir)
