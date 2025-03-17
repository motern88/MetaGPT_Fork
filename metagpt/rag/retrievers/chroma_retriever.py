"""Chroma retriever."""

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore


class ChromaRetriever(VectorIndexRetriever):
    """Chroma 检索器，继承自 VectorIndexRetriever，支持基于 Chroma 向量存储进行检索。"""

    @property
    def vector_store(self) -> ChromaVectorStore:
        """返回 Chroma 向量存储实例。"""
        return self._vector_store

    def add_nodes(self, nodes: list[BaseNode], **kwargs) -> None:
        """支持添加节点。

        将节点插入到索引中。
        """
        self._index.insert_nodes(nodes, **kwargs)

    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化存储。

        Chromadb 自动保存，因此无需实现持久化操作。
        """
        pass

    def query_total_count(self) -> int:
        """支持查询总节点数量。

        返回 Chroma 向量存储中集合的节点数量。
        """
        return self.vector_store._collection.count()

    def clear(self, **kwargs) -> None:
        """支持删除所有节点。

        获取所有节点的 ID 并删除它们。
        """
        ids = self.vector_store._collection.get()["ids"]
        if ids:
            self.vector_store._collection.delete(ids=ids)  # 删除所有节点
