"""Elasticsearch retriever."""

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode


class ElasticsearchRetriever(VectorIndexRetriever):
    """Elasticsearch 检索器，继承自 VectorIndexRetriever，支持基于 Elasticsearch 的检索。"""

    def add_nodes(self, nodes: list[BaseNode], **kwargs) -> None:
        """支持添加节点。

        将节点插入到 Elasticsearch 索引中。
        """
        self._index.insert_nodes(nodes, **kwargs)

    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化存储。

        Elasticsearch 会自动保存，因此无需手动实现持久化操作。
        """
        pass
