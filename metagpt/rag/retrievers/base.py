"""Base retriever."""

from abc import abstractmethod

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryType

from metagpt.utils.reflection import check_methods


class RAGRetriever(BaseRetriever):
    """继承自 llama_index 的 RAG 检索器类，用于检索节点。"""

    @abstractmethod
    async def _aretrieve(self, query: QueryType) -> list[NodeWithScore]:
        """异步检索节点"""

    def _retrieve(self, query: QueryType) -> list[NodeWithScore]:
        """同步检索节点"""


class ModifiableRAGRetriever(RAGRetriever):
    """支持修改的 RAG 检索器类，允许添加节点。"""

    @classmethod
    def __subclasshook__(cls, C):
        if cls is ModifiableRAGRetriever:
            return check_methods(C, "add_nodes")
        return NotImplemented

    @abstractmethod
    def add_nodes(self, nodes: list[BaseNode], **kwargs) -> None:
        """支持添加文档，必须实现该函数"""


class PersistableRAGRetriever(RAGRetriever):
    """支持持久化的 RAG 检索器类，可以将数据保存到磁盘。"""

    @classmethod
    def __subclasshook__(cls, C):
        if cls is PersistableRAGRetriever:
            return check_methods(C, "persist")
        return NotImplemented

    @abstractmethod
    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化操作，必须实现该函数"""


class QueryableRAGRetriever(RAGRetriever):
    """支持查询总节点数量的 RAG 检索器类。"""

    @classmethod
    def __subclasshook__(cls, C):
        if cls is QueryableRAGRetriever:
            return check_methods(C, "query_total_count")
        return NotImplemented

    @abstractmethod
    def query_total_count(self) -> int:
        """支持查询总节点数，必须实现该函数"""


class DeletableRAGRetriever(RAGRetriever):
    """支持删除所有节点的 RAG 检索器类。"""

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DeletableRAGRetriever:
            return check_methods(C, "clear")
        return NotImplemented

    @abstractmethod
    def clear(self, **kwargs) -> int:
        """支持删除所有节点，必须实现该函数"""
