"""BM25 retriever."""
from pathlib import Path
from typing import Callable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.retrievers.bm25 import BM25Retriever
from rank_bm25 import BM25Okapi


class DynamicBM25Retriever(BM25Retriever):
    """继承自 BM25 检索器的动态 BM25 检索器，支持动态更新和持久化存储。"""

    def __init__(
        self,
        nodes: list[BaseNode],
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[list[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        index: VectorStoreIndex = None,
    ) -> None:
        """初始化 DynamicBM25Retriever 类。

        参数：
        - nodes: 初始节点列表。
        - tokenizer: 可选的分词器，用于将字符串转换为词项列表。
        - similarity_top_k: 相似度检索时返回的节点数量。
        - callback_manager: 可选的回调管理器。
        - objects: 可选的对象列表。
        - object_map: 可选的对象映射。
        - verbose: 是否启用详细输出。
        - index: 可选的向量存储索引。
        """
        super().__init__(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )
        self._index = index

    def add_nodes(self, nodes: list[BaseNode], **kwargs) -> None:
        """支持添加节点。

        更新节点并重新计算 BM25 索引。
        """
        self._nodes.extend(nodes)  # 添加新节点
        self._corpus = [self._tokenizer(node.get_content()) for node in self._nodes]  # 更新语料库
        self.bm25 = BM25Okapi(self._corpus)  # 更新 BM25 索引

        if self._index:
            self._index.insert_nodes(nodes, **kwargs)  # 如果存在索引，则将节点插入索引

    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化存储。

        将当前的索引数据存储到指定目录。
        """
        if self._index:
            self._index.storage_context.persist(persist_dir)  # 持久化存储到指定目录

    def query_total_count(self) -> int:
        """查询总节点数量。

        返回当前节点的数量。
        """
        return len(self._nodes)

    def clear(self, **kwargs) -> None:
        """支持删除所有节点。

        删除所有节点并清空内存。
        """
        self._delete_json_files(kwargs.get("persist_dir"))  # 删除存储目录中的所有 JSON 文件
        self._nodes = []  # 清空节点列表

    @staticmethod
    def _delete_json_files(directory: str):
        """删除指定目录中的所有 JSON 文件。

        参数：
        - directory: 存储 JSON 文件的目录路径。
        """
        if not directory:
            return

        for file in Path(directory).glob("*.json"):  # 遍历目录中的所有 JSON 文件
            file.unlink()  # 删除文件
