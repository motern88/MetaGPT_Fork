"""Hybrid retriever."""

import copy

from llama_index.core.schema import BaseNode, QueryType

from metagpt.rag.retrievers.base import RAGRetriever


class SimpleHybridRetriever(RAGRetriever):
    """一个组合型检索器，聚合来自多个检索器的搜索结果。"""

    def __init__(self, *retrievers):
        """初始化方法，接收多个检索器作为参数，组合成一个复合检索器。

        参数:
            retrievers: 一系列检索器对象，它们将共同参与检索过程。
        """
        self.retrievers: list[RAGRetriever] = retrievers
        super().__init__()

    async def _aretrieve(self, query: QueryType, **kwargs):
        """异步地从所有配置的检索器中检索并聚合搜索结果。

        该方法向 `retrievers` 列表中的每个检索器发送查询，并结合结果，确保每个节点是唯一的（基于节点 ID）。

        参数:
            query: 查询类型，指定要搜索的内容。
            kwargs: 额外的关键字参数，传递给每个检索器。

        返回:
            返回唯一的检索结果，去重后返回。
        """
        all_nodes = []
        for retriever in self.retrievers:
            # 防止检索器修改查询
            query_copy = copy.deepcopy(query)
            nodes = await retriever.aretrieve(query_copy, **kwargs)
            all_nodes.extend(nodes)

        # 合并所有节点，去重
        result = []
        node_ids = set()
        for n in all_nodes:
            if n.node.node_id not in node_ids:
                result.append(n)
                node_ids.add(n.node.node_id)
        return result

    def add_nodes(self, nodes: list[BaseNode]) -> None:
        """支持添加节点。

        将节点添加到每个检索器中。
        """
        for r in self.retrievers:
            r.add_nodes(nodes)

    def persist(self, persist_dir: str, **kwargs) -> None:
        """支持持久化存储。

        将每个检索器的状态持久化到指定目录。
        """
        for r in self.retrievers:
            r.persist(persist_dir, **kwargs)
