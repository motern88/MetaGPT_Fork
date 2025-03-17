"""Base Ranker."""

from abc import abstractmethod
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class RAGRanker(BaseNodePostprocessor):
    """继承自 llama_index 类，用于对节点进行排序和后处理"""

    @abstractmethod
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """后处理节点

        Args:
            nodes: 一个包含节点及其分数的列表。
            query_bundle: 可选，包含查询信息的封装对象。

        Returns:
            返回经过后处理的节点列表。
        """
