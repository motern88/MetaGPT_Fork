"""Object ranker."""

import heapq
import json
from typing import Literal, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field

from metagpt.rag.schema import ObjectNode


class ObjectSortPostprocessor(BaseNodePostprocessor):
    """根据对象的字段进行排序，可以选择升序或降序。

    假设传入的节点列表是带有分数的 ObjectNode 列表。
    """

    field_name: str = Field(..., description="对象字段的名称，字段的值必须是可以比较的。")
    order: Literal["desc", "asc"] = Field(default="desc", description="排序方向。")
    top_n: int = 5  # 排序后保留的前 N 个节点

    @classmethod
    def class_name(cls) -> str:
        return "ObjectSortPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """后处理节点。

        Args:
            nodes: 带有分数的节点列表。
            query_bundle: 可选，包含查询信息的封装对象。

        Returns:
            返回经过后处理排序的节点列表。
        """
        if query_bundle is None:
            raise ValueError("缺少查询封装信息。")

        if not nodes:
            return []

        self._check_metadata(nodes[0].node)  # 检查元数据

        # 获取排序键：根据字段名称提取对象的值
        sort_key = lambda node: json.loads(node.node.metadata["obj_json"])[self.field_name]
        return self._get_sort_func()(self.top_n, nodes, key=sort_key)

    def _check_metadata(self, node: ObjectNode):
        """检查节点的元数据，确保对象 JSON 格式正确，并且包含所需的字段。"""
        try:
            obj_dict = json.loads(node.metadata.get("obj_json"))
        except Exception as e:
            raise ValueError(f"元数据中的对象 JSON 无效: {node.metadata}, 错误: {e}")

        if self.field_name not in obj_dict:
            raise ValueError(f"在对象中找不到字段 '{self.field_name}': {obj_dict}")

    def _get_sort_func(self):
        """根据排序方向返回对应的排序函数"""
        return heapq.nlargest if self.order == "desc" else heapq.nsmallest
