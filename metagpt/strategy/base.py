# -*- coding: utf-8 -*-
# @Date    : 12/25/2023 9:16 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from abc import ABC
from typing import List

from anytree import Node, RenderTree
from pydantic import BaseModel


class BaseParser(BaseModel, ABC):
    """解析器基类，提供解析、提议、采样和值计算的接口。"""

    def __call__(self, *args, **kwargs):
        """该方法需要在子类中实现，抛出未实现错误。"""
        raise NotImplementedError

    def propose(self, current_state: str, **kwargs) -> str:
        """该方法需要在子类中实现，用于提出一个新的提案。"""
        raise NotImplementedError

    def sample(self, current_state: str, **kwargs) -> str:
        """该方法需要在子类中实现，用于根据当前状态生成一个样本。"""
        raise NotImplementedError

    def value(self, input: str, **kwargs) -> str:
        """该方法需要在子类中实现，用于计算输入的值。"""
        raise NotImplementedError


class BaseEvaluator(BaseModel, ABC):
    """评估器基类，提供评估和状态验证的接口。"""

    def __call__(self, *args, **kwargs):
        """该方法需要在子类中实现，抛出未实现错误。"""
        raise NotImplementedError

    def status_verify(self, *args, **kwargs):
        """该方法需要在子类中实现，用于验证状态。"""
        raise NotImplementedError


class ThoughtNode(Node):
    """表示思维树中的一个节点。"""

    name: str = ""  # 节点名称
    value: int = 0  # 节点值
    id: int = 0  # 节点ID
    valid_status: bool = True  # 节点有效状态

    def update_value(self, value) -> None:
        """更新思维节点的值。"""
        self.value = value

    def update_valid_status(self, status) -> None:
        """更新思维节点的有效状态。"""
        self.valid_status = status


class ThoughtTree(RenderTree):
    """表示思维树的数据结构。"""

    @property
    def all_nodes(self) -> List[ThoughtNode]:
        """
        获取思维树中所有节点的列表。

        返回:
            List[ThoughtNode]: 包含思维树中所有节点的列表。
        """
        all_nodes = [node for _, _, node in self]
        return all_nodes

    def update_node(self, thought: List[dict] = [], current_node: ThoughtNode = None) -> List[ThoughtNode]:
        """
        更新思维树，添加新的思维节点。

        参数:
            thought (List[dict]): 包含思维节点信息的字典列表。
            current_node (ThoughtNode): 当前节点，新的思维节点将被添加到该节点下。

        返回:
            List[ThoughtNode]: 代表更新后的树节点的 ThoughtNode 实例列表。
        """
        nodes = []
        for node_info in thought:
            node = ThoughtNode(
                name=node_info["node_state_instruction"], parent=current_node, id=int(node_info["node_id"])
            )
            nodes.append(node)
        return nodes

    def parse_node_path(self, node) -> List[str]:
        """
        解析并获取给定思维节点的层级路径。

        该方法遍历所提供节点的父节点，并构建从根节点到指定节点的完整路径。

        参数:
            node: 需要解析的思维节点。

        返回:
            List[str]: 表示给定思维节点的完整层级路径的列表，
                       列表按从根节点到给定节点的顺序排列。
        """
        full_node_path = []
        while node is not None:
            full_node_path.append(node.name)
            node = node.parent
        full_node_path.reverse()
        return full_node_path

    def show(self) -> None:
        """打印更新后的思维树。"""
        print("\nUpdated Tree:")
        for pre, _, node in self:
            print(f"{pre}{node.name}, value: {node.value}, valid_status: {node.valid_status}")
