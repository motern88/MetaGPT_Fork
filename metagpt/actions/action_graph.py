#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/30 13:52
@Author  : alexanderwu
@File    : action_graph.py
"""
from __future__ import annotations

# from metagpt.actions.action_node import ActionNode


class ActionGraph:
    """ActionGraph: 表示操作之间依赖关系的有向图"""

    def __init__(self):
        """初始化操作图"""
        self.nodes = {}  # 存储节点的字典，键是节点的唯一标识符
        self.edges = {}  # 存储边的字典，键是起始节点的标识符，值是目标节点的标识符列表
        self.execution_order = []  # 执行顺序，用于拓扑排序后的节点顺序

    def add_node(self, node):
        """向图中添加一个节点"""
        self.nodes[node.key] = node  # 使用节点的唯一标识符（key）作为字典的键

    def add_edge(self, from_node: "ActionNode", to_node: "ActionNode"):
        """向图中添加一条边，表示从 from_node 到 to_node 的依赖关系"""
        if from_node.key not in self.edges:
            self.edges[from_node.key] = []  # 如果没有该节点的边，则创建一个空列表
        self.edges[from_node.key].append(to_node.key)  # 添加目标节点的标识符
        from_node.add_next(to_node)  # 将目标节点添加为起始节点的后继节点
        to_node.add_prev(from_node)  # 将起始节点添加为目标节点的前驱节点

    def topological_sort(self):
        """对图进行拓扑排序"""
        visited = set()  # 用于记录已访问的节点
        stack = []  # 用于存储排序后的节点顺序

        def visit(k):
            """访问节点并递归地访问其所有后继节点"""
            if k not in visited:
                visited.add(k)  # 标记节点为已访问
                if k in self.edges:
                    for next_node in self.edges[k]:
                        visit(next_node)  # 递归访问所有后继节点
                stack.insert(0, k)  # 将节点插入到栈的最前面，确保拓扑排序顺序

        # 遍历所有节点进行访问
        for key in self.nodes:
            visit(key)

        self.execution_order = stack  # 设置拓扑排序后的节点执行顺序
