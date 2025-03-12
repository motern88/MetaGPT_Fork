#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/30 17:15
@Author  : alexanderwu
@File    : search_space.py
"""


class SearchSpace:
    """SearchSpace: 用于定义一个搜索空间，搜索空间中的节点是 ActionNode 类。

    该类用于管理一个由 ActionNode 类实例构成的搜索空间。搜索空间中的每个节点通过一个唯一的 key 来标识。
    """

    def __init__(self):
        """初始化搜索空间，创建一个空字典用于存储节点"""
        self.search_space = {}

    def add_node(self, node):
        """向搜索空间中添加一个节点

        Args:
            node (ActionNode): 需要添加到搜索空间中的节点。
        """
        self.search_space[node.key] = node

    def get_node(self, key):
        """根据给定的 key 获取对应的节点

        Args:
            key (str): 节点的唯一标识符。

        Returns:
            ActionNode: 对应的 ActionNode 节点。
        """
        return self.search_space[key]
