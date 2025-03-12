#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/30 17:13
@Author  : alexanderwu
@File    : solver.py
"""
from abc import abstractmethod

from metagpt.actions.action_graph import ActionGraph
from metagpt.provider.base_llm import BaseLLM
from metagpt.strategy.search_space import SearchSpace


class BaseSolver:
    """AbstractSolver: 定义求解器的接口类。

    该类是所有具体求解器的基类，提供了解决问题的基本接口。
    """

    def __init__(self, graph: ActionGraph, search_space: SearchSpace, llm: BaseLLM, context):
        """
        初始化求解器。

        :param graph: ActionGraph，表示问题的动作图。
        :param search_space: SearchSpace，表示搜索空间。
        :param llm: BaseLLM，表示用于求解的语言模型。
        :param context: Context，表示求解所需的上下文信息。
        """
        self.graph = graph
        self.search_space = search_space
        self.llm = llm
        self.context = context

    @abstractmethod
    async def solve(self):
        """抽象方法，用于求解问题。

        子类需要实现此方法来解决具体问题。
        """


class NaiveSolver(BaseSolver):
    """NaiveSolver: 一个简单的求解器，依次遍历图中的所有节点并执行它们。

    该求解器会根据图的拓扑排序依次执行每个节点的操作。
    """

    async def solve(self):
        self.graph.topological_sort()  # 对图进行拓扑排序
        for key in self.graph.execution_order:  # 按拓扑排序遍历图的节点
            op = self.graph.nodes[key]
            await op.fill(req=self.context, llm=self.llm, mode="root")  # 执行节点操作


class TOTSolver(BaseSolver):
    """TOTSolver: 思维树（Tree of Thought）求解器

    该求解器尚未实现。
    """

    async def solve(self):
        raise NotImplementedError  # 尚未实现


class DataInterpreterSolver(BaseSolver):
    """DataInterpreterSolver: 在图中执行代码的求解器

    该求解器尚未实现。
    """

    async def solve(self):
        raise NotImplementedError  # 尚未实现


class ReActSolver(BaseSolver):
    """ReActSolver: ReAct 算法求解器

    该求解器尚未实现。
    """

    async def solve(self):
        raise NotImplementedError  # 尚未实现



class IOSolver(BaseSolver):
    """IOSolver: 使用语言模型直接解决问题的求解器

    该求解器尚未实现。
    """

    async def solve(self):
        raise NotImplementedError  # 尚未实现


class COTSolver(BaseSolver):
    """COTSolver: 思维链（Chain of Thought）求解器

    该求解器尚未实现。
    """

    async def solve(self):
        raise NotImplementedError  # 尚未实现
