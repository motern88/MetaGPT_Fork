# -*- coding: utf-8 -*-
# @Date    : 12/23/2023 4:51 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.strategy.base import ThoughtNode, ThoughtTree
from metagpt.strategy.tot_schema import MethodSelect, Strategy, ThoughtSolverConfig
from metagpt.utils.common import CodeParser

OUTPUT_FORMAT = """
Each output should be strictly a list of nodes, in json format, like this:
```json
    [
        {
            "node_id": str = "unique identifier for a solution, can be an ordinal",
            "node_state_instruction": "specified sample of solution",
        },
        ...
    ]
```
"""


class ThoughtSolverBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    thought_tree: Optional[ThoughtTree] = Field(default=None)
    llm: BaseLLM = Field(default_factory=LLM, exclude=True)
    config: ThoughtSolverConfig = Field(default_factory=ThoughtSolverConfig)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm.use_system_prompt = False  # 关闭 LLM 的系统提示

    async def solve(self, init_prompt):
        """
        解决问题的方法，子类必须实现。
        """
        raise NotImplementedError("Subclasses must implement the solve method")

    async def generate_thoughts(self, current_state="", current_node=None) -> List[ThoughtNode]:
        """
        生成思维节点（子节点）。

        参数：
            current_state (str): 当前状态。
            current_node (ThoughtNode): 当前思维树中的节点。

        返回：
            List[ThoughtNode]: 生成的思维节点列表。
        """
        state_prompt = self.config.parser.propose(
            current_state=current_state, **{"n_generate_sample": self.config.n_generate_sample}
        )
        rsp = await self.llm.aask(msg=state_prompt + "\n" + OUTPUT_FORMAT)
        thoughts = CodeParser.parse_code(text=rsp)
        thoughts = eval(thoughts)  # 解析 LLM 生成的代码
        return self.thought_tree.update_node(thoughts, current_node=current_node)

    async def evaluate_node(self, node, parent_value) -> None:
        """
        评估节点，更新其状态和值。

        参数：
            node (ThoughtNode): 需要评估的节点。
            parent_value (float): 父节点的值。
        """
        eval_prompt = self.config.parser.value(input=node.name, **{"node_id": node.id})
        evaluation = await self.llm.aask(msg=eval_prompt)

        value = self.config.evaluator(evaluation, **{"node_id": node.id})
        status = self.config.evaluator.status_verify(value)

        node.update_valid_status(status=status)
        node.update_value(parent_value + value)  # 更新节点值

    def select_nodes(self, thought_nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        """
        选择合适的思维节点。

        参数：
            thought_nodes (List[ThoughtNode]): 待选择的节点列表。

        返回：
            List[ThoughtNode]: 选择的节点。
        """
        nodes = []
        if self.config.method_select == MethodSelect.SAMPLE:
            raise NotImplementedError
        elif self.config.method_select == MethodSelect.GREEDY:
            nodes = sorted(thought_nodes, key=lambda x: x.value, reverse=True)[: self.config.n_select_sample]
        for node in thought_nodes:
            if node not in nodes:
                node.parent = None  # 移除未选择的节点
        return nodes

    def update_solution(self):
        """
        选择最高分的思维路径。

        返回：
            Tuple[List[ThoughtNode], List[str]]: 最优解的思维路径及节点。
        """
        best_node = max(self.thought_tree.all_nodes, key=lambda x: x.value, default=None)
        best_solution_path = self.thought_tree.parse_node_path(best_node)
        return [best_node], best_solution_path


class BFSSolver(ThoughtSolverBase):
    async def solve(self, init_prompt=""):
        """
        使用 BFS 解决问题。

        参数:
            init_prompt (str): 初始提示词。

        返回:
            List[str]: 通过 BFS 得到的最佳解决方案路径。
        """
        root = ThoughtNode(init_prompt)
        self.thought_tree = ThoughtTree(root)
        current_nodes = [root]
        for step in range(self.config.max_steps):
            solutions = await self._bfs_build(current_nodes)

            selected_nodes = self.select_nodes(solutions)
            current_nodes = selected_nodes

            self.thought_tree.show()

        best_solution, best_solution_path = self.update_solution()
        logger.info(f"最佳解决方案: {best_solution_path}")
        return best_solution_path

    async def _bfs_build(self, current_nodes):
        """
        通过 BFS 方式构建思维树。

        参数:
            current_nodes (List[ThoughtNode]): 需要扩展的当前节点。

        返回:
            List[ThoughtNode]: 经过扩展后得到的解决方案节点。
        """
        tasks = []
        for node in current_nodes:
            current_state = self.config.parser(node.name)
            current_value = node.value
            tasks.append(self.generate_and_evaluate_nodes(current_state, current_value, node))

        thought_nodes_list = await asyncio.gather(*tasks)
        solutions = [child_node for thought_nodes in thought_nodes_list for child_node in thought_nodes]
        return solutions


class DFSSolver(ThoughtSolverBase):
    async def solve(self, init_prompt=""):
        """
        使用深度优先搜索（DFS）求解问题。
        """
        root = ThoughtNode(init_prompt)
        self.thought_tree = ThoughtTree(root)
        for _ in range(self.config.n_solution_sample):
            await self._dfs(root)

        best_solution, best_solution_path = self.update_solution()
        logger.info(f"最佳解决方案：{best_solution_path}")
        return best_solution_path

    async def _dfs(self, root_node):
        """
        使用 DFS 方式遍历思维树。

        参数:
            root_node (ThoughtNode): 思维树的根节点。

        返回:
            List[str]: 通过 DFS 得到的解决方案路径。
        """
        impossible_state_cnt = 0
        node = root_node
        for step in range(self.max_steps):
            current_state = self.config.parser(node.name)
            current_value = node.value
            thought_nodes = await self.generate_thoughts(current_state, current_node=node)
            await self.evaluate_node(thought_nodes[0], parent_value=current_value)
            if thought_nodes[0].valid_status is False:
                impossible_state_cnt += 1
            if impossible_state_cnt >= 2:
                logger.info("已达到不可行状态，终止")
                break
            node = thought_nodes[0]
        _solution_path = self.thought_tree.parse_node_path(node)
        self.thought_tree.show()
        return _solution_path


# MCTS（蒙特卡洛树搜索）求解器
class MCTSSolver(ThoughtSolverBase):
    async def solve(self, init_prompt=""):
        raise NotImplementedError


# 思维树求解器
class TreeofThought(BaseModel):
    config: ThoughtSolverConfig = Field(default_factory=ThoughtSolverConfig)
    solver: ThoughtSolverBase = Field(default_factory=ThoughtSolverBase)
    strategy: Strategy = Field(default=Strategy.BFS)

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型字段

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._initialize_solver(self.strategy)

    def _initialize_solver(self, strategy):
        """
        根据策略初始化求解器。

        参数:
            strategy (Strategy): 需要使用的求解策略。

        返回:
            ThoughtSolverBase: 初始化的求解器实例。
        """
        if strategy == Strategy.BFS:
            self.solver = BFSSolver(config=self.config)
        elif strategy == Strategy.DFS:
            self.solver = DFSSolver(config=self.config)
        elif strategy == Strategy.MCTS:
            self.solver = MCTSSolver(config=self.config)
        else:
            raise NotImplementedError(f"无效策略: {strategy}，仅支持 BFS/DFS/MCTS！")

    async def solve(self, init_prompt=""):
        """
        使用指定策略解决问题。

        参数:
            init_prompt (str): 初始提示词。

        返回:
            Any: 选定策略的求解结果。
        """
        await self.solver.solve(init_prompt)
