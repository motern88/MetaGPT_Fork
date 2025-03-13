# -*- coding: utf-8 -*-
# @Date    : 12/25/2023 9:14 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from enum import Enum

from pydantic import BaseModel, Field

from metagpt.strategy.base import BaseEvaluator, BaseParser

# 定义方法选择的枚举类
class MethodSelect(Enum):
    SAMPLE = "sample"  # 采样方法
    GREEDY = "greedy"  # 贪心方法

# 定义搜索策略的枚举类
class Strategy(Enum):
    BFS = "BFS"  # 广度优先搜索
    DFS = "DFS"  # 深度优先搜索
    MCTS = "MCTS"  # 蒙特卡洛树搜索

# 配置类，用于控制思维求解器的参数
class ThoughtSolverConfig(BaseModel):
    max_steps: int = 3  # 最大搜索步数
    method_select: MethodSelect = MethodSelect.GREEDY  # 选择的搜索方法（贪心或采样）
    n_generate_sample: int = 5  # 每个节点生成的样本数
    n_select_sample: int = 3  # 每条路径选择的样本数
    n_solution_sample: int = 5  # 仅适用于深度优先搜索（DFS）的解样本数
    parser: BaseParser = Field(default_factory=BaseParser)  # 解析器实例
    evaluator: BaseEvaluator = Field(default_factory=BaseEvaluator)  # 评估器实例
