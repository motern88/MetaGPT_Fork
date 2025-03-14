# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/28
@Author  : mashenquan
@File    : openai.py
@Desc    : mashenquan, 2023/8/28. Separate the `CostManager` class to support user-level cost accounting.
"""

import re
from typing import NamedTuple

from pydantic import BaseModel

from metagpt.logs import logger
from metagpt.utils.token_counter import FIREWORKS_GRADE_TOKEN_COSTS, TOKEN_COSTS


# 定义一个用于存储成本的命名元组
class Costs(NamedTuple):
    total_prompt_tokens: int  # 总的 prompt tokens 数量
    total_completion_tokens: int  # 总的 completion tokens 数量
    total_cost: float  # 总成本
    total_budget: float  # 总预算

# 成本管理器基类
class CostManager(BaseModel):
    """计算使用接口的开销。"""

    total_prompt_tokens: int = 0  # 初始化总的 prompt tokens 数量
    total_completion_tokens: int = 0  # 初始化总的 completion tokens 数量
    total_budget: float = 0  # 总预算
    max_budget: float = 10.0  # 最大预算
    total_cost: float = 0  # 总成本
    token_costs: dict[str, dict[str, float]] = TOKEN_COSTS  # 各种模型的 token 成本

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        更新总成本、prompt tokens 和 completion tokens 的数量。

        参数:
        prompt_tokens (int): 用于 prompt 的 token 数量
        completion_tokens (int): 用于 completion 的 token 数量
        model (str): 使用的模型名称
        """
        if prompt_tokens + completion_tokens == 0 or not model:
            return
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        if model not in self.token_costs:
            logger.warning(f"未找到模型 {model} 在 TOKEN_COSTS 中。")
            return

        cost = (
            prompt_tokens * self.token_costs[model]["prompt"]
            + completion_tokens * self.token_costs[model]["completion"]
        ) / 1000
        self.total_cost += cost
        logger.info(
            f"总运行成本: ${self.total_cost:.3f} | 最大预算: ${self.max_budget:.3f} | "
            f"当前成本: ${cost:.3f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )

    def get_total_prompt_tokens(self):
        """
        获取总的 prompt tokens 数量。

        返回:
        int: 总的 prompt tokens 数量
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        获取总的 completion tokens 数量。

        返回:
        int: 总的 completion tokens 数量
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        获取 API 调用的总成本。

        返回:
        float: API 调用的总成本
        """
        return self.total_cost

    def get_costs(self) -> Costs:
        """获取所有成本数据"""
        return Costs(self.total_prompt_tokens, self.total_completion_tokens, self.total_cost, self.total_budget)


# 自托管的开源 LLM 模型，没有成本
class TokenCostManager(CostManager):
    """开源 LLM 模型是自托管的，因此是免费的，没有成本"""

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        更新总成本、prompt tokens 和 completion tokens 的数量。

        参数:
        prompt_tokens (int): 用于 prompt 的 token 数量
        completion_tokens (int): 用于 completion 的 token 数量
        model (str): 使用的模型名称
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")


# Fireworks 模型的成本管理器
class FireworksCostManager(CostManager):
    def model_grade_token_costs(self, model: str) -> dict[str, float]:
        """
        获取 Fireworks 模型的 token 成本，基于模型的大小。

        参数:
        model (str): 模型名称

        返回:
        dict: 包含模型的 prompt 和 completion token 成本的字典
        """
        def _get_model_size(model: str) -> float:
            size = re.findall(".*-([0-9.]+)b", model)
            size = float(size[0]) if len(size) > 0 else -1
            return size

        if "mixtral-8x7b" in model:
            token_costs = FIREWORKS_GRADE_TOKEN_COSTS["mixtral-8x7b"]
        else:
            model_size = _get_model_size(model)
            if 0 < model_size <= 16:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["16"]
            elif 16 < model_size <= 80:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["80"]
            else:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["-1"]
        return token_costs

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """
        根据 Fireworks 的定价模型，更新总成本、prompt tokens 和 completion tokens 数量。

        参数:
        prompt_tokens (int): 用于 prompt 的 token 数量
        completion_tokens (int): 用于 completion 的 token 数量
        model (str): 使用的模型名称
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        token_costs = self.model_grade_token_costs(model)
        cost = (prompt_tokens * token_costs["prompt"] + completion_tokens * token_costs["completion"]) / 1000000
        self.total_cost += cost
        logger.info(
            f"总运行成本: ${self.total_cost:.4f}, "
            f"当前成本: ${cost:.4f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )