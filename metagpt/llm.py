#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:45
@Author  : alexanderwu
@File    : llm.py
"""
from typing import Optional

from metagpt.configs.llm_config import LLMConfig
from metagpt.context import Context
from metagpt.provider.base_llm import BaseLLM


def LLM(llm_config: Optional[LLMConfig] = None, context: Context = None) -> BaseLLM:
    """
    获取默认的 LLM（大语言模型）提供者。

    参数：
        llm_config (Optional[LLMConfig])：可选的 LLM 配置，如果提供，将使用该配置创建 LLM。
        context (Context)：可选的上下文对象，若未提供则创建一个新的 Context 实例。

    返回：
        BaseLLM：根据配置返回对应的 LLM 实例。
    """
    ctx = context or Context()  # 如果未提供 context，则创建一个新的 Context 实例
    if llm_config is not None:
        return ctx.llm_with_cost_manager_from_llm_config(llm_config)  # 使用 LLM 配置创建 LLM，并启用成本管理
    return ctx.llm()  # 返回默认的 LLM
