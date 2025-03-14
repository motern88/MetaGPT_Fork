#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:29
@Author  : femto Zheng
@File    : make_sk_kernel.py
"""
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)

from metagpt.config2 import config


def make_sk_kernel():
    """
    创建一个 SK Kernel 实例，并根据配置选择适当的聊天服务。

    如果配置中包含 Azure LLM，则使用 AzureChatCompletion 服务。
    如果配置中包含 OpenAI LLM，则使用 OpenAIChatCompletion 服务。

    返回:
        kernel: 配置好的 SK Kernel 实例。
    """
    kernel = sk.Kernel()  # 创建 SK Kernel 实例

    # 检查是否有 Azure LLM 配置
    if llm := config.get_azure_llm():
        # 如果有 Azure LLM 配置，添加 AzureChatCompletion 服务
        kernel.add_chat_service(
            "chat_completion",  # 服务名称
            AzureChatCompletion(llm.model, llm.base_url, llm.api_key),  # 服务实例
        )
    # 检查是否有 OpenAI LLM 配置
    elif llm := config.get_openai_llm():
        # 如果有 OpenAI LLM 配置，添加 OpenAIChatCompletion 服务
        kernel.add_chat_service(
            "chat_completion",  # 服务名称
            OpenAIChatCompletion(llm.model, llm.api_key),  # 服务实例
        )

    return kernel  # 返回配置好的 Kernel 实例
