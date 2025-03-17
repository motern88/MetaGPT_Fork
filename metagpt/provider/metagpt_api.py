# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : metagpt_api.py
@Desc    : MetaGPT LLM provider.
"""
from openai.types import CompletionUsage

from metagpt.configs.llm_config import LLMType
from metagpt.provider import OpenAILLM
from metagpt.provider.llm_provider_registry import register_provider


@register_provider(LLMType.METAGPT)
class MetaGPTLLM(OpenAILLM):
    """MetaGPT LLM 提供者，继承自 OpenAILLM，专门用于 MetaGPT 模型的实现"""

    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        """
        计算完成请求时的资源使用情况

        当前的计费逻辑基于使用频率。如果将来有基于令牌数量的计费逻辑，
        请根据需要在此处进行相应的调整。

        参数:
            messages: 消息列表，每条消息是一个字典，包含了消息的角色和内容。
            rsp: 模型的响应文本。

        返回:
            CompletionUsage: 返回一个表示请求使用情况的对象，包括提示令牌、完成令牌和总令牌数。
        """
        return CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
