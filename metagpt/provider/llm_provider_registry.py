#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 17:26
@Author  : alexanderwu
@File    : llm_provider_registry.py
"""
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.base_llm import BaseLLM


class LLMProviderRegistry:
    """LLM提供者注册中心，负责管理和获取不同的LLM提供者实例"""

    def __init__(self):
        """初始化注册表，存储提供者"""
        self.providers = {}

    def register(self, key, provider_cls):
        """注册LLM提供者到注册表

        参数:
            key: 提供者的标识符，通常是枚举类型或字符串。
            provider_cls: 提供者类，用于生成LLM实例。
        """
        self.providers[key] = provider_cls

    def get_provider(self, enum: LLMType):
        """根据枚举类型获取相应的提供者实例

        参数:
            enum: LLM类型的枚举，用于获取特定类型的提供者。

        返回:
            provider_cls: 对应LLM类型的提供者类。
        """
        return self.providers[enum]


def register_provider(keys):
    """注册提供者到注册表的装饰器函数"""

    def decorator(cls):
        """装饰器内部函数"""
        if isinstance(keys, list):
            # 如果 keys 是列表，将所有键注册到LLM_REGISTRY
            for key in keys:
                LLM_REGISTRY.register(key, cls)
        else:
            # 如果 keys 是单个键，直接注册
            LLM_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_llm_instance(config: LLMConfig) -> BaseLLM:
    """根据配置创建LLM实例

    参数:
        config: LLM配置对象，包含API类型、模型等信息。

    返回:
        BaseLLM: 生成的LLM实例。
    """
    # 从注册表中获取默认的LLM提供者，并创建实例
    llm = LLM_REGISTRY.get_provider(config.api_type)(config)

    # 如果配置中的系统提示符与提供者的默认值不同，进行调整
    if llm.use_system_prompt and not config.use_system_prompt:
        # 对于像o1系列模型，默认OpenAI提供者的use_system_prompt为True，但对o1-*类型应为False
        llm.use_system_prompt = config.use_system_prompt
    return llm


# 注册表实例
LLM_REGISTRY = LLMProviderRegistry()
