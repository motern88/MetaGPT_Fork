#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 20:58
@Author  : alexanderwu
@File    : embedding.py
"""
from llama_index.embeddings.openai import OpenAIEmbedding

from metagpt.config2 import config


def get_embedding() -> OpenAIEmbedding:
    """获取 OpenAI 的嵌入模型实例。

    该函数首先从配置中获取 OpenAI LLM（大语言模型）实例，并确保其 API 类型被正确设置为 'openai'。如果配置不正确，则抛出异常。

    返回:
        OpenAIEmbedding: 配置好的 OpenAI 嵌入模型实例。

    异常:
        ValueError: 如果未正确设置 OpenAI 配置（即 api_type 不是 'openai'），则抛出此异常。

    示例:
        embedding = get_embedding()
        # 获取一个 OpenAI 嵌入模型实例，用于后续的嵌入操作。
    """
    llm = config.get_openai_llm()
    if llm is None:
        raise ValueError("To use OpenAIEmbedding, please ensure that config.llm.api_type is correctly set to 'openai'.")

    embedding = OpenAIEmbedding(api_key=llm.api_key, api_base=llm.base_url)
    return embedding
