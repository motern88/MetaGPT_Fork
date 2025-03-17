#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : text_to_embedding.py
@Desc    : Text-to-Embedding skill, which provides text-to-embedding functionality.
"""
from typing import Optional

from metagpt.config2 import Config
from metagpt.tools.openai_text_to_embedding import oas3_openai_text_to_embedding


async def text_to_embedding(text, model="text-embedding-ada-002", config: Optional[Config] = None):
    """文本转向量嵌入（Embedding）

    :param text: 用于生成嵌入向量的文本。
    :param model: 选择用于嵌入的模型，默认为 'text-embedding-ada-002'。
                  可选模型列表详见：https://api.openai.com/v1/models。
    :param config: OpenAI 配置对象，包含 API 密钥等信息。
                   详细信息请参考：https://platform.openai.com/account/api-keys。
    :return: 若成功，则返回 :class:`ResultEmbedding` 类的 JSON 对象，否则返回 `{}`。
    """
    config = config if config else Config.default()  # 若未提供配置，则使用默认配置
    openai_api_key = config.get_openai_llm().api_key  # 获取 OpenAI API 密钥
    proxy = config.get_openai_llm().proxy  # 获取代理地址（若有）

    # 调用 OpenAI API 生成文本的嵌入向量
    return await oas3_openai_text_to_embedding(text, model=model, openai_api_key=openai_api_key, proxy=proxy)