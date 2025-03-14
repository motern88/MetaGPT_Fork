#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : openai_text_to_embedding.py
@Desc    : OpenAI Text-to-Embedding OAS3 api, which provides text-to-embedding functionality.
            For more details, checkout: `https://platform.openai.com/docs/api-reference/embeddings/object`
"""
from typing import List

import aiohttp
import requests
from pydantic import BaseModel, Field

from metagpt.logs import logger


class Embedding(BaseModel):
    """表示从嵌入端点返回的嵌入向量。"""

    object: str  # 对象类型，始终为 "embedding"。
    embedding: List[float]  # 嵌入向量，表示为浮动数值的列表。向量的长度取决于使用的模型，详情请参阅嵌入指南。
    index: int  # 嵌入在嵌入列表中的索引。


class Usage(BaseModel):
    """表示OpenAI API请求中的令牌使用情况。"""

    prompt_tokens: int = 0  # 提示令牌的数量。
    total_tokens: int = 0  # 总令牌的数量（包括提示令牌和生成令牌）。


class ResultEmbedding(BaseModel):
    """表示从 OpenAI 嵌入 API 返回的结果，包括嵌入和使用信息。"""

    class Config:
        alias = {"object_": "object"}  # 配置别名，以便将 "object_" 映射为 "object"。

    object_: str = ""  # 对象类型，通常为 "embedding"。
    data: List[Embedding] = []  # 包含的嵌入数据列表。
    model: str = ""  # 使用的模型名称。
    usage: Usage = Field(default_factory=Usage)  # 使用情况，默认创建一个 Usage 实例。


class OpenAIText2Embedding:
    def __init__(self, api_key: str, proxy: str):
        """
        初始化 OpenAIText2Embedding 类

        :param api_key: OpenAI API 密钥，详细信息请查看：`https://platform.openai.com/account/api-keys`
        :param proxy: 代理设置（如果有）
        """
        self.api_key = api_key
        self.proxy = proxy

    async def text_2_embedding(self, text, model="text-embedding-ada-002"):
        """将文本转换为嵌入

        :param text: 用于生成嵌入的文本。
        :param model: 模型的 ID，例如 'text-embedding-ada-002'。详细信息请查看：`https://api.openai.com/v1/models`。
        :return: 如果成功，返回 `ResultEmbedding` 类的 JSON 对象，否则返回空字典 `{}`。
        """

        proxies = {"proxy": self.proxy} if self.proxy else {}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        data = {"input": text, "model": model}
        url = "https://api.openai.com/v1/embeddings"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, **proxies) as response:
                    data = await response.json()
                    return ResultEmbedding(**data)
        except requests.exceptions.RequestException as e:
            logger.error(f"发生错误:{e}")
        return ResultEmbedding()


# 导出
async def oas3_openai_text_to_embedding(text, openai_api_key: str, model="text-embedding-ada-002", proxy: str = ""):
    """将文本转换为嵌入

    :param text: 用于生成嵌入的文本。
    :param model: 模型的 ID，例如 'text-embedding-ada-002'。详细信息请查看：`https://api.openai.com/v1/models`。
    :param openai_api_key: OpenAI API 密钥，详细信息请查看：`https://platform.openai.com/account/api-keys`
    :param proxy: 代理设置（如果有）
    :return: 如果成功，返回 `ResultEmbedding` 类的 JSON 对象，否则返回空字典 `{}`。
    """
    if not text:
        return ""
    return await OpenAIText2Embedding(api_key=openai_api_key, proxy=proxy).text_2_embedding(text, model=model)