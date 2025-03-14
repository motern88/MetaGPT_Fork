#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/17
@Author  : mashenquan
@File    : openai_text_to_image.py
@Desc    : OpenAI Text-to-Image OAS3 api, which provides text-to-image functionality.
"""

import aiohttp
import requests

from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM


class OpenAIText2Image:
    def __init__(self, llm: BaseLLM):
        """初始化 OpenAIText2Image 类

        :param llm: 一个 LLM 实例，用于与模型交互。
        """
        self.llm = llm

    async def text_2_image(self, text, size_type="1024x1024"):
        """将文本转换为图像

        :param text: 用于生成图像的文本。
        :param size_type: 图像的尺寸，可以是 ['256x256', '512x512', '1024x1024'] 中的一个。
        :return: 返回经过 Base64 编码的图像数据。
        """
        try:
            # 调用 LLM 的图像生成接口
            result = await self.llm.aclient.images.generate(prompt=text, n=1, size=size_type)
        except Exception as e:
            # 如果发生异常，记录错误并返回空字符串
            logger.error(f"发生错误: {e}")
            return ""
        if result and len(result.data) > 0:
            # 获取图像数据并返回
            return await OpenAIText2Image.get_image_data(result.data[0].url)
        return ""

    @staticmethod
    async def get_image_data(url):
        """从 URL 获取图像数据并将其编码为 Base64 格式

        :param url: 图像的 URL 地址
        :return: 返回 Base64 编码的图像数据。
        """
        try:
            # 使用 aiohttp 请求图像
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()  # 如果响应是 4xx 或 5xx 错误，抛出异常
                    image_data = await response.read()
            return image_data  # 返回图像数据

        except requests.exceptions.RequestException as e:
            # 如果发生请求异常，记录错误并返回 0
            logger.error(f"发生错误: {e}")
            return 0


# 导出函数
async def oas3_openai_text_to_image(text, size_type: str = "1024x1024", llm: BaseLLM = None):
    """将文本转换为图像

    :param text: 用于生成图像的文本。
    :param size_type: 图像的尺寸，可以是 ['256x256', '512x512', '1024x1024'] 中的一个。
    :param llm: LLM 实例
    :return: 返回经过 Base64 编码的图像数据。
    """
    if not text:
        return ""
    return await OpenAIText2Image(llm).text_2_image(text, size_type=size_type)
