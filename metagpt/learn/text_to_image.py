#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : text_to_image.py
@Desc    : Text-to-Image skill, which provides text-to-image functionality.
"""
import base64
from typing import Optional

from metagpt.config2 import Config
from metagpt.const import BASE64_FORMAT
from metagpt.llm import LLM
from metagpt.tools.metagpt_text_to_image import oas3_metagpt_text_to_image
from metagpt.tools.openai_text_to_image import oas3_openai_text_to_image
from metagpt.utils.s3 import S3


async def text_to_image(text, size_type: str = "512x512", config: Optional[Config] = None):
    """文本转图像（Text-to-Image）

    :param text: 用于生成图像的文本描述。
    :param size_type: 生成图像的尺寸：
                      - 如果使用 OpenAI，支持的尺寸包括 ['256x256', '512x512', '1024x1024']；
                      - 如果使用 MetaGPT，支持的尺寸包括 ['512x512', '512x768']。
    :param config: 配置对象（包含 API 相关信息）。
    :return: 以 Base64 编码格式返回生成的图像数据。
    """
    config = config if config else Config.default()  # 若未提供配置，则使用默认配置
    image_declaration = "data:image/png;base64,"  # 图片的 Base64 前缀声明

    model_url = config.metagpt_tti_url  # 获取 MetaGPT 模型 URL
    if model_url:
        # 使用 MetaGPT 进行文本生成图像
        binary_data = await oas3_metagpt_text_to_image(text, size_type, model_url)
    elif config.get_openai_llm():
        # 使用 OpenAI 进行文本生成图像
        llm = LLM(llm_config=config.get_openai_llm())
        binary_data = await oas3_openai_text_to_image(text, size_type, llm=llm)
    else:
        raise ValueError("缺少必要的参数。")  # 若没有可用的模型，抛出异常

    # 将图像数据转换为 Base64 编码
    base64_data = base64.b64encode(binary_data).decode("utf-8")

    # 使用 S3 存储图像并生成访问链接
    s3 = S3(config.s3)
    url = await s3.cache(data=base64_data, file_ext=".png", format=BASE64_FORMAT)
    if url:
        return f"![{text}]({url})"  # 返回 Markdown 格式的图片链接
    return image_declaration + base64_data if base64_data else ""  # 若无 S3 存储，则直接返回 Base64 编码数据
