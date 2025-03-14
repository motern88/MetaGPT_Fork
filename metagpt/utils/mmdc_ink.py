#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : alitrack
@File    : mermaid.py
"""
import base64
from typing import List, Optional

from aiohttp import ClientError, ClientSession

from metagpt.logs import logger


async def mermaid_to_file(mermaid_code, output_file_without_suffix, suffixes: Optional[List[str]] = None):
    """将Mermaid代码转换为各种文件格式。

    参数：
        mermaid_code (str): 要转换的Mermaid代码。
        output_file_without_suffix (str): 输出文件名，不带后缀。
        suffixes (Optional[List[str]], 可选): 要生成的文件后缀，支持 "png"、"pdf" 和 "svg"。默认为 ["png"]。

    返回：
        int: 如果转换成功，返回0；如果转换失败，返回-1。
    """
    # 将Mermaid代码编码为base64字符串
    encoded_string = base64.b64encode(mermaid_code.encode()).decode()

    # 设置文件后缀，如果没有指定，则默认为 "png"
    suffixes = suffixes or ["png"]

    # 遍历每种后缀格式，生成对应的文件
    for suffix in suffixes:
        # 生成输出文件路径
        output_file = f"{output_file_without_suffix}.{suffix}"

        # 根据文件后缀决定请求的路径类型：svg或img
        path_type = "svg" if suffix == "svg" else "img"

        # 构建URL，其中包含编码后的Mermaid代码
        url = f"https://mermaid.ink/{path_type}/{encoded_string}"

        # 使用异步请求获取Mermaid渲染结果
        async with ClientSession() as session:
            try:
                # 发送GET请求并获取响应
                async with session.get(url) as response:
                    if response.status == 200:
                        # 如果请求成功，读取响应内容并写入文件
                        text = await response.content.read()
                        with open(output_file, "wb") as f:
                            f.write(text)
                        logger.info(f"正在生成 {output_file}..")
                    else:
                        # 如果请求失败，记录错误并返回-1
                        logger.error(f"生成 {output_file} 失败")
                        return -1
            except ClientError as e:
                # 捕获网络错误并记录
                logger.error(f"网络错误: {e}")
                return -1

    # 如果所有后缀的转换都成功，返回0
    return 0
