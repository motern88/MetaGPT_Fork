#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : tutorial_assistant.py
@Describe : Actions of the tutorial assistant, including writing directories and document content.
"""

from typing import Dict

from metagpt.actions import Action
from metagpt.prompts.tutorial_assistant import CONTENT_PROMPT, DIRECTORY_PROMPT
from metagpt.utils.common import OutputParser


class WriteDirectory(Action):
    """编写教程目录的动作类。

    参数:
        name: 动作名称。
        language: 输出语言，默认是 "Chinese"。
    """

    name: str = "WriteDirectory"  # 动作名称
    language: str = "Chinese"  # 输出语言，默认为中文

    async def run(self, topic: str, *args, **kwargs) -> Dict:
        """根据主题生成教程目录。

        参数:
            topic: 教程的主题。

        返回:
            包含教程目录信息的字典，格式为：{"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}。
        """
        # 使用模板生成目录请求的提示
        prompt = DIRECTORY_PROMPT.format(topic=topic, language=self.language)
        # 向语言模型发送请求并获取响应
        resp = await self._aask(prompt=prompt)
        # 提取并返回响应中的结构化数据
        return OutputParser.extract_struct(resp, dict)


class WriteContent(Action):
    """编写教程内容的动作类。

    参数:
        name: 动作名称。
        directory: 要写入的目录内容。
        language: 输出语言，默认是 "Chinese"。
    """

    name: str = "WriteContent"  # 动作名称
    directory: dict = dict()  # 目录内容
    language: str = "Chinese"  # 输出语言，默认为中文

    async def run(self, topic: str, *args, **kwargs) -> str:
        """根据目录和主题编写文档内容。

        参数:
            topic: 教程的主题。

        返回:
            编写好的教程内容。
        """
        # 使用模板生成内容请求的提示
        prompt = CONTENT_PROMPT.format(topic=topic, language=self.language, directory=self.directory)
        # 向语言模型发送请求并获取响应
        return await self._aask(prompt=prompt)
