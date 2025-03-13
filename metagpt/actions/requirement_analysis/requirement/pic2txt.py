#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/27
@Author  : mashenquan
@File    : pic2txt.py
"""
import json
from pathlib import Path
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import encode_image, general_after_log, to_markdown_code_block


@register_tool(include_functions=["run"])
class Pic2Txt(Action):
    """Pic2Txt 处理以下情况：
    给定一些描述用户需求的图片以及上下文描述，输出完整的文本化用户需求。
    """

    async def run(
        self,
        *,
        image_paths: List[str],  # 图片文件路径列表，表示用户需求的图片
        textual_user_requirement: str = "",  # 附带的文本化用户需求片段，可选
        legacy_output: str = "",  # 上次生成的完整文本化用户需求，可选
        evaluation_conclusion: str = "",  # 处理需求后的评估结论，可选
        additional_technical_requirements: str = "",  # 额外的技术要求，可选
    ) -> str:
        """
        根据描述用户需求的图片和文本片段，生成完整的文本化用户需求。

        参数：
            image_paths (List[str]): 包含用户需求的图片路径列表。
            textual_user_requirement (str, 可选): 用户需求的文本片段，与图片内容配合使用。
            legacy_output (str, 可选): 上次生成的完整文本化用户需求，可用于改进。
            evaluation_conclusion (str, 可选): 需求处理后的评估结论。
            additional_technical_requirements (str, 可选): 额外的技术要求信息。

        返回：
            str: 解析图片与文本后生成的完整文本化用户需求。

        异常：
            ValueError: 如果 `image_paths` 为空，则抛出此异常。
            OSError: 如果图片文件读取失败，则抛出此异常。

        示例：
            >>> images = ["requirements/pic/1.png", "requirements/pic/2.png", "requirements/pic/3.png"]
            >>> textual_user_requirements = "用户需求段落 1 ..., ![](1.png)。 段落 2...![](2.png)..."
            >>> action = Pic2Txt()
            >>> intact_textual_user_requirements = await action.run(image_paths=images, textual_user_requirement=textual_user_requirements)
            >>> print(intact_textual_user_requirements)
            "用户需求段落 1 ..., ![...](1.png) 该图片描述了... 段落 2...![...](2.png)..."
        """
        descriptions = {}  # 存储图片描述信息
        for i in image_paths:
            filename = Path(i)
            base64_image = encode_image(filename)  # 将图片编码为 Base64
            rsp = await self._pic2txt(
                "根据图片内容生成一段描述性文字，生成文本的语言应与图片中的语言一致。",
                base64_image=base64_image,
            )
            descriptions[filename.name] = rsp  # 记录图片文件名和对应的描述

        # 组装完整的提示词
        prompt = PROMPT.format(
            textual_user_requirement=textual_user_requirement,
            acknowledge=to_markdown_code_block(val=json.dumps(descriptions), type_="json"),
            legacy_output=to_markdown_code_block(val=legacy_output),
            evaluation_conclusion=evaluation_conclusion,
            additional_technical_requirements=to_markdown_code_block(val=additional_technical_requirements),
        )
        return await self._write(prompt)  # 生成完整的文本化用户需求

    @retry(
        wait=wait_random_exponential(min=1, max=20),  # 失败后等待随机时间重试，最小 1 秒，最大 20 秒
        stop=stop_after_attempt(6),  # 最多重试 6 次
        after=general_after_log(logger),  # 记录日志
    )
    async def _write(self, prompt: str) -> str:
        """调用 LLM 生成最终的文本化用户需求。"""
        rsp = await self.llm.aask(prompt)
        return rsp

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _pic2txt(self, prompt: str, base64_image: str) -> str:
        """调用 LLM 解析图片内容并生成文本描述。"""
        rsp = await self.llm.aask(prompt, images=base64_image)
        return rsp


# 用于生成最终文本化用户需求的提示模板
PROMPT = """
## 文本化用户需求
{textual_user_requirement}

## 确认信息
{acknowledge}

## 以往输出
{legacy_output}

## 评估结论
{evaluation_conclusion}

## 额外技术要求
{additional_technical_requirements}

---
你是一个工具，可以根据用户提供的文本片段和 UI 图片生成完整的文本化用户需求。
"文本化用户需求" 部分提供了一些用户需求的文本片段；
"确认信息" 部分包含这些图片的描述；
"以往输出" 是上次你生成的完整文本化用户需求，你需要在此基础上改进；
"额外技术要求" 规定了生成的文本化用户需求必须满足的附加要求；
你需要将 "确认信息" 中对应图片的描述合并到 "文本化用户需求" 中，使其形成完整、自然、连贯的描述；
请根据 "文本化用户需求" 片段和 UI 图片生成完整的文本化用户需求。
"""
