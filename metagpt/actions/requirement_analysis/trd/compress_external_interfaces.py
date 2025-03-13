#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : compress_external_interfaces.py
@Desc    : The implementation of Chapter 2.1.5 of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import general_after_log


@register_tool(include_functions=["run"])
class CompressExternalInterfaces(Action):
    """CompressExternalInterfaces 处理以下情况：
    1. 给定一段关于外部系统接口的确认文本，从中提取并压缩相关信息。
    """

    @retry(
        wait=wait_random_exponential(min=1, max=20),  # 在 1 到 20 秒之间随机指数回退等待时间
        stop=stop_after_attempt(6),  # 最多重试 6 次
        after=general_after_log(logger),  # 失败后记录日志
    )
    async def run(
            self,
            *,
            acknowledge: str,  # 包含外部系统接口信息的确认文本
    ) -> str:
        """
        从给定的确认文本中提取并压缩外部系统接口的信息。

        参数:
            acknowledge (str): 包含外部系统接口详细信息的自然语言确认文本。

        返回:
            str: 一个压缩版本的外部系统接口信息。

        示例:
            >>> compress_acknowledge = CompressExternalInterfaces()
            >>> acknowledge = "## 接口\\n..."
            >>> available_external_interfaces = await compress_acknowledge.run(acknowledge=acknowledge)
            >>> print(available_external_interfaces)
            ```json\n[\n{\n"id": 1,\n"inputs": {...}]
        """
        return await self.llm.aask(
            msg=acknowledge,
            system_msgs=[
                "提取并压缩外部系统接口的信息。",
                "返回一个 Markdown JSON 格式的列表，每个对象包含：\n"
                '- "id" 键，表示接口 ID;\n'
                '- "inputs" 键，包含输入参数的字典，每个参数由名称和描述组成;\n'
                '- "outputs" 键，包含返回值的字典，每个返回值由名称和描述组成;\n',
            ],
        )
