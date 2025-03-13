#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/19 15:02
@Author  : DevXiaolan
@File    : prepare_interview.py
"""
from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode

# 定义一个名为 QUESTIONS 的 ActionNode，用于生成面试问题
QUESTIONS = ActionNode(
    key="Questions",  # 节点的键
    expected_type=list[str],  # 期望的类型是字符串列表
    instruction="""Role: 你是我们公司的一名面试官，擅长前端或后端开发；
Requirement: 提供一组问题，供面试官根据面试者的简历提问；
Attention: 请以 markdown 格式提供问题列表，至少包含 10 个问题。""",  # 指定的指令说明
    example=["1. What ...", "2. How ..."],  # 示例问题
)


class PrepareInterview(Action):
    """PrepareInterview 类用于准备面试问题。"""

    name: str = "PrepareInterview"  # 动作的名称

    async def run(self, context):
        """执行准备面试的问题生成操作。"""
        # 使用 QUESTIONS 填充问题，传入上下文和 LLM（大语言模型）进行问题生成
        return await QUESTIONS.fill(req=context, llm=self.llm)
