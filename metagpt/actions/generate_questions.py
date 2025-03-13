#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : generate_questions.py
"""
from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode

QUESTIONS = ActionNode(
    key="Questions",  # 问题的关键字
    expected_type=list[str],  # 期望的类型是字符串列表
    instruction="任务：根据上下文提出更多关于你感兴趣的细节问题，字数限制为 150 个字。请在此提供你想要询问的具体细节。",
    example=["1. What ...", "2. How ...", "3. ..."],  # 问题的示例
)

class GenerateQuestions(Action):
    """此类允许 LLM 基于特定的“##TOPIC”(讨论主题)和“##RECORD”(讨论记录)进一步挖掘值得注意的细节，从而深化讨论。"""

    name: str = "GenerateQuestions"  # 动作名称为 "GenerateQuestions"

    async def run(self, context) -> ActionNode:
        """基于提供的上下文生成问题"""
        return await QUESTIONS.fill(req=context, llm=self.llm)  # 使用 LLM 填充并返回问题
