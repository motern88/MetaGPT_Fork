#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/28
@Author  : mashenquan
@File    : talk_action.py
@Desc    : Act as it’s a talk
"""
from typing import Optional

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.schema import Message

class TalkAction(Action):
    i_context: str  # 存储当前对话的上下文信息
    history_summary: str = ""  # 存储历史对话总结
    knowledge: str = ""  # 存储知识背景
    rsp: Optional[Message] = None  # 存储返回的消息

    @property
    def agent_description(self):
        # 获取代理的描述，通常是角色的身份描述
        return self.context.kwargs.agent_description

    @property
    def language(self):
        # 获取语言，如果没有指定则使用配置中的语言
        return self.context.kwargs.language or self.config.language

    @property
    def prompt(self):
        # 生成给模型的提示文本
        prompt = ""
        if self.agent_description:
            # 如果有代理描述，加入描述信息
            prompt = (
                f"You are {self.agent_description}. Your responses should align with the role-play agreement, "
                f"maintaining the character's persona and habits. When faced with unrelated questions, playfully "
                f"decline to answer without revealing your AI nature to preserve the character's image.\n\n"
            )
        # 如果有知识背景，则加入知识部分
        prompt += f"Knowledge:\n{self.knowledge}\n\n" if self.knowledge else ""
        # 加入历史对话总结
        prompt += f"{self.history_summary}\n\n"
        prompt += (
            "If the information is insufficient, you can search in the historical conversation or knowledge above.\n"
        )
        # 根据语言属性生成相应的提示
        language = self.language
        prompt += (
            f"Answer the following questions strictly in {language}, and the answers must follow the Markdown format.\n "
            f"{self.i_context}"
        )
        logger.debug(f"PROMPT: {prompt}")  # 打印生成的提示信息
        return prompt

    @property
    def prompt_gpt4(self):
        # 为 GPT-4 生成特定的提示模板
        kvs = {
            "{role}": self.agent_description or "",  # 角色描述
            "{history}": self.history_summary or "",  # 历史对话总结
            "{knowledge}": self.knowledge or "",  # 知识背景
            "{language}": self.language,  # 语言
            "{ask}": self.i_context,  # 当前问题
        }
        # 使用模板格式替换相应内容
        prompt = TalkActionPrompt.FORMATION_LOOSE
        for k, v in kvs.items():
            prompt = prompt.replace(k, v)
        logger.info(f"PROMPT: {prompt}")  # 打印生成的提示信息
        return prompt

    # async def run_old(self, *args, **kwargs) -> ActionOutput:
    #     prompt = self.prompt
    #     rsp = await self.llm.aask(msg=prompt, system_msgs=[])
    #     logger.debug(f"PROMPT:{prompt}\nRESULT:{rsp}\n")
    #     self._rsp = ActionOutput(content=rsp)
    #     return self._rsp

    @property
    def aask_args(self):
        # 获取 AASK 请求所需的参数
        language = self.language
        system_msgs = [
            f"You are {self.agent_description}.",  # 角色描述
            "Your responses should align with the role-play agreement, "
            "maintaining the character's persona and habits. When faced with unrelated questions, playfully "
            "decline to answer without revealing your AI nature to preserve the character's image.",
            "If the information is insufficient, you can search in the context or knowledge.",
            f"Answer the following questions strictly in {language}, and the answers must follow the Markdown format.",
        ]
        format_msgs = []
        # 如果有知识背景，加入知识消息
        if self.knowledge:
            format_msgs.append({"role": "assistant", "content": self.knowledge})
        # 如果有历史总结，加入历史总结消息
        if self.history_summary:
            format_msgs.append({"role": "assistant", "content": self.history_summary})
        # 返回消息的上下文、格式化消息和系统消息
        return self.i_context, format_msgs, system_msgs

    async def run(self, with_message=None, **kwargs) -> Message:
        # 执行操作，生成最终的回复
        msg, format_msgs, system_msgs = self.aask_args  # 获取 AASK 请求参数
        rsp = await self.llm.aask(msg=msg, format_msgs=format_msgs, system_msgs=system_msgs, stream=False)
        # 将回复内容封装为消息对象
        self.rsp = Message(content=rsp, role="assistant", cause_by=self)
        return self.rsp  # 返回消息


class TalkActionPrompt:
    FORMATION = """Formation: "Capacity and role" defines the role you are currently playing;
  "[HISTORY_BEGIN]" and "[HISTORY_END]" tags enclose the historical conversation;
  "[KNOWLEDGE_BEGIN]" and "[KNOWLEDGE_END]" tags enclose the knowledge may help for your responses;
  "Statement" defines the work detail you need to complete at this stage;
  "[ASK_BEGIN]" and [ASK_END] tags enclose the questions;
  "Constraint" defines the conditions that your responses must comply with.
  "Personality" defines your language style。
  "Insight" provides a deeper understanding of the characters' inner traits.
  "Initial" defines the initial setup of a character.

Capacity and role: {role}
Statement: Your responses should align with the role-play agreement, maintaining the
 character's persona and habits. When faced with unrelated questions, playfully decline to answer without revealing
 your AI nature to preserve the character's image.

[HISTORY_BEGIN]

{history}

[HISTORY_END]

[KNOWLEDGE_BEGIN]

{knowledge}

[KNOWLEDGE_END]

Statement: If the information is insufficient, you can search in the historical conversation or knowledge.
Statement: Unless you are a language professional, answer the following questions strictly in {language}
, and the answers must follow the Markdown format. Strictly excluding any tag likes "[HISTORY_BEGIN]"
, "[HISTORY_END]", "[KNOWLEDGE_BEGIN]", "[KNOWLEDGE_END]" in responses.
 

{ask}
"""

    FORMATION_LOOSE = """Formation: "Capacity and role" defines the role you are currently playing;
  "[HISTORY_BEGIN]" and "[HISTORY_END]" tags enclose the historical conversation;
  "[KNOWLEDGE_BEGIN]" and "[KNOWLEDGE_END]" tags enclose the knowledge may help for your responses;
  "Statement" defines the work detail you need to complete at this stage;
  "Constraint" defines the conditions that your responses must comply with.
  "Personality" defines your language style。
  "Insight" provides a deeper understanding of the characters' inner traits.
  "Initial" defines the initial setup of a character.

Capacity and role: {role}
Statement: Your responses should maintaining the character's persona and habits. When faced with unrelated questions
, playfully decline to answer without revealing your AI nature to preserve the character's image. 

[HISTORY_BEGIN]

{history}

[HISTORY_END]

[KNOWLEDGE_BEGIN]

{knowledge}

[KNOWLEDGE_END]

Statement: If the information is insufficient, you can search in the historical conversation or knowledge.
Statement: Unless you are a language professional, answer the following questions strictly in {language}
, and the answers must follow the Markdown format. Strictly excluding any tag likes "[HISTORY_BEGIN]"
, "[HISTORY_END]", "[KNOWLEDGE_BEGIN]", "[KNOWLEDGE_END]" in responses.


{ask}
"""
