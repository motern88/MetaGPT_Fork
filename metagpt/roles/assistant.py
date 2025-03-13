#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/7
@Author  : mashenquan
@File    : assistant.py
@Desc   : 我正在尝试将 UML 中的某些符号概念引入到 MetaGPT 中，使其能够通过符号串联自由构建流程。
        同时，我也在努力使这些符号可配置和标准化，从而使构建流程的过程更加便捷。
        有关活动图中 fork 节点的更多信息，请参见：https://www.uml-diagrams.org/activity-diagrams.html。
        此文件定义了一个 fork 风格的元角色，能够根据配置文件在运行时生成任意角色。
@Modified By: mashenquan, 2023/8/22. 为 _think 的返回值提供了定义：返回 false 表示无法继续推理。

"""
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from metagpt.actions.skill_action import ArgumentsParingAction, SkillAction
from metagpt.actions.talk_action import TalkAction
from metagpt.learn.skill_loader import SkillsDeclaration
from metagpt.logs import logger
from metagpt.memory.brain_memory import BrainMemory
from metagpt.roles import Role
from metagpt.schema import Message


class MessageType(Enum):
    Talk = "TALK"  # 定义消息类型为 TALK
    Skill = "SKILL"  # 定义消息类型为 SKILL


class Assistant(Role):
    """为解决常见问题提供帮助的助手角色"""

    name: str = "Lily"  # 助手的名字
    profile: str = "An assistant"  # 助手角色简介
    goal: str = "Help to solve problem"  # 助手的目标是帮助解决问题
    constraints: str = "Talk in {language}"  # 限制条件：使用指定的语言进行对话
    desc: str = ""  # 描述，初始化为空
    memory: BrainMemory = Field(default_factory=BrainMemory)  # 助手的记忆，使用 BrainMemory 类
    skills: Optional[SkillsDeclaration] = None  # 助手的技能，初始化为 None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        language = kwargs.get("language") or self.context.kwargs.language  # 获取语言参数
        self.constraints = self.constraints.format(language=language)  # 设置语言限制条件

    async def think(self) -> bool:
        """进行思考，逐步完成任务"""
        last_talk = await self.refine_memory()  # 获取上一次的对话内容
        if not last_talk:
            return False  # 如果没有对话内容，返回 False
        if not self.skills:  # 如果没有技能声明
            skill_path = Path(self.context.kwargs.SKILL_PATH) if self.context.kwargs.SKILL_PATH else None  # 获取技能路径
            self.skills = await SkillsDeclaration.load(skill_yaml_file_name=skill_path)  # 加载技能

        prompt = ""
        skills = self.skills.get_skill_list(context=self.context)  # 获取技能列表
        for desc, name in skills.items():
            prompt += f"If the text explicitly want you to {desc}, return `[SKILL]: {name}` brief and clear. For instance: [SKILL]: {name}\n"
        prompt += 'Otherwise, return `[TALK]: {talk}` brief and clear. For instance: if {talk} is "xxxx" return [TALK]: xxxx\n\n'
        prompt += f"Now what specific action is explicitly mentioned in the text: {last_talk}\n"
        rsp = await self.llm.aask(prompt, ["You are an action classifier"], stream=False)  # 向 LLM 请求结果
        logger.info(f"THINK: {prompt}\n, THINK RESULT: {rsp}\n")  # 打印思考的结果
        return await self._plan(rsp, last_talk=last_talk)  # 规划行动

    async def act(self) -> Message:
        """根据当前的任务执行行动"""
        result = await self.rc.todo.run()  # 执行待办任务
        if not result:
            return None  # 如果没有结果，返回 None
        if isinstance(result, str):
            msg = Message(content=result, role="assistant", cause_by=self.rc.todo)  # 如果是字符串，创建消息
        elif isinstance(result, Message):
            msg = result  # 如果是消息对象，直接使用
        else:
            msg = Message(content=result.content, instruct_content=result.instruct_content, cause_by=type(self.rc.todo))  # 其他类型转换为消息
        self.memory.add_answer(msg)  # 将消息添加到记忆中
        return msg

    async def talk(self, text):
        """进行对话并将其添加到记忆中"""
        self.memory.add_talk(Message(content=text))  # 将对话内容添加到记忆

    async def _plan(self, rsp: str, **kwargs) -> bool:
        """根据思考的结果进行规划"""
        skill, text = BrainMemory.extract_info(input_string=rsp)  # 从结果中提取技能和文本
        handlers = {
            MessageType.Talk.value: self.talk_handler,  # 如果是 TALK 类型，调用 talk_handler
            MessageType.Skill.value: self.skill_handler,  # 如果是 SKILL 类型，调用 skill_handler
        }
        handler = handlers.get(skill, self.talk_handler)  # 获取相应的处理函数
        return await handler(text, **kwargs)  # 执行处理函数

    async def talk_handler(self, text, **kwargs) -> bool:
        """处理 TALK 类型的消息"""
        history = self.memory.history_text  # 获取历史对话
        text = kwargs.get("last_talk") or text  # 获取最后一次对话的文本
        self.set_todo(
            TalkAction(i_context=text, knowledge=self.memory.get_knowledge(), history_summary=history, llm=self.llm)  # 设置待办任务
        )
        return True

    async def skill_handler(self, text, **kwargs) -> bool:
        """处理 SKILL 类型的消息"""
        last_talk = kwargs.get("last_talk")  # 获取最后一次对话
        skill = self.skills.get_skill(text)  # 获取技能
        if not skill:
            logger.info(f"skill not found: {text}")  # 如果没有找到技能，记录日志
            return await self.talk_handler(text=last_talk, **kwargs)  # 调用 talk_handler 进行对话处理
        action = ArgumentsParingAction(skill=skill, llm=self.llm, ask=last_talk)  # 创建参数解析动作
        await action.run(**kwargs)  # 执行动作
        if action.args is None:
            return await self.talk_handler(text=last_talk, **kwargs)  # 如果没有参数，调用 talk_handler 处理对话
        self.set_todo(SkillAction(skill=skill, args=action.args, llm=self.llm, name=skill.name, desc=skill.description))  # 设置待办任务
        return True

    async def refine_memory(self) -> str:
        """精炼记忆，将相关的对话内容提取并合并"""
        last_talk = self.memory.pop_last_talk()  # 获取最后一次对话
        if last_talk is None:  # 如果没有对话内容，返回 None
            return None
        if not self.memory.is_history_available:  # 如果没有历史可用，返回最后一次对话
            return last_talk
        history_summary = await self.memory.summarize(max_words=800, keep_language=True, llm=self.llm)  # 总结历史
        if last_talk and await self.memory.is_related(text1=last_talk, text2=history_summary, llm=self.llm):
            # 如果最后一次对话和历史内容相关，合并内容
            merged = await self.memory.rewrite(sentence=last_talk, context=history_summary, llm=self.llm)
            return f"{merged} {last_talk}"

        return last_talk

    def get_memory(self) -> str:
        """获取当前的记忆"""
        return self.memory.model_dump_json()  # 返回记忆的 JSON 格式

    def load_memory(self, m):
        """加载记忆"""
        try:
            self.memory = BrainMemory(**m)  # 尝试加载记忆
        except Exception as e:
            logger.exception(f"load error:{e}, data:{m}")  # 如果加载失败，记录错误日志
