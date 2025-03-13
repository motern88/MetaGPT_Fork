#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 17:25
@Author  : alexanderwu
@File    : searcher.py
@Modified By: mashenquan, 2023-11-1. According to Chapter 2.2.1 and 2.2.2 of RFC 116, change the data type of
        the `cause_by` value in the `Message` to a string to support the new message distribution feature.
"""

from typing import Optional

from pydantic import Field, model_validator

from metagpt.actions import SearchAndSummarize
from metagpt.actions.action_node import ActionNode
from metagpt.actions.action_output import ActionOutput
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.tools.search_engine import SearchEngine


class Searcher(Role):
    """
    表示一个负责为用户提供搜索服务的角色。

    属性:
        name (str): 搜索者的名称。
        profile (str): 角色描述。
        goal (str): 搜索者的目标。
        constraints (str): 搜索者的约束或限制。
        search_engine (SearchEngine): 使用的搜索引擎。
    """

    name: str = Field(default="Alice")  # 搜索者的名称，默认为 Alice
    profile: str = Field(default="Smart Assistant")  # 搜索者的角色描述，默认为智能助手
    goal: str = "为用户提供搜索服务"  # 搜索者的目标，说明该角色的主要任务
    constraints: str = "回答应该丰富且完整"  # 搜索者的约束条件，要求回答内容丰富且完整
    search_engine: Optional[SearchEngine] = None  # 搜索引擎，可选，如果存在则使用该引擎

    @model_validator(mode="after")
    def post_root(self):
        """验证后初始化操作，设置搜索引擎及操作"""
        if self.search_engine:
            # 如果提供了搜索引擎，则创建 SearchAndSummarize 操作
            self.set_actions([SearchAndSummarize(search_engine=self.search_engine, context=self.context)])
        else:
            # 如果没有提供搜索引擎，则使用默认的 SearchAndSummarize 操作
            self.set_actions([SearchAndSummarize])
        return self  # 返回当前角色对象

    async def _act_sp(self) -> Message:
        """执行单一过程中的搜索操作"""
        logger.info(f"{self._setting}: 正在执行 {self.rc.todo}({self.rc.todo.name})")
        # 执行任务并获取响应
        response = await self.rc.todo.run(self.rc.memory.get(k=0))

        # 判断响应类型，并创建对应的消息对象
        if isinstance(response, (ActionOutput, ActionNode)):
            msg = Message(
                content=response.content,
                instruct_content=response.instruct_content,
                role=self.profile,
                cause_by=self.rc.todo,
            )
        else:
            msg = Message(content=response, role=self.profile, cause_by=self.rc.todo)

        # 将消息添加到内存中
        self.rc.memory.add(msg)
        return msg  # 返回消息对象

    async def _act(self) -> Message:
        """决定搜索者的操作模式"""
        return await self._act_sp()  # 调用 _act_sp 执行搜索操作
