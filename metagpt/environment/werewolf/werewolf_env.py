#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : MG Werewolf Env

from typing import Iterable

from pydantic import Field

from metagpt.environment.base_env import Environment
from metagpt.environment.werewolf.werewolf_ext_env import WerewolfExtEnv
from metagpt.schema import Message


class WerewolfEnv(WerewolfExtEnv, Environment):
    # 环境中的回合计数
    round_cnt: int = Field(default=0)

    def add_roles(self, roles: Iterable["Role"]):
        """增加一批在当前环境的角色
        Add a batch of characters in the current environment
        """
        # 遍历所有角色并将它们添加到环境中，角色名称作为键，因为同一角色可以在多个玩家中共享
        for role in roles:
            self.roles[role.name] = role  # 使用角色名称作为键

        # 为每个角色设置系统消息和环境上下文
        for role in roles:
            role.context = self.context  # 将当前环境的上下文赋给角色
            role.set_env(self)  # 将环境设置到角色中

    def publish_message(self, message: Message, add_timestamp: bool = True):
        """发布信息到当前环境"""
        if add_timestamp:
            # 由于消息内容可能会重复，例如，在两晚之间杀死同一个人
            # 因此需要添加唯一的回合计数前缀，以防相同的消息在添加到记忆时被自动去重
            message.content = f"{self.round_cnt} | " + message.content
        # 调用父类的发布消息方法
        super().publish_message(message)

    async def run(self, k=1):
        """按顺序处理所有角色的运行"""
        for _ in range(k):
            # 遍历所有角色并执行每个角色的行为
            for role in self.roles.values():
                await role.run()
            # 每处理完一轮，回合计数加一
            self.round_cnt += 1
