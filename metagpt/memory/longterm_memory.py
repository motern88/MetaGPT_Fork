#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : the implement of Long-term memory
"""

from typing import Optional

from pydantic import ConfigDict, Field

from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.memory.memory_storage import MemoryStorage
from metagpt.roles.role import RoleContext
from metagpt.schema import Message


class LongTermMemory(Memory):
    """
    角色的长期记忆
    - 启动时恢复记忆
    - 记忆变化时更新
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_storage: MemoryStorage = Field(default_factory=MemoryStorage)  # 存储长期记忆的对象
    rc: Optional[RoleContext] = None  # 角色上下文
    msg_from_recover: bool = False  # 标记是否为恢复记忆时的消息

    def recover_memory(self, role_id: str, rc: RoleContext):
        """
        恢复角色的记忆
        - 恢复指定角色的记忆
        - 如果记忆存储为空，发出警告
        """
        self.memory_storage.recover_memory(role_id)  # 从存储中恢复记忆
        self.rc = rc
        if not self.memory_storage.is_initialized:
            # 如果记忆存储未初始化，说明是第一次运行该角色，记忆为空
            logger.warning(f"Role {role_id} 第一次运行，长期记忆为空")
        else:
            # 恢复了已存在的记忆
            logger.warning(f"Role {role_id} 已有记忆存储，并已恢复。")
        self.msg_from_recover = True  # 设置为从恢复中获取的消息
        # self.add_batch(messages) # TODO: 暂时不需要
        self.msg_from_recover = False  # 恢复后重新设置标记

    def add(self, message: Message):
        """
        向长期记忆中添加一条消息
        - 仅当消息是由角色的关注动作触发时才添加
        - 恢复记忆时不重复添加
        """
        super().add(message)
        for action in self.rc.watch:
            if message.cause_by == action and not self.msg_from_recover:
                # 当前只将角色关注的消息添加到其记忆存储中
                # 且忽略从恢复中重复添加的消息
                self.memory_storage.add(message)

    async def find_news(self, observed: list[Message], k=0) -> list[Message]:
        """
        从最近的 k 条记忆中查找新消息（未见过的消息），当 k=0 时，查找所有记忆中的新消息
        1. 查找短期记忆（stm）中的新消息
        2. 基于长期记忆（ltm）进一步筛选，过滤掉相似的消息，最终得到新消息
        """
        stm_news = super().find_news(observed, k=k)  # 获取短期记忆中的新消息
        if not self.memory_storage.is_initialized:
            # 如果记忆存储未初始化，直接返回短期记忆中的新消息
            return stm_news

        ltm_news: list[Message] = []  # 存储长期记忆中的新消息
        for mem in stm_news:
            # 筛选出与长期记忆中未见过的消息，保留新消息
            mem_searched = await self.memory_storage.search_similar(mem)
            if len(mem_searched) == 0:
                ltm_news.append(mem)
        return ltm_news[-k:]  # 返回最近的 k 条新消息

    def persist(self):
        """
        持久化记忆，将当前的记忆存储到持久化介质中
        """
        self.memory_storage.persist()

    def delete(self, message: Message):
        """
        删除一条消息
        - 从短期记忆中删除消息
        - TODO: 删除长期记忆中的对应消息
        """
        super().delete(message)
        # TODO: 从记忆存储中删除该消息

    def clear(self):
        """
        清空短期和长期记忆
        """
        super().clear()
        self.memory_storage.clean()  # 清空长期记忆存储
