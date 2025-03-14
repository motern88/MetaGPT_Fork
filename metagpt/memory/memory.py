#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 12:15
@Author  : alexanderwu
@File    : memory.py
@Modified By: mashenquan, 2023-11-1. According to RFC 116: Updated the type of index key.
"""
from collections import defaultdict
from typing import DefaultDict, Iterable, Optional, Set

from pydantic import BaseModel, Field, SerializeAsAny

from metagpt.const import IGNORED_MESSAGE_ID
from metagpt.schema import Message
from metagpt.utils.common import any_to_str, any_to_str_set
from metagpt.utils.exceptions import handle_exception


class Memory(BaseModel):
    """最基本的记忆：超级记忆"""

    storage: list[SerializeAsAny[Message]] = []  # 存储所有消息的列表
    index: DefaultDict[str, list[SerializeAsAny[Message]]] = Field(default_factory=lambda: defaultdict(list))  # 按动作分类的索引
    ignore_id: bool = False  # 是否忽略消息的 ID

    def add(self, message: Message):
        """向存储中添加新消息，同时更新索引"""
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID  # 如果忽略 ID，设置为特殊标识
        if message in self.storage:
            return  # 如果消息已经存在，不再添加
        self.storage.append(message)
        if message.cause_by:
            self.index[message.cause_by].append(message)  # 根据消息的触发动作更新索引

    def add_batch(self, messages: Iterable[Message]):
        """批量添加消息"""
        for message in messages:
            self.add(message)

    def get_by_role(self, role: str) -> list[Message]:
        """返回指定角色的所有消息"""
        return [message for message in self.storage if message.role == role]

    def get_by_content(self, content: str) -> list[Message]:
        """返回所有包含指定内容的消息"""
        return [message for message in self.storage if content in message.content]

    def delete_newest(self) -> "Message":
        """删除存储中最新的消息"""
        if len(self.storage) > 0:
            newest_msg = self.storage.pop()  # 弹出最新的消息
            if newest_msg.cause_by and newest_msg in self.index[newest_msg.cause_by]:
                self.index[newest_msg.cause_by].remove(newest_msg)  # 从索引中移除
        else:
            newest_msg = None
        return newest_msg

    def delete(self, message: Message):
        """删除指定的消息，同时更新索引"""
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID  # 如果忽略 ID，设置为特殊标识
        self.storage.remove(message)  # 从存储中移除消息
        if message.cause_by and message in self.index[message.cause_by]:
            self.index[message.cause_by].remove(message)  # 从索引中移除

    def clear(self):
        """清空存储和索引"""
        self.storage = []
        self.index = defaultdict(list)

    def count(self) -> int:
        """返回存储中消息的数量"""
        return len(self.storage)

    def try_remember(self, keyword: str) -> list[Message]:
        """尝试回忆所有包含指定关键字的消息"""
        return [message for message in self.storage if keyword in message.content]

    def get(self, k=0) -> list[Message]:
        """返回最近的 k 条记忆，当 k=0 时返回所有消息"""
        return self.storage[-k:]

    def find_news(self, observed: list[Message], k=0) -> list[Message]:
        """从最近的 k 条记忆中查找新消息，当 k=0 时从所有记忆中查找新消息"""
        already_observed = self.get(k)  # 获取已知的消息
        news: list[Message] = []  # 存储新消息
        for i in observed:
            if i in already_observed:
                continue  # 如果消息已经被观察过，则跳过
            news.append(i)  # 添加新消息
        return news

    def get_by_action(self, action) -> list[Message]:
        """返回所有由指定动作触发的消息"""
        index = any_to_str(action)  # 将动作转换为字符串
        return self.index[index]  # 从索引中获取相应的消息

    def get_by_actions(self, actions: Set) -> list[Message]:
        """返回所有由指定动作集合触发的消息"""
        rsp = []  # 存储响应的消息
        indices = any_to_str_set(actions)  # 将动作集合转换为字符串集合
        for action in indices:
            if action not in self.index:
                continue  # 如果动作没有相关的消息，跳过
            rsp += self.index[action]  # 将相关消息添加到结果中
        return rsp

    @handle_exception
    def get_by_position(self, position: int) -> Optional[Message]:
        """返回指定位置的消息，如果位置无效，则返回 None"""
        return self.storage[position]  # 返回指定位置的消息
