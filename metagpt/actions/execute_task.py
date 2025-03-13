#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:26
@Author  : femto Zheng
@File    : execute_task.py
"""


from metagpt.actions import Action
from metagpt.schema import Message


# 定义 ExecuteTask 类，继承自 Action 类，用于表示执行任务的操作
class ExecuteTask(Action):
    name: str = "ExecuteTask"  # 该类的名称，表示执行任务
    i_context: list[Message] = []  # 存储消息的上下文列表，默认为空

    # 定义异步方法 run，用于执行任务，接受任意参数
    async def run(self, *args, **kwargs):
        pass  # 当前方法为空，具体实现留给子类或后续开发
