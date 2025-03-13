#!/usr/bin/env python
# coding: utf-8
"""
@Time    : 2023/7/11 10:03
@Author  : chengmaoyu
@File    : action_output
"""

from pydantic import BaseModel


class ActionOutput:
    content: str  # 输出的内容，类型为字符串
    instruct_content: BaseModel  # 指令内容，类型为 Pydantic 的 BaseModel 类

    def __init__(self, content: str, instruct_content: BaseModel):
        """
        初始化 ActionOutput 类的实例
        :param content: 输出的文本内容
        :param instruct_content: 需要传递的指令内容，类型为 BaseModel 的实例
        """
        self.content = content  # 设置 content 属性
        self.instruct_content = instruct_content  # 设置 instruct_content 属性
