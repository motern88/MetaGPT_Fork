#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 17:46
@Author  : alexanderwu
@File    : add_requirement.py
"""
from metagpt.actions import Action


class UserRequirement(Action):
    """
    用户需求类，表示一个没有具体实现细节的需求
    继承自 Action 类，用于定义用户需求的基础结构
    """
