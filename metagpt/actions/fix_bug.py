# -*- coding: utf-8 -*-
"""
@Time    : 2023-12-12
@Author  : mashenquan
@File    : fix_bug.py
"""
from metagpt.actions import Action


class FixBug(Action):
    """修复 bug 的动作类，没有任何实现细节"""

    name: str = "FixBug"  # 动作名称为 "FixBug"
