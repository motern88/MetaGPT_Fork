#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 16:15
@Author  : alexanderwu
@File    : singleton.py
"""
import abc


class Singleton(abc.ABCMeta, type):
    """
    单例元类，用于确保类只有一个实例。
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """单例元类的 `__call__` 方法，确保每个类只有一个实例。"""
        if cls not in cls._instances:
            # 如果当前类没有实例化过，就创建一个新的实例
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # 返回类的唯一实例
        return cls._instances[cls]
