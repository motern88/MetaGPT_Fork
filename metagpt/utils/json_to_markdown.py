#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/11 11:50
@Author  : femto Zheng
@File    : json_to_markdown.py
"""


def json_to_markdown(data, depth=2):
    """
    将 JSON 对象转换为 Markdown 格式，支持为键添加标题并为数组添加列表，能够处理嵌套对象。

    参数:
        data: JSON 对象（字典）或其他值。
        depth (int): 当前的 Markdown 标题深度级别。

    返回:
        str: JSON 数据的 Markdown 表示。
    """
    markdown = ""  # 初始化转换后的 Markdown 字符串

    if isinstance(data, dict):  # 如果 data 是字典类型
        for key, value in data.items():  # 遍历字典的键值对
            if isinstance(value, list):  # 如果值是数组类型
                markdown += "#" * depth + f" {key}\n\n"  # 为键添加标题
                items = [str(item) for item in value]  # 将数组元素转换为字符串
                markdown += "- " + "\n- ".join(items) + "\n\n"  # 将数组元素按列表形式添加
            elif isinstance(value, dict):  # 如果值是嵌套的字典
                markdown += "#" * depth + f" {key}\n\n"  # 为键添加标题
                markdown += json_to_markdown(value, depth + 1)  # 递归处理嵌套字典
            else:  # 其他类型的值
                markdown += "#" * depth + f" {key}\n\n{value}\n\n"  # 直接添加键和值
    else:  # 如果 data 不是字典类型，直接转换为字符串
        markdown = str(data)

    return markdown  # 返回转换后的 Markdown 字符串
