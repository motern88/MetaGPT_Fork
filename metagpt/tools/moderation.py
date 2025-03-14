#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/26 14:27
@Author  : zhanglei
@File    : moderation.py
"""
from typing import Union

from metagpt.provider.base_llm import BaseLLM


class Moderation:
    def __init__(self, llm: BaseLLM):
        """
        初始化 Moderation 类

        :param llm: 传入的 LLM（大语言模型）实例，用于执行内容审核
        """
        self.llm = llm

    def handle_moderation_results(self, results):
        """
        处理审核结果

        :param results: 审核结果列表，每个结果包含 flagged 信息和各类别的审核标记
        :return: 格式化后的结果列表，包含每个条目的 flagged 信息和被标记为 flagged 的类别
        """
        resp = []
        for item in results:
            categories = item.categories.dict()  # 获取每个审核项的类别
            true_categories = [category for category, item_flagged in categories.items() if item_flagged]  # 获取被标记为 flagged 的类别
            resp.append({"flagged": item.flagged, "true_categories": true_categories})  # 返回格式化后的结果
        return resp

    async def amoderation_with_categories(self, content: Union[str, list[str]]):
        """
        异步内容审核，并返回被标记为 flagged 的类别

        :param content: 需要审核的内容，可以是字符串或字符串列表
        :return: 返回审核结果列表，其中每项包含 flagged 状态及其对应的类别
        """
        resp = []
        if content:  # 如果有内容需要审核
            # 调用大语言模型的 amoderation 方法进行内容审核
            moderation_results = await self.llm.amoderation(content=content)
            resp = self.handle_moderation_results(moderation_results.results)  # 处理审核结果并返回
        return resp

    async def amoderation(self, content: Union[str, list[str]]):
        """
        异步内容审核，返回内容是否被标记为 flagged

        :param content: 需要审核的内容，可以是字符串或字符串列表
        :return: 返回审核结果列表，每项包含是否被标记为 flagged
        """
        resp = []
        if content:  # 如果有内容需要审核
            # 调用大语言模型的 amoderation 方法进行内容审核
            moderation_results = await self.llm.amoderation(content=content)
            results = moderation_results.results
            for item in results:
                resp.append(item.flagged)  # 只返回每项内容是否被标记为 flagged
        return resp
