#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/25 17:21
@Author  : alexanderwu
@File    : sales.py
"""

from typing import Optional

from pydantic import Field, model_validator

from metagpt.actions import SearchAndSummarize, UserRequirement
from metagpt.roles import Role
from metagpt.tools.search_engine import SearchEngine


class Sales(Role):
    name: str = "John Smith"  # 销售角色的名称
    profile: str = "Retail Sales Guide"  # 销售角色的职位描述
    desc: str = (
        "作为零售销售指南，我的名字是 John Smith。我专门解答客户咨询，"
        "并以专业的知识和精确的答案提供服务。我的回答仅基于我们知识库中的信息。"
        "如果您的问题超出了这个范围，我会诚实地告知无法提供答案，而不是猜测或假设。"
        "请注意，我的每个回复都将以一个经验丰富的销售指南的专业态度和礼貌进行。"
    )  # 销售角色的详细描述

    store: Optional[object] = Field(default=None, exclude=True)  # 必须实现 tools.SearchInterface（搜索引擎接口）

    @model_validator(mode="after")
    def validate_stroe(self):
        """验证 store 属性并根据需要设置操作"""
        if self.store:
            # 如果 store 存在，则创建搜索引擎并设置相关操作
            search_engine = SearchEngine.from_search_func(search_func=self.store.asearch, proxy=self.config.proxy)
            action = SearchAndSummarize(search_engine=search_engine, context=self.context)
        else:
            # 如果 store 不存在，则直接设置默认的搜索和总结操作
            action = SearchAndSummarize
        self.set_actions([action])  # 设置角色的操作
        self._watch([UserRequirement])  # 监视用户需求
        return self  # 返回当前角色对象
