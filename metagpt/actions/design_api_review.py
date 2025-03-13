#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:31
@Author  : alexanderwu
@File    : design_api_review.py
"""

from typing import Optional

from metagpt.actions.action import Action


# 定义 DesignReview 类，继承自 Action 类，用于处理 API 设计的评审
class DesignReview(Action):
    name: str = "DesignReview"  # 该类的名称，表示设计评审
    i_context: Optional[str] = None  # 可选的上下文信息，可以为空

    # 定义异步方法 run，接收产品需求文档 (prd) 和 API 设计 (api_design) 作为输入
    async def run(self, prd, api_design):
        # 构造评审的提示信息，包含产品需求文档和 API 设计
        prompt = (
            f"Here is the Product Requirement Document (PRD):\n\n{prd}\n\nHere is the list of APIs designed "
            f"based on this PRD:\n\n{api_design}\n\nPlease review whether this API design meets the requirements"
            f" of the PRD, and whether it complies with good design practices."
        )

        # 调用 _aask 方法（假定为调用外部服务或模型进行评审），并传入构造的提示信息
        api_review = await self._aask(prompt)

        # 返回评审结果
        return api_review
