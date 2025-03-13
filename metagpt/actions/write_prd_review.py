#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd_review.py
"""

from typing import Optional

from metagpt.actions.action import Action


class WritePRDReview(Action):
    # 定义类的属性
    name: str = ""  # 类的名称，可以用于标识这个动作
    i_context: Optional[str] = None  # 可选的上下文信息，默认为 None

    prd: Optional[str] = None  # 可选的 PRD（产品需求文档），默认为 None
    desc: str = "基于 PRD 进行 PRD 审核，提供清晰详细的反馈"  # 类的描述，表示该动作的功能
    prd_review_prompt_template: str = """
给定以下产品需求文档（PRD）：
{prd}

作为项目经理，请审查并提供反馈和建议。
"""  # 模板字符串，包含用于生成反馈的 PRD 内容

    # 异步运行方法，用于生成PRD审核反馈
    async def run(self, prd):
        self.prd = prd  # 设置类属性 prd 为传入的 PRD 文档内容
        prompt = self.prd_review_prompt_template.format(prd=self.prd)  # 根据模板格式化 PRD 内容
        review = await self._aask(prompt)  # 异步请求生成反馈
        return review  # 返回生成的反馈内容
