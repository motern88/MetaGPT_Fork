#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : alexanderwu
@File    : write_review.py
"""
from typing import List

from metagpt.actions import Action
from metagpt.actions.action_node import ActionNode

# 定义 ActionNode 作为评论和反馈的动作
REVIEW = ActionNode(
    key="Review",  # 该节点的唯一标识符
    expected_type=List[str],  # 预期输出类型为字符串列表
    instruction="作为一位经验丰富的审阅者，审阅给定的输出。提出一系列关键问题，简明清晰地帮助写作者改进他们的作品。",
    example=[  # 给定的示例
        "这是一个很好的 PRD，但我认为可以通过增加更多细节来改进。",
    ],
)

LGTM = ActionNode(
    key="LGTM",  # 该节点的唯一标识符
    expected_type=str,  # 预期输出类型为字符串
    instruction="如果输出足够好，给出 LGTM（Looks Good To Me）反馈；如果输出不好，给出 LBTM（Looks Bad To Me）反馈。",
    example="LGTM",  # 示例输出
)

# 组合 REVIEW 和 LGTM 节点为一个新的 WRITE_REVIEW_NODE
WRITE_REVIEW_NODE = ActionNode.from_children("WRITE_REVIEW_NODE", [REVIEW, LGTM])

# 定义 WriteReview 类，用于生成审阅反馈
class WriteReview(Action):
    """为给定的上下文写一个审阅意见。"""

    name: str = "WriteReview"  # 类的名称

    # 异步运行方法，用于生成审阅
    async def run(self, context):
        # 调用 WRITE_REVIEW_NODE 来生成审阅，并返回其结果
        return await WRITE_REVIEW_NODE.fill(req=context, llm=self.llm, schema="json")