#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : evaluate_action.py
@Desc    : The implementation of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""
from typing import Optional

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.utils.common import CodeParser, general_after_log, to_markdown_code_block


class EvaluationData(BaseModel):
    """表示评估数据的模型。

    属性:
        is_pass (bool): 表示评估是否通过。
        conclusion (Optional[str]): 评估的结论或备注。
    """

    is_pass: bool
    conclusion: Optional[str] = None


class EvaluateAction(Action):
    """评估操作的基类。

    该类提供了使用指定语言模型评估提示的方法。
    """

    @retry(
        wait=wait_random_exponential(min=1, max=20),  # 等待时间的指数回退
        stop=stop_after_attempt(6),  # 最多重试6次
        after=general_after_log(logger),  # 每次重试后进行日志记录
    )
    async def _evaluate(self, prompt: str) -> (bool, str):
        """评估给定的提示。

        参数:
            prompt (str): 要评估的提示。

        返回:
            tuple: 包含两个元素的元组：
                - bool: 表示评估是否通过。
                - str: 包含评估数据的 JSON 字符串。
        """
        rsp = await self.llm.aask(prompt)  # 使用语言模型进行评估
        json_data = CodeParser.parse_code(text=rsp, lang="json")  # 解析返回的 JSON 数据
        data = EvaluationData.model_validate_json(json_data)  # 验证并创建 EvaluationData 实例
        return data.is_pass, to_markdown_code_block(val=json_data, type_="json")  # 返回评估结果

    async def _vote(self, prompt: str) -> EvaluationData:
        """多次评估提示并返回评估结果的一致性。

        参数:
            prompt (str): 要评估的提示。

        返回:
            EvaluationData: 包含评估结果和评估摘要的对象。
        """
        evaluations = {}  # 存储每次评估的结果
        for i in range(3):  # 执行3次评估
            vote, evaluation = await self._evaluate(prompt)  # 获取每次评估的结果
            val = evaluations.get(vote, [])  # 获取当前评估结果的列表
            val.append(evaluation)  # 添加评估内容
            if len(val) > 1:  # 如果有超过一个结果，则返回最终结果
                return EvaluationData(is_pass=vote, conclusion="\n".join(val))  # 返回包含多个评估结果的结论
            evaluations[vote] = val  # 将当前评估结果存储到字典中
