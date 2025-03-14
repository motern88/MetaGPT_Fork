"""Simple scorer."""

import json

from pydantic import Field

from metagpt.exp_pool.schema import Score
from metagpt.exp_pool.scorers.base import BaseScorer
from metagpt.llm import LLM
from metagpt.provider.base_llm import BaseLLM
from metagpt.utils.common import CodeParser

SIMPLE_SCORER_TEMPLATE = """
Role: You are a highly efficient assistant, tasked with evaluating a response to a given request. The response is generated by a large language model (LLM). 

I will provide you with a request and a corresponding response. Your task is to assess this response and provide a score from a human perspective.

## Context
### Request
{req}

### Response
{resp}

## Format Example
```json
{{
    "val": "the value of the score, int from 1 to 10, higher is better.",
    "reason": "an explanation supporting the score."
}}
```

## Instructions
- Understand the request and response given by the user.
- Evaluate the response based on its quality relative to the given request.
- Provide a score from 1 to 10, where 10 is the best.
- Provide a reason supporting your score.

## Constraint
Format: Just print the result in json format like **Format Example**.

## Action
Follow instructions, generate output and make sure it follows the **Constraint**.
"""


# 简单评分器类，继承自 BaseScorer
class SimpleScorer(BaseScorer):
    # 默认使用 BaseLLM 作为语言模型
    llm: BaseLLM = Field(default_factory=LLM)

    # 评估响应质量的方法
    async def evaluate(self, req: str, resp: str) -> Score:
        """根据给定的请求和响应，通过 LLM 评估响应的质量，并返回评分。

        参数:
            req (str): 请求字符串。
            resp (str): 响应字符串。

        返回:
            Score: 包含评分（1-10）和评分理由的对象。
        """

        # 格式化评分模板，生成评估提示
        prompt = SIMPLE_SCORER_TEMPLATE.format(req=req, resp=resp)

        # 使用 LLM 请求评分结果
        resp = await self.llm.aask(prompt)

        # 解析返回的 JSON 格式的评分数据
        resp_json = json.loads(CodeParser.parse_code(resp, lang="json"))

        # 返回评分对象
        return Score(**resp_json)
