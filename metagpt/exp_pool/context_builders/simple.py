"""Simple context builder."""


from typing import Any

from metagpt.exp_pool.context_builders.base import BaseContextBuilder

SIMPLE_CONTEXT_TEMPLATE = """
## Context

### Experiences
-----
{exps}
-----

## User Requirement
{req}

## Instruction
Consider **Experiences** to generate a better answer.
"""


# 简单上下文构建器类，继承自 BaseContextBuilder
class SimpleContextBuilder(BaseContextBuilder):
    # 构建简单上下文的异步方法
    async def build(self, req: Any) -> str:
        # 返回格式化后的上下文字符串
        return SIMPLE_CONTEXT_TEMPLATE.format(req=req, exps=self.format_exps())
