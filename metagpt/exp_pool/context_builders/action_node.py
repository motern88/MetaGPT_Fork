"""Action Node context builder."""

from typing import Any

from metagpt.exp_pool.context_builders.base import BaseContextBuilder

# 动作节点上下文模板
ACTION_NODE_CONTEXT_TEMPLATE = """
{req}

### Experiences
-----
{exps}
-----

## Instruction
考虑**Experiences**以生成更好的回答。
"""

# 动作节点上下文构建器
class ActionNodeContextBuilder(BaseContextBuilder):
    # 构建动作节点上下文字符串
    async def build(self, req: Any) -> str:
        """构建动作节点上下文字符串。

        如果没有经验，返回原始的 `req`；
        否则返回包含 `req` 和格式化经验的上下文。
        """

        # 格式化经验
        exps = self.format_exps()

        # 如果有经验，返回格式化后的上下文；否则仅返回原始请求
        return ACTION_NODE_CONTEXT_TEMPLATE.format(req=req, exps=exps) if exps else req
