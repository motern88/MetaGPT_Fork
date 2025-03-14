"""RoleZero context builder."""

import copy
from typing import Any

from metagpt.const import EXPERIENCE_MASK
from metagpt.exp_pool.context_builders.base import BaseContextBuilder


# 角色零上下文构建器类，继承自 BaseContextBuilder
class RoleZeroContextBuilder(BaseContextBuilder):
    # 构建角色零上下文的异步方法
    async def build(self, req: Any) -> list[dict]:
        """构建角色零上下文字符串。

        注意：
            1. `req` 的预期格式，例如: [{...}, {"role": "user", "content": "context"}]。
            2. 如果 `req` 为空，返回原始的 `req`。
            3. 创建 `req` 的副本，并用实际经验替换副本中示例内容。
        """

        # 如果请求为空，直接返回原始请求
        if not req:
            return req

        # 格式化经验
        exps = self.format_exps()
        # 如果没有经验，返回原始请求
        if not exps:
            return req

        # 深拷贝 `req`
        req_copy = copy.deepcopy(req)

        # 替换副本中最后一个元素的 content 字段的示例内容
        req_copy[-1]["content"] = self.replace_example_content(req_copy[-1].get("content", ""), exps)

        # 返回更新后的请求副本
        return req_copy

    # 替换示例内容的方法
    def replace_example_content(self, text: str, new_example_content: str) -> str:
        return self.fill_experience(text, new_example_content)

    # 填充经验内容的方法
    @staticmethod
    def fill_experience(text: str, new_example_content: str) -> str:
        # 将文本中的 EXPERIENCE_MASK 替换为新的示例内容
        replaced_text = text.replace(EXPERIENCE_MASK, new_example_content)
        return replaced_text
