"""Base context builder."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from metagpt.exp_pool.schema import Experience

EXP_TEMPLATE = """Given the request: {req}, We can get the response: {resp}, which scored: {score}."""


# 经验模板，用于格式化经验信息
EXP_TEMPLATE = """给定请求: {req}, 我们可以得到回应: {resp}, 得分为: {score}。"""

# 基础上下文构建器类
class BaseContextBuilder(BaseModel, ABC):
    # 配置字典，允许自定义类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 经验列表
    exps: list[Experience] = []

    # 构建上下文的抽象方法
    @abstractmethod
    async def build(self, req: Any) -> Any:
        """根据请求构建上下文。

        请不要修改 `req`。如果需要修改，请使用 `copy.deepcopy` 创建其副本。
        """

    # 格式化经验列表为字符串
    def format_exps(self) -> str:
        """将经验格式化为带编号的字符串列表。

        示例:
            1. 给定请求: req1, 我们可以得到回应: resp1, 得分为: 8。
            2. 给定请求: req2, 我们可以得到回应: resp2, 得分为: 9。

        返回:
            str: 格式化后的经验字符串。
        """

        result = []
        for i, exp in enumerate(self.exps, start=1):
            # 获取经验的得分值，如果没有得分则显示 "N/A"
            score_val = exp.metric.score.val if exp.metric and exp.metric.score else "N/A"
            # 根据经验模板格式化每个经验
            result.append(f"{i}. " + EXP_TEMPLATE.format(req=exp.req, resp=exp.resp, score=score_val))

        # 将所有经验以换行符连接并返回
        return "\n".join(result)
