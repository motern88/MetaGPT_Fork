"""Base scorer."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from metagpt.exp_pool.schema import Score


# 基础评分器类，继承自 BaseModel 和 ABC
class BaseScorer(BaseModel, ABC):
    # 配置，允许使用任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 评估响应质量的抽象方法
    @abstractmethod
    async def evaluate(self, req: str, resp: str) -> Score:
        """根据给定的请求评估响应的质量。

        参数:
            req (str): 请求字符串。
            resp (str): 响应字符串。

        返回:
            Score: 返回一个 Score 对象，表示响应质量的评分。
        """
