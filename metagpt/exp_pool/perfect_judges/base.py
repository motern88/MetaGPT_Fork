"""Base perfect judge."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from metagpt.exp_pool.schema import Experience


# 基础完美判断器类，继承自 BaseModel 和 ABC
class BasePerfectJudge(BaseModel, ABC):
    # 配置，允许使用任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 判断经验是否完美的抽象方法
    @abstractmethod
    async def is_perfect_exp(self, exp: Experience, serialized_req: str, *args, **kwargs) -> bool:
        """判断经验是否完美。

        参数:
            exp (Experience): 要评估的经验对象。
            serialized_req (str): 序列化后的请求，用于与经验中的请求进行比较。

        返回:
            bool: 如果经验符合完美的标准，返回 True；否则返回 False。
        """