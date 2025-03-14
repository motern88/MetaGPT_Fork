"""Simple perfect judge."""


from pydantic import ConfigDict

from metagpt.exp_pool.perfect_judges.base import BasePerfectJudge
from metagpt.exp_pool.schema import MAX_SCORE, Experience


# 简单完美判断器类，继承自 BasePerfectJudge
class SimplePerfectJudge(BasePerfectJudge):
    # 配置，允许使用任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 判断经验是否完美的具体实现
    async def is_perfect_exp(self, exp: Experience, serialized_req: str, *args, **kwargs) -> bool:
        """判断经验是否完美。

        参数:
            exp (Experience): 要评估的经验对象。
            serialized_req (str): 序列化后的请求，用于与经验中的请求进行比较。

        返回:
            bool: 如果序列化请求与经验请求匹配，并且经验的分数为完美分数，返回 True；否则返回 False。
        """

        # 检查经验是否具有有效的分数
        if not exp.metric or not exp.metric.score:
            return False

        # 判断序列化请求是否与经验请求相同，并且分数是否为最大值
        return serialized_req == exp.req and exp.metric.score.val == MAX_SCORE
