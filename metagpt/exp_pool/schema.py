"""Experience schema."""
import time
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

MAX_SCORE = 10  # 最大分数

DEFAULT_SIMILARITY_TOP_K = 2  # 默认相似度最优结果数量

LOG_NEW_EXPERIENCE_PREFIX = "New experience: "  # 新体验的日志前缀


class QueryType(str, Enum):
    """查询体验的类型。"""

    EXACT = "exact"  # 精确匹配
    SEMANTIC = "semantic"  # 语义匹配


class ExperienceType(str, Enum):
    """体验的类型。"""

    SUCCESS = "success"  # 成功
    FAILURE = "failure"  # 失败
    INSIGHT = "insight"  # 洞察


class EntryType(Enum):
    """体验条目的类型。"""

    AUTOMATIC = "Automatic"  # 自动
    MANUAL = "Manual"  # 手动


class Score(BaseModel):
    """评分（度量标准）。"""

    val: int = Field(default=1, description="评分的值，介于1到10之间，值越大表示更好。")
    reason: str = Field(default="", description="评分的理由。")


class Metric(BaseModel):
    """体验的度量标准。"""

    time_cost: float = Field(default=0.000, description="时间成本，单位是毫秒。")
    money_cost: float = Field(default=0.000, description="金钱成本，单位是美元。")
    score: Score = Field(default=None, description="评分，包括值和理由。")


class Trajectory(BaseModel):
    """体验的轨迹。"""

    plan: str = Field(default="", description="计划。")
    action: str = Field(default="", description="计划的行动。")
    observation: str = Field(default="", description="行动的结果。")
    reward: int = Field(default=0, description="对行动的衡量。")


class Experience(BaseModel):
    """体验。"""

    req: str = Field(..., description="请求的内容。")
    resp: str = Field(..., description="响应的内容，类型可以是字符串/JSON/代码。")
    metric: Optional[Metric] = Field(default=None, description="度量标准。")
    exp_type: ExperienceType = Field(default=ExperienceType.SUCCESS, description="体验的类型。")
    entry_type: EntryType = Field(default=EntryType.AUTOMATIC, description="条目的类型：手动或自动。")
    tag: str = Field(default="", description="标记体验。")
    traj: Optional[Trajectory] = Field(default=None, description="体验轨迹。")
    timestamp: Optional[float] = Field(default_factory=time.time)  # 时间戳，默认当前时间
    uuid: Optional[UUID] = Field(default_factory=uuid4)  # 唯一标识符，默认生成新的UUID

    def rag_key(self):
        """返回该体验的查询键。"""
        return self.req
