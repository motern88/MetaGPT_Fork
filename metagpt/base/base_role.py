from abc import abstractmethod
from typing import Optional, Union

from metagpt.base.base_serialization import BaseSerialization


class BaseRole(BaseSerialization):
    """所有角色的抽象基类。"""

    name: str

    @property
    def is_idle(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def think(self):
        """思考接下来要做什么，并决定下一步的行动。"""
        raise NotImplementedError

    @abstractmethod
    def act(self):
        """执行当前的动作。"""
        raise NotImplementedError

    @abstractmethod
    async def react(self) -> "Message":
        """通过三种策略之一来回应观察到的消息。"""

    @abstractmethod
    async def run(self, with_message: Optional[Union[str, "Message", list[str]]] = None) -> Optional["Message"]:
        """观察并根据观察结果进行思考和行动。"""

    @abstractmethod
    def get_memories(self, k: int = 0) -> list["Message"]:
        """返回该角色最近的 k 条记忆。"""
