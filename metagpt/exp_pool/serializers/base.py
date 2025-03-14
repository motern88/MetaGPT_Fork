"""Base serializer."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


# BaseSerializer 类是一个基类，用于定义请求和响应的序列化和反序列化方法
class BaseSerializer(BaseModel, ABC):
    # 模型配置，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 序列化请求的方法，所有继承此类的子类必须实现该方法
    @abstractmethod
    def serialize_req(self, **kwargs) -> str:
        """序列化请求以便存储。

        请勿修改 kwargs。如果需要修改，请使用 copy.deepcopy 创建一个副本。
        注意：copy.deepcopy 可能会引发错误，例如 TypeError: cannot pickle '_thread.RLock' 对象。
        """
        pass

    # 序列化响应的方法，所有继承此类的子类必须实现该方法
    @abstractmethod
    def serialize_resp(self, resp: Any) -> str:
        """序列化函数的返回值以便存储。

        请勿修改 resp。其余部分与 `serialize_req` 方法相同。
        """
        pass

    # 反序列化响应的方法，所有继承此类的子类必须实现该方法
    @abstractmethod
    def deserialize_resp(self, resp: str) -> Any:
        """将存储的响应反序列化为函数的返回值"""
        pass
