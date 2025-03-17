"""Base Factory."""

from typing import Any, Callable


class GenericFactory:
    """通用工厂类，用于根据任意键获取对象。"""

    def __init__(self, creators: dict[Any, Callable] = None):
        """初始化工厂。

        creators 是一个字典，键是标识符，值是对应的创建函数，用于创建对象。
        """
        self._creators = creators or {}  # 创建函数字典

    def get_instances(self, keys: list[Any], **kwargs) -> list[Any]:
        """根据键列表获取对象实例列表。"""
        return [self.get_instance(key, **kwargs) for key in keys]  # 返回对象实例列表

    def get_instance(self, key: Any, **kwargs) -> Any:
        """根据键获取对象实例。

        如果键未找到，则抛出异常。
        """
        creator = self._creators.get(key)  # 获取创建函数
        if creator:
            return creator(**kwargs)  # 调用创建函数并返回对象

        self._raise_for_key(key)  # 如果键未找到，抛出异常

    def _raise_for_key(self, key: Any):
        """抛出键未找到的异常。"""
        raise ValueError(f"未注册的键: {key}")  # 抛出异常


class ConfigBasedFactory(GenericFactory):
    """基于配置的工厂类，用于根据对象类型获取对象。"""

    def get_instance(self, key: Any, **kwargs) -> Any:
        """根据键的类型获取对象实例。

        键是配置对象（例如 pydantic 模型），根据键的类型调用创建函数，并将键传递给函数。
        如果键未找到，则抛出异常。
        """
        creator = self._creators.get(type(key))  # 根据键的类型获取创建函数
        if creator:
            return creator(key, **kwargs)  # 调用创建函数并返回对象

        self._raise_for_key(key)  # 如果键未找到，抛出异常

    def _raise_for_key(self, key: Any):
        """抛出键未找到的异常。"""
        raise ValueError(f"未知的配置: `{type(key)}`, {key}")  # 抛出异常

    @staticmethod
    def _val_from_config_or_kwargs(key: str, config: object = None, **kwargs) -> Any:
        """从配置对象或 kwargs 中获取值。

        优先使用配置对象的值，除非它为 None，此时从 kwargs 中查找。
        如果未找到，则返回 None。
        """
        if config is not None and hasattr(config, key):  # 如果配置对象存在且包含该键
            val = getattr(config, key)  # 获取值
            if val is not None:  # 如果值不为 None
                return val  # 返回值

        if key in kwargs:  # 如果键在 kwargs 中
            return kwargs[key]  # 返回 kwargs 中的值

        return None  # 如果未找到，返回 None
