from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_serializer, model_validator


class BaseSerialization(BaseModel, extra="forbid"):
    """
    多态子类序列化/反序列化混入类
    - 首先，我们需要知道，pydantic 并没有为多态设计。
    - 如果 Engineer 是 Role 的子类，它将被序列化为 Role。如果我们想要将其序列化为 Engineer，则需要
      在 Engineer 类中添加 `class name` 字段。因此，Engineer 需要继承 SerializationMixin。

    更多详情：
    - https://docs.pydantic.dev/latest/concepts/serialization/
    - https://github.com/pydantic/pydantic/discussions/7008 讨论了如何避免 `__get_pydantic_core_schema__`
    """

    __is_polymorphic_base = False  # 标记是否为多态基类
    __subclasses_map__ = {}  # 存储子类映射

    @model_serializer(mode="wrap")
    def __serialize_with_class_type__(self, default_serializer) -> Any:
        # 默认序列化器，然后附加 `__module_class_name` 字段并返回
        ret = default_serializer(self)
        ret["__module_class_name"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        return ret

    @model_validator(mode="wrap")
    @classmethod
    def __convert_to_real_type__(cls, value: Any, handler):
        # 如果值不是字典类型，直接使用默认处理器
        if isinstance(value, dict) is False:
            return handler(value)

        # 如果是字典，确保移除 `__module_class_name` 字段
        # 因为我们不允许额外的关键字，但希望确保
        # 例如：Cat.model_validate(cat.model_dump()) 可以正常工作
        class_full_name = value.pop("__module_class_name", None)

        # 如果不是多态基类，使用默认处理器
        if not cls.__is_polymorphic_base:
            if class_full_name is None:
                return handler(value)
            elif str(cls) == f"<class '{class_full_name}'>":
                return handler(value)
            else:
                # f"尝试实例化 {class_full_name} 但是这不是多态基类")
                pass

        # 否则，我们查找正确的多态类型并进行实例化
        if class_full_name is None:
            raise ValueError("缺少 __module_class_name 字段")

        class_type = cls.__subclasses_map__.get(class_full_name, None)

        if class_type is None:
            # TODO 可以尝试动态导入
            raise TypeError(f"尝试实例化 {class_full_name}，但是它尚未定义!")

        return class_type(**value)

    def __init_subclass__(cls, is_polymorphic_base: bool = False, **kwargs):
        """在子类初始化时，设置是否为多态基类并将其添加到子类映射"""
        cls.__is_polymorphic_base = is_polymorphic_base
        cls.__subclasses_map__[f"{cls.__module__}.{cls.__qualname__}"] = cls
        super().__init_subclass__(**kwargs)
