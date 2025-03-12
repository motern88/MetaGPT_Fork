#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 16:32
@Author  : alexanderwu
@File    : context.py
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from metagpt.config2 import Config
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import (
    CostManager,
    FireworksCostManager,
    TokenCostManager,
)


class AttrDict(BaseModel):
    """一个类似字典的对象，支持通过属性访问键，兼容 Pydantic。"""

    model_config = ConfigDict(extra="allow")  # 允许额外字段

    def __init__(self, **kwargs):
        """初始化时，将传入的键值对存入对象的 __dict__ 中。"""
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        """支持通过属性访问键，若键不存在，则返回 None。"""
        return self.__dict__.get(key, None)

    def __setattr__(self, key, value):
        """支持通过属性设置键值对。"""
        self.__dict__[key] = value

    def __delattr__(self, key):
        """支持通过属性删除键值对，若键不存在，则抛出异常。"""
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"无此属性: {key}")

    def set(self, key, val: Any):
        """设置键值对。"""
        self.__dict__[key] = val

    def get(self, key, default: Any = None):
        """获取键对应的值，若键不存在，则返回默认值。"""
        return self.__dict__.get(key, default)

    def remove(self, key):
        """删除键值对。"""
        if key in self.__dict__:
            self.__delattr__(key)


class Context(BaseModel):
    """MetaGPT 运行环境上下文"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 允许任意类型

    kwargs: AttrDict = AttrDict()  # 存储额外的环境参数
    config: Config = Field(default_factory=Config.default)  # 配置信息

    cost_manager: CostManager = CostManager()  # 费用管理器

    _llm: Optional[BaseLLM] = None  # 缓存的 LLM 实例

    def new_environ(self):
        """返回一个新的操作系统环境变量字典。"""
        env = os.environ.copy()
        # i = self.options
        # env.update({k: v for k, v in i.items() if isinstance(v, str)})
        return env

    def _select_costmanager(self, llm_config: LLMConfig) -> CostManager:
        """根据 LLM 配置选择适当的费用管理器。"""
        if llm_config.api_type == LLMType.FIREWORKS:
            return FireworksCostManager()
        elif llm_config.api_type == LLMType.OPEN_LLM:
            return TokenCostManager()
        else:
            return self.cost_manager

    def llm(self) -> BaseLLM:
        """返回一个 LLM 实例（支持缓存）。"""
        # if self._llm is None:
        self._llm = create_llm_instance(self.config.llm)
        if self._llm.cost_manager is None:
            self._llm.cost_manager = self._select_costmanager(self.config.llm)
        return self._llm

    def llm_with_cost_manager_from_llm_config(self, llm_config: LLMConfig) -> BaseLLM:
        """根据 LLM 配置返回一个 LLM 实例（支持缓存）。"""
        # if self._llm is None:
        llm = create_llm_instance(llm_config)
        if llm.cost_manager is None:
            llm.cost_manager = self._select_costmanager(llm_config)
        return llm

    def serialize(self) -> Dict[str, Any]:
        """将对象的属性序列化为字典。

        Returns:
            Dict[str, Any]: 包含序列化数据的字典。
        """
        return {
            "kwargs": {k: v for k, v in self.kwargs.__dict__.items()},  # 序列化 kwargs
            "cost_manager": self.cost_manager.model_dump_json(),  # 将 cost_manager 转换为 JSON 字符串
        }

    def deserialize(self, serialized_data: Dict[str, Any]):
        """反序列化给定的字典数据，并更新对象的属性。

        Args:
            serialized_data (Dict[str, Any]): 包含序列化数据的字典。
        """
        if not serialized_data:
            return
        kwargs = serialized_data.get("kwargs")
        if kwargs:
            for k, v in kwargs.items():
                self.kwargs.set(k, v)  # 逐个恢复 kwargs 数据
        cost_manager = serialized_data.get("cost_manager")
        if cost_manager:
            self.cost_manager.model_validate_json(cost_manager)  # 使用 Pydantic 验证并解析 JSON 数据
