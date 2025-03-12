#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/11 17:25
@Author  : alexanderwu
@File    : context_mixin.py
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from metagpt.config2 import Config
from metagpt.context import Context
from metagpt.provider.base_llm import BaseLLM


class ContextMixin(BaseModel):
    """用于上下文和配置的 Mixin 类"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Pydantic 在继承时对 _private_attr 处理存在 Bug，因此改用 private_* 作为私有属性
    # 相关问题：
    # - https://github.com/pydantic/pydantic/issues/7142
    # - https://github.com/pydantic/pydantic/issues/7083
    # - https://github.com/pydantic/pydantic/issues/7091

    # Env/Role/Action 使用此上下文作为私有上下文，否则使用 self.context 作为公共上下文
    private_context: Optional[Context] = Field(default=None, exclude=True)
    # Env/Role/Action 使用此配置作为私有配置，否则使用 self.context.config 作为公共配置
    private_config: Optional[Config] = Field(default=None, exclude=True)
    # Env/Role/Action 使用此 llm 作为私有 llm，否则使用 self.context._llm 实例
    private_llm: Optional[BaseLLM] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_context_mixin_extra(self):
        """在模型验证后处理额外字段"""
        self._process_context_mixin_extra()
        return self

    def _process_context_mixin_extra(self):
        """处理额外字段"""
        kwargs = self.model_extra or {}
        self.set_context(kwargs.pop("context", None))
        self.set_config(kwargs.pop("config", None))
        self.set_llm(kwargs.pop("llm", None))

    def set(self, k, v, override=False):
        """设置属性

        Args:
            k: 属性名
            v: 属性值
            override: 是否覆盖已有属性（默认为 False）
        """
        if override or not self.__dict__.get(k):
            self.__dict__[k] = v

    def set_context(self, context: Context, override=True):
        """设置上下文

        Args:
            context: Context 对象
            override: 是否覆盖已有上下文（默认为 True）
        """
        self.set("private_context", context, override)

    def set_config(self, config: Config, override=False):
        """设置配置

        Args:
            config: Config 对象
            override: 是否覆盖已有配置（默认为 False）
        """
        self.set("private_config", config, override)
        if config is not None:
            _ = self.llm  # 初始化 llm

    def set_llm(self, llm: BaseLLM, override=False):
        """设置 llm

        Args:
            llm: BaseLLM 对象
            override: 是否覆盖已有 llm（默认为 False）
        """
        self.set("private_llm", llm, override)

    @property
    def config(self) -> Config:
        """获取角色配置：优先使用角色的私有配置，否则使用全局上下文配置"""
        if self.private_config:
            return self.private_config
        return self.context.config

    @config.setter
    def config(self, config: Config) -> None:
        """设置角色配置"""
        self.set_config(config)

    @property
    def context(self) -> Context:
        """获取角色上下文：优先使用角色的私有上下文，否则创建新的上下文"""
        if self.private_context:
            return self.private_context
        return Context()

    @context.setter
    def context(self, context: Context) -> None:
        """设置角色上下文"""
        self.set_context(context)

    @property
    def llm(self) -> BaseLLM:
        """获取角色 LLM：如果不存在，则从角色配置初始化"""
        if not self.private_llm:
            self.private_llm = self.context.llm_with_cost_manager_from_llm_config(self.config.llm)
        return self.private_llm

    @llm.setter
    def llm(self, llm: BaseLLM) -> None:
        """设置角色 LLM"""
        self.private_llm = llm
