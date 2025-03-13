#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : action.py
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from metagpt.actions.action_node import ActionNode
from metagpt.configs.models_config import ModelsConfig
from metagpt.context_mixin import ContextMixin
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.schema import (
    CodePlanAndChangeContext,
    CodeSummarizeContext,
    CodingContext,
    RunCodeContext,
    SerializationMixin,
    TestingContext,
)


class Action(SerializationMixin, ContextMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""  # 操作名称
    i_context: Union[  # 输入上下文，可以是多种类型
        dict, CodingContext, CodeSummarizeContext, TestingContext, RunCodeContext, CodePlanAndChangeContext, str, None
    ] = ""
    prefix: str = ""  # 在 aask* 时会加上前缀，作为 system_message
    desc: str = ""  # 用于技能管理器的描述
    node: ActionNode = Field(default=None, exclude=True)  # 动作节点，默认排除
    llm_name_or_type: Optional[str] = None  # LLM 模型名称或类型，配置为 None 时使用 config2.yaml 中的 llm 配置

    @model_validator(mode="after")
    @classmethod
    def _update_private_llm(cls, data: Any) -> Any:
        """在模型验证后更新私有 LLM 配置

        参数:
            data (Any): 当前模型数据

        返回:
            data (Any): 更新后的模型数据
        """
        config = ModelsConfig.default().get(data.llm_name_or_type)  # 获取 LLM 配置
        if config:
            llm = create_llm_instance(config)  # 创建 LLM 实例
            llm.cost_manager = data.llm.cost_manager  # 传递成本管理器
            data.llm = llm  # 更新数据中的 LLM 实例
        return data

    @property
    def prompt_schema(self):
        """返回提示模式"""
        return self.config.prompt_schema

    @property
    def project_name(self):
        """返回项目名称"""
        return self.config.project_name

    @project_name.setter
    def project_name(self, value):
        """设置项目名称"""
        self.config.project_name = value

    @property
    def project_path(self):
        """返回项目路径"""
        return self.config.project_path

    @model_validator(mode="before")
    @classmethod
    def set_name_if_empty(cls, values):
        """在空值时设置默认名称"""
        if "name" not in values or not values["name"]:
            values["name"] = cls.__name__  # 默认使用类名作为名称
        return values

    @model_validator(mode="before")
    @classmethod
    def _init_with_instruction(cls, values):
        """初始化时使用指令"""
        if "instruction" in values:  # 如果存在指令，创建节点
            name = values["name"]
            i = values.pop("instruction")
            values["node"] = ActionNode(key=name, expected_type=str, instruction=i, example="", schema="raw")
        return values

    def set_prefix(self, prefix):
        """设置前缀以供后续使用"""
        self.prefix = prefix
        self.llm.system_prompt = prefix  # 更新 LLM 系统提示
        if self.node:
            self.node.llm = self.llm  # 将 LLM 赋值给节点
        return self

    def __str__(self):
        """返回类名的字符串表示"""
        return self.__class__.__name__

    def __repr__(self):
        """返回类名的字符串表示"""
        return self.__str__()

    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        """附加默认前缀并执行 aask 操作"""
        return await self.llm.aask(prompt, system_msgs)

    async def _run_action_node(self, *args, **kwargs):
        """运行动作节点"""
        msgs = args[0]
        context = "## 历史消息\n"
        context += "\n".join([f"{idx}: {i}" for idx, i in enumerate(reversed(msgs))])  # 创建历史消息上下文
        return await self.node.fill(req=context, llm=self.llm)  # 填充节点并返回

    async def run(self, *args, **kwargs):
        """运行动作"""
        if self.node:
            return await self._run_action_node(*args, **kwargs)  # 如果有节点，运行节点
        raise NotImplementedError("run 方法应该在子类中实现。")  # 子类必须实现此方法

    def override_context(self):
        """将 `private_context` 和 `context` 设置为相同的 `Context` 对象"""
        if not self.private_context:
            self.private_context = self.context  # 如果没有私有上下文，则使用默认上下文
