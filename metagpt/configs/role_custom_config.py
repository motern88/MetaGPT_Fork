#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/22 16:33
@Author  : Justin
@File    : role_custom_config.py
"""
from metagpt.configs.llm_config import LLMConfig
from metagpt.utils.yaml_model import YamlModel


class RoleCustomConfig(YamlModel):
    """角色自定义配置类
    该类用于为角色定义自定义配置，包含角色的标识信息和与之关联的 LLM 配置。

    属性：
        role (str): 角色的类名或角色 ID，用于唯一标识角色。
        llm (LLMConfig): 关联的 LLM 配置，定义了与该角色相关联的 LLM 模型的配置。
    """

    role: str = ""  # 角色的类名或角色 ID
    llm: LLMConfig  # 关联的 LLM 配置
