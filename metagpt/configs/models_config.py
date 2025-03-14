#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models_config.py

This module defines the ModelsConfig class for handling configuration of LLM models.

Attributes:
    CONFIG_ROOT (Path): Root path for configuration files.
    METAGPT_ROOT (Path): Root path for MetaGPT files.

Classes:
    ModelsConfig (YamlModel): Configuration class for LLM models.
"""
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from metagpt.config2 import merge_dict
from metagpt.configs.llm_config import LLMConfig
from metagpt.const import CONFIG_ROOT, METAGPT_ROOT
from metagpt.utils.yaml_model import YamlModel


class ModelsConfig(YamlModel):
    """
    `config2.yaml` 中的 `models` 配置类。

    属性：
        models (Dict[str, LLMConfig]): 一个字典，将模型名称或类型映射到 LLMConfig 对象。

    方法：
        update_llm_model(cls, value): 验证并更新 LLM 模型配置。
        from_home(cls, path): 从 ~/.metagpt/config2.yaml 加载配置。
        default(cls): 从预定义路径加载默认配置。
        get(self, name_or_type: str) -> Optional[LLMConfig]: 根据模型名称或 API 类型获取 LLMConfig。
    """

    models: Dict[str, LLMConfig] = Field(default_factory=dict)  # 存储模型名称或类型与 LLMConfig 对象的映射

    @field_validator("models", mode="before")
    @classmethod
    def update_llm_model(cls, value):
        """
        验证并更新 LLM 模型配置。

        参数：
            value (Dict[str, Union[LLMConfig, dict]]): LLM 配置字典。

        返回：
            Dict[str, Union[LLMConfig, dict]]: 更新后的 LLM 配置字典。
        """
        for key, config in value.items():
            if isinstance(config, LLMConfig):  # 如果是 LLMConfig 类型
                config.model = config.model or key  # 如果 model 没有设置，则使用 key 作为模型名称
            elif isinstance(config, dict):  # 如果是字典类型
                config["model"] = config.get("model") or key  # 如果 model 没有设置，则使用 key 作为模型名称
        return value

    @classmethod
    def from_home(cls, path):
        """
        从 ~/.metagpt/config2.yaml 加载配置。

        参数：
            path (str): 配置文件的相对路径。

        返回：
            Optional[ModelsConfig]: 加载的 ModelsConfig 对象，如果文件不存在则返回 None。
        """
        pathname = CONFIG_ROOT / path
        if not pathname.exists():
            return None
        return ModelsConfig.from_yaml_file(pathname)

    @classmethod
    def default(cls):
        """
        从预定义路径加载默认配置。

        返回：
            ModelsConfig: 默认的 ModelsConfig 对象。
        """
        default_config_paths: List[Path] = [
            METAGPT_ROOT / "config/config2.yaml",  # 默认配置路径 1
            CONFIG_ROOT / "config2.yaml",  # 默认配置路径 2
        ]

        dicts = [ModelsConfig.read_yaml(path) for path in default_config_paths]  # 读取配置文件内容
        final = merge_dict(dicts)  # 合并配置文件中的字典
        return ModelsConfig(**final)  # 返回合并后的配置对象

    def get(self, name_or_type: str) -> Optional[LLMConfig]:
        """
        根据名称或 API 类型获取 LLMConfig 对象。

        参数：
            name_or_type (str): LLM 模型的名称或 API 类型。

        返回：
            Optional[LLMConfig]: 如果找到则返回 LLMConfig 对象，否则返回 None。
        """
        if not name_or_type:  # 如果没有提供名称或类型，则返回 None
            return None
        model = self.models.get(name_or_type)  # 首先尝试通过名称查找模型
        if model:
            return model
        for m in self.models.values():  # 如果没有找到，则根据 API 类型查找
            if m.api_type == name_or_type:
                return m
        return None  # 如果没有找到，返回 None
