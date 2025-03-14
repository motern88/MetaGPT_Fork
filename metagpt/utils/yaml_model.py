#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 10:18
@Author  : alexanderwu
@File    : YamlModel.py
"""
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, model_validator


class YamlModel(BaseModel):
    """Yaml模型的基类"""

    extra_fields: Optional[Dict[str, str]] = None  # 可选的额外字段，存储为字典

    @classmethod
    def read_yaml(cls, file_path: Path, encoding: str = "utf-8") -> Dict:
        """读取yaml文件并返回一个字典

        Args:
            file_path (Path): yaml文件路径
            encoding (str): 文件编码，默认为utf-8

        Returns:
            dict: 读取的yaml内容，转换为字典
        """
        if not file_path.exists():
            return {}  # 如果文件不存在，返回空字典
        with open(file_path, "r", encoding=encoding) as file:
            return yaml.safe_load(file)  # 安全地加载yaml内容为字典

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "YamlModel":
        """从yaml文件创建一个YamlModel实例

        Args:
            file_path (Path): yaml文件路径

        Returns:
            YamlModel: 从yaml文件创建的YamlModel实例
        """
        return cls(**cls.read_yaml(file_path))  # 通过读取yaml文件并传入字典来实例化

    def to_yaml_file(self, file_path: Path, encoding: str = "utf-8") -> None:
        """将YamlModel实例保存到yaml文件

        Args:
            file_path (Path): 输出的yaml文件路径
            encoding (str): 文件编码，默认为utf-8
        """
        with open(file_path, "w", encoding=encoding) as file:
            yaml.dump(self.model_dump(), file)  # 将实例的数据以yaml格式写入文件


class YamlModelWithoutDefault(YamlModel):
    """没有默认值的YamlModel子类"""

    @model_validator(mode="before")
    @classmethod
    def check_not_default_config(cls, values):
        """检查配置中是否存在默认值

        如果配置中包含'YOUR'，则抛出错误，要求用户在config2.yaml中设置配置。

        Args:
            values (dict): 配置的值字典

        Raises:
            ValueError: 如果配置中包含'YOUR'，则抛出错误
        """
        if any(["YOUR" in v for v in values]):
            raise ValueError("Please set your config in config2.yaml")  # 如果配置包含'YOUR'，则抛出错误
        return values  # 返回原始值
