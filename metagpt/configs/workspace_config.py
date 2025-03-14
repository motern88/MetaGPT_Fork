#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 19:09
@Author  : alexanderwu
@File    : workspace_config.py
"""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import field_validator, model_validator

from metagpt.const import DEFAULT_WORKSPACE_ROOT
from metagpt.utils.yaml_model import YamlModel


class WorkspaceConfig(YamlModel):
    """工作空间配置类

    该类用于配置工作空间的路径和UID等信息。

    属性：
        path (Path): 工作空间的路径，默认为 `DEFAULT_WORKSPACE_ROOT`。
        use_uid (bool): 是否使用UID来创建独特的工作空间，默认为 `False`。
        uid (str): UID，用于创建唯一的工作空间目录，默认为空字符串。
    """

    path: Path = DEFAULT_WORKSPACE_ROOT  # 工作空间的路径，默认为默认工作空间根路径
    use_uid: bool = False  # 是否使用UID来创建独特的工作空间目录
    uid: str = ""  # 唯一标识符（UID），用于创建工作空间的唯一路径

    @field_validator("path")
    @classmethod
    def check_workspace_path(cls, v):
        """验证并转换工作空间路径为Path类型

        如果传入的路径是字符串，则将其转换为Path对象。

        Args:
            v (str or Path): 输入的工作空间路径

        Returns:
            Path: 转换后的工作空间路径
        """
        if isinstance(v, str):
            v = Path(v)  # 如果路径是字符串，转换为Path对象
        return v

    @model_validator(mode="after")
    def check_uid_and_update_path(self):
        """检查UID并更新工作空间路径

        如果启用了UID并且UID为空，则生成一个新的UID，并将其添加到工作空间路径中。
        然后确保工作空间路径存在，如果不存在则创建。

        Returns:
            WorkspaceConfig: 更新后的工作空间配置对象
        """
        if self.use_uid and not self.uid:
            # 如果启用了UID且UID为空，生成一个新的UID并更新工作空间路径
            self.uid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[-8:]}"
            self.path = self.path / self.uid  # 将UID添加到路径中

        # 如果工作空间路径不存在，则创建该路径及其父目录
        self.path.mkdir(parents=True, exist_ok=True)
        return self
