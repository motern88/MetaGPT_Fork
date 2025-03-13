#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field


class BaseEnvActionType(IntEnum):
    # # NONE = 0  # 无动作执行，仅获取观察值
    pass


class BaseEnvAction(BaseModel):
    """环境动作类型及其相关的参数，用于动作函数或 API"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_type: int = Field(default=0, description="动作类型")


class BaseEnvObsType(IntEnum):
    # # NONE = 0                     # 获取环境的完整观察值
    pass


class BaseEnvObsParams(BaseModel):
    """用于获取环境观察结果的观察参数，根据不同的 EnvObsType"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    obs_type: int = Field(default=0, description="观察类型")
