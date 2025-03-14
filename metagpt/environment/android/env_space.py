#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pydantic import ConfigDict, Field, field_validator

from metagpt.base.base_env_space import (
    BaseEnvAction,
    BaseEnvActionType,
    BaseEnvObsParams,
    BaseEnvObsType,
)


# 定义了环境动作类型的枚举类
class EnvActionType(BaseEnvActionType):
    NONE = 0  # 不执行任何动作，只获取观察
    SYSTEM_BACK = 1  # 系统后退动作
    SYSTEM_TAP = 2  # 系统点击动作
    USER_INPUT = 3  # 用户输入动作
    USER_LONGPRESS = 4  # 用户长按动作
    USER_SWIPE = 5  # 用户滑动动作
    USER_SWIPE_TO = 6  # 用户滑动到指定位置的动作

# 定义环境动作的类
class EnvAction(BaseEnvAction):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_type: int = Field(default=EnvActionType.NONE, description="动作类型")  # 动作类型
    coord: npt.NDArray[np.int64] = Field(
        default_factory=lambda: np.zeros(2, dtype=np.int64), description="操作坐标"  # 操作坐标（起始位置）
    )
    tgt_coord: npt.NDArray[np.int64] = Field(
        default_factory=lambda: np.zeros(2, dtype=np.int64), description="目标操作坐标"  # 目标操作坐标（目标位置）
    )
    input_txt: str = Field(default="", description="用户输入的文本")  # 用户输入的文本
    orient: str = Field(default="up", description="滑动方向")  # 滑动的方向（up, down, left, right）
    dist: str = Field(default="medium", description="滑动的距离")  # 滑动的距离（long, medium, short）

    # 对坐标进行验证
    @field_validator("coord", "tgt_coord", mode="before")
    @classmethod
    def check_coord(cls, coord) -> npt.NDArray[np.int64]:
        if not isinstance(coord, np.ndarray):
            return np.array(coord)

# 定义环境观察类型的枚举类
class EnvObsType(BaseEnvObsType):
    NONE = 0  # 获取整个环境的观察
    GET_SCREENSHOT = 1  # 获取截图
    GET_XML = 2  # 获取XML文件

# 定义环境观察参数的类
class EnvObsParams(BaseEnvObsParams):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    obs_type: int = Field(default=EnvObsType.NONE, description="观察类型")  # 观察类型
    ss_name: str = Field(default="", description="截图文件名")  # 截图文件名
    xml_name: str = Field(default="", description="XML文件名")  # XML文件名
    local_save_dir: Union[str, Path] = Field(default="", description="保存文件的本地目录")  # 保存文件的本地目录

# 定义环境观察值的类型
EnvObsValType = str

# 获取环境的观察空间
def get_observation_space() -> spaces.Dict:
    space = spaces.Dict({"screenshot": spaces.Text(256), "xml": spaces.Text(256)})
    return space

# 获取环境的动作空间
def get_action_space(device_shape: tuple[int, int]) -> spaces.Dict:
    space = spaces.Dict(
        {
            "action_type": spaces.Discrete(len(EnvActionType)),  # 动作类型的离散空间
            "coord": spaces.Box(
                np.array([0, 0], dtype=np.int64), np.array([device_shape[0], device_shape[1]], dtype=np.int64)
            ),  # 起始坐标的连续空间
            "tgt_coord": spaces.Box(
                np.array([0, 0], dtype=np.int64), np.array([device_shape[0], device_shape[1]], dtype=np.int64)
            ),  # 目标坐标的连续空间
            "input_txt": spaces.Text(256),  # 用户输入文本的空间
            "orient": spaces.Text(16),  # 滑动方向的空间
            "dist": spaces.Text(16),  # 滑动距离的空间
        }
    )
    return space
