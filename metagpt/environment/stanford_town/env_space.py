#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from typing import Any, Optional, Union

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


class EnvActionType(BaseEnvActionType):
    """定义不同的环境操作类型"""

    NONE = 0  # 无操作，仅获取观察结果

    ADD_TILE_EVENT = 1  # 向某个瓦片添加事件三元组
    RM_TILE_EVENT = 2  # 从瓦片中移除事件三元组
    TURN_TILE_EVENT_IDLE = 3  # 将瓦片上的事件三元组设为“空闲”
    RM_TITLE_SUB_EVENT = 4  # 移除瓦片中与给定主体相关的事件三元组


class EnvAction(BaseEnvAction):
    """环境操作的类型及其相关参数"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置字典，允许任意类型

    action_type: int = Field(default=EnvActionType.NONE, description="操作类型")  # 操作类型，默认为无操作
    coord: npt.NDArray[np.int64] = Field(
        default_factory=lambda: np.zeros(2, dtype=np.int64), description="瓦片坐标"  # 默认坐标为(0, 0)
    )
    subject: str = Field(default="", description="事件的主体名称")  # 事件的主体（第一个元素）
    event: tuple[str, Optional[str], Optional[str], Optional[str]] = Field(
        default=["", None, None, None], description="瓦片事件"  # 事件为一个四元组，第一个元素为主体，其他为可选项
    )

    @field_validator("coord", mode="before")
    @classmethod
    def check_coord(cls, coord) -> npt.NDArray[np.int64]:
        """检查并返回瓦片坐标，如果输入不是 ndarray，则转换为 ndarray"""
        if not isinstance(coord, np.ndarray):
            return np.array(coord)


class EnvObsType(BaseEnvObsType):
    """定义不同的观察类型"""

    NONE = 0  # 获取整个环境的观察结果

    GET_TITLE = 1  # 获取给定瓦片坐标的详细信息字典
    TILE_PATH = 2  # 获取给定瓦片坐标的地址
    TILE_NBR = 3  # 获取给定瓦片坐标的邻居及其视距范围


class EnvObsParams(BaseEnvObsParams):
    """不同类型的观察所需的参数"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置字典，允许任意类型

    obs_type: int = Field(default=EnvObsType.NONE, description="观察类型")  # 观察类型
    coord: npt.NDArray[np.int64] = Field(
        default_factory=lambda: np.zeros(2, dtype=np.int64), description="瓦片坐标"  # 默认坐标为(0, 0)
    )
    level: str = Field(default="", description="瓦片的不同级别")  # 瓦片的级别
    vision_radius: int = Field(default=0, description="当前瓦片的视距半径")  # 瓦片的视距范围

    @field_validator("coord", mode="before")
    @classmethod
    def check_coord(cls, coord) -> npt.NDArray[np.int64]:
        """检查并返回瓦片坐标，如果输入不是 ndarray，则转换为 ndarray"""
        if not isinstance(coord, np.ndarray):
            return np.array(coord)


EnvObsValType = Union[list[list[str]], dict[str, set[tuple[int, int]]], list[list[dict[str, Any]]]]

def get_observation_space() -> spaces.Dict:
    """返回环境观察空间"""
    space = spaces.Dict(
        {"collision_maze": spaces.Discrete(2), "tiles": spaces.Discrete(2), "address_tiles": spaces.Discrete(2)}
    )
    return space


def get_action_space(maze_shape: tuple[int, int]) -> spaces.Dict:
    """返回环境操作空间，其中字段对应操作函数的输入参数（不包括 `action_type`）"""
    space = spaces.Dict(
        {
            "action_type": spaces.Discrete(len(EnvActionType)),  # 操作类型，离散型空间
            "coord": spaces.Box(
                np.array([0, 0], dtype=np.int64), np.array([maze_shape[0], maze_shape[1]], dtype=np.int64)
            ),  # 瓦片的坐标，定义为一个Box空间
            "subject": spaces.Text(256),  # 事件的主体（字符串）
            "event": spaces.Tuple(
                (spaces.Text(256), spaces.Text(256), spaces.Text(256), spaces.Text(256))
            ),  # 事件为四元组，由四个字符串组成
        }
    )
    return space
