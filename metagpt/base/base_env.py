#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : base environment

import typing
from abc import abstractmethod
from typing import Any, Optional

from metagpt.base.base_env_space import BaseEnvAction, BaseEnvObsParams
from metagpt.base.base_serialization import BaseSerialization

if typing.TYPE_CHECKING:
    from metagpt.schema import Message


class BaseEnvironment(BaseSerialization):
    """基础环境类"""

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        重置环境并返回初始观察值。

        参数:
            seed: 可选的随机种子，用于初始化环境。
            options: 可选的环境配置选项。

        返回:
            返回一个包含初始观察值的元组，格式为 (环境状态, 额外信息)。
        """

    @abstractmethod
    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        """
        获取环境的部分观察值。

        参数:
            obs_params: 可选的观察参数，定义获取部分观察时的细节。

        返回:
            返回当前观察值。
        """

    @abstractmethod
    def step(self, action: BaseEnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        根据给定的动作更新环境，并返回新的观察结果。

        参数:
            action: 需要执行的动作。

        返回:
            返回一个包含以下内容的元组：
            - 新的环境状态
            - 奖励值
            - 环境是否已终止
            - 动作是否有效
            - 额外信息
        """

    @abstractmethod
    def publish_message(self, message: "Message", peekable: bool = True) -> bool:
        """
        发布消息给接收者。

        参数:
            message: 需要发布的消息。
            peekable: 是否允许消息被预览（可选）。

        返回:
            如果消息成功发布，返回 True；否则返回 False。
        """

    @abstractmethod
    async def run(self, k=1):
        """
        同时处理所有任务。

        参数:
            k: 处理任务的数量，默认为 1。

        返回:
            None
        """
