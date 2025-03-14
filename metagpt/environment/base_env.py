#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : base env of executing environment

import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Set, Union

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from metagpt.base import BaseEnvironment, BaseRole
from metagpt.base.base_env_space import BaseEnvAction, BaseEnvObsParams
from metagpt.context import Context
from metagpt.environment.api.env_api import (
    EnvAPIAbstract,
    ReadAPIRegistry,
    WriteAPIRegistry,
)
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.schema import Message
from metagpt.utils.common import get_function_schema, is_coroutine_func, is_send_to
from metagpt.utils.git_repository import GitRepository


class EnvType(Enum):
    ANDROID = "Android"  # Android 环境
    GYM = "Gym"  # 健身房环境
    WEREWOLF = "Werewolf"  # 狼人环境
    MINECRAFT = "Minecraft"  # 我的世界环境
    STANFORDTOWN = "StanfordTown"  # 斯坦福镇环境


env_write_api_registry = WriteAPIRegistry()  # 注册写API
env_read_api_registry = ReadAPIRegistry()  # 注册读API


def mark_as_readable(func):
    """标记函数为可读取的，它从 ExtEnv 中观察某些内容"""
    env_read_api_registry[func.__name__] = get_function_schema(func)  # 将函数的schema注册到读取API中
    return func


def mark_as_writeable(func):
    """标记函数为可写的，它对 ExtEnv 进行某些操作"""
    env_write_api_registry[func.__name__] = get_function_schema(func)  # 将函数的schema注册到写入API中
    return func


class ExtEnv(BaseEnvironment, BaseModel):
    """External Env，用于集成实际的游戏环境"""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置模型允许任意类型

    action_space: spaces.Space[ActType] = Field(default_factory=spaces.Space, exclude=True)  # 动作空间
    observation_space: spaces.Space[ObsType] = Field(default_factory=spaces.Space, exclude=True)  # 观察空间

    def _check_api_exist(self, rw_api: Optional[str] = None):
        """检查API是否存在"""
        if not rw_api:
            raise ValueError(f"{rw_api} 不存在")

    def get_all_available_apis(self, mode: str = "read") -> list[Any]:
        """获取所有可用的读/写API定义"""
        assert mode in ["read", "write"]
        if mode == "read":
            return env_read_api_registry.get_apis()  # 获取读API列表
        else:
            return env_write_api_registry.get_apis()  # 获取写API列表

    async def read_from_api(self, env_action: Union[str, EnvAPIAbstract]):
        """通过特定API从ExtEnv读取观察信息"""
        if isinstance(env_action, str):
            env_read_api = env_read_api_registry.get(api_name=env_action)["func"]
            self._check_api_exist(env_read_api)
            if is_coroutine_func(env_read_api):
                res = await env_read_api(self)
            else:
                res = env_read_api(self)
        elif isinstance(env_action, EnvAPIAbstract):
            env_read_api = env_read_api_registry.get(api_name=env_action.api_name)["func"]
            self._check_api_exist(env_read_api)
            if is_coroutine_func(env_read_api):
                res = await env_read_api(self, *env_action.args, **env_action.kwargs)
            else:
                res = env_read_api(self, *env_action.args, **env_action.kwargs)
        return res

    async def write_thru_api(self, env_action: Union[str, Message, EnvAPIAbstract, list[EnvAPIAbstract]]):
        """通过特定API执行写操作"""
        res = None
        if isinstance(env_action, Message):
            self.publish_message(env_action)  # 发布消息
        elif isinstance(env_action, EnvAPIAbstract):
            env_write_api = env_write_api_registry.get(env_action.api_name)["func"]
            self._check_api_exist(env_write_api)
            if is_coroutine_func(env_write_api):
                res = await env_write_api(self, *env_action.args, **env_action.kwargs)
            else:
                res = env_write_api(self, *env_action.args, **env_action.kwargs)

        return res

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """实现这个方法以获取初始观察结果"""

    @abstractmethod
    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        """实现这个方法，如果您想从环境中获取部分观察结果"""

    @abstractmethod
    def step(self, action: BaseEnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """实现这个方法来输入一个动作并获取环境的新的观察结果"""


class Environment(ExtEnv):
    """环境，承载一批角色，角色可以向环境发布消息，可以被其他角色观察到
    Environment，托管一批角色，角色可以向环境发布消息，也可以被其他角色观察到
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置模型允许任意类型

    desc: str = Field(default="")  # 环境描述
    roles: dict[str, SerializeAsAny[BaseRole]] = Field(default_factory=dict, validate_default=True)  # 环境中的角色
    member_addrs: Dict[BaseRole, Set] = Field(default_factory=dict, exclude=True)  # 成员地址
    history: Memory = Field(default_factory=Memory)  # 用于调试的历史记录
    context: Context = Field(default_factory=Context, exclude=True)  # 上下文

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pass

    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        pass

    def step(self, action: BaseEnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        pass

    @model_validator(mode="after")
    def init_roles(self):
        """初始化角色"""
        self.add_roles(self.roles.values())  # 向环境中添加角色
        return self

    def add_role(self, role: BaseRole):
        """增加一个角色到当前环境"""
        self.roles[role.name] = role
        role.set_env(self)  # 将角色关联到环境
        role.context = self.context  # 设置角色的上下文

    def add_roles(self, roles: Iterable[BaseRole]):
        """增加一批角色到当前环境"""
        for role in roles:
            self.roles[role.name] = role

        for role in roles:  # 设置角色的系统消息
            role.context = self.context
            role.set_env(self)

    def publish_message(self, message: Message, peekable: bool = True) -> bool:
        """
        分发消息到接收者。
        根据RFC 116第2.2.1章节中的消息路由结构设计，与RFC 113中为整个系统规划的内容一致，
        消息中的路由信息仅负责指定消息接收者，而不关心消息接收者的位置。如何将消息路由到接收者是
        由RFC 113中设计的传输框架来处理的问题。
        """
        logger.debug(f"publish_message: {message.dump()}")
        found = False
        # 根据RFC 113第2.2.3.2章节中的路由特性规划
        for role, addrs in self.member_addrs.items():
            if is_send_to(message, addrs):
                role.put_message(message)
                found = True
        if not found:
            logger.warning(f"Message no recipients: {message.dump()}")
        self.history.add(message)  # 供调试使用

        return True

    async def run(self, k=1):
        """处理所有角色的运行
        一次性处理所有角色的操作
        """
        for _ in range(k):
            futures = []
            for role in self.roles.values():
                if role.is_idle:
                    continue
                future = role.run()
                futures.append(future)

            if futures:
                await asyncio.gather(*futures)
            logger.debug(f"is idle: {self.is_idle}")

    def get_roles(self) -> dict[str, BaseRole]:
        """获取环境内的所有角色
        一次性处理所有角色的操作
        """
        return self.roles

    def get_role(self, name: str) -> BaseRole:
        """获取环境内指定的角色
        获取所有环境中的角色
        """
        return self.roles.get(name, None)

    def role_names(self) -> list[str]:
        return [i.name for i in self.roles.values()]

    @property
    def is_idle(self):
        """如果为True，表示所有动作都已执行完毕。"""
        for r in self.roles.values():
            if not r.is_idle:
                return False
        return True

    def get_addresses(self, obj):
        """获取对象的地址。"""
        return self.member_addrs.get(obj, {})

    def set_addresses(self, obj, addresses):
        """设置对象的地址"""
        self.member_addrs[obj] = addresses

    def archive(self, auto_archive=True):
        """归档环境数据"""
        if auto_archive and self.context.kwargs.get("project_path"):
            git_repo = GitRepository(self.context.kwargs.project_path)
            git_repo.archive()