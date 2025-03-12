#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : team.py
@Modified By: mashenquan, 2023/11/27. Add an archiving operation after completing the project, as specified in
        Section 2.2.3.3 of RFC 135.
"""

import warnings
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from metagpt.const import SERDESER_PATH
from metagpt.context import Context
from metagpt.environment import Environment
from metagpt.environment.mgx.mgx_env import MGXEnv
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.common import (
    NoMoneyException,
    read_json_file,
    serialize_decorator,
    write_json_file,
)


class Team(BaseModel):
    """
    团队：拥有一个或多个角色（代理人）、标准操作流程（SOP）和用于即时通讯的环境，
    专门用于多代理活动，如协作编写可执行代码。
    """

    # 配置模型，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 环境对象，默认为空
    env: Optional[Environment] = None
    # 投资金额，默认为 10.0
    investment: float = Field(default=10.0)
    # 团队的创意或项目理念，默认为空
    idea: str = Field(default="")
    # 是否使用 MGX 环境，默认为 True
    use_mgx: bool = Field(default=True)

    def __init__(self, context: Context = None, **data: Any):
        # 初始化方法，设置上下文环境，并根据传入数据进行相应的初始化
        super(Team, self).__init__(**data)
        ctx = context or Context()
        # 根据是否使用 MGX 环境来决定使用哪个环境
        if not self.env and not self.use_mgx:
            self.env = Environment(context=ctx)
        elif not self.env and self.use_mgx:
            self.env = MGXEnv(context=ctx)
        else:
            self.env.context = ctx  # 如果环境已分配，则更新上下文
        # 如果传入角色数据，则进行雇佣
        if "roles" in data:
            self.hire(data["roles"])
        # 如果传入环境描述，则设置环境的描述
        if "env_desc" in data:
            self.env.desc = data["env_desc"]

    def serialize(self, stg_path: Path = None):
        """将团队数据序列化并保存到文件"""
        stg_path = SERDESER_PATH.joinpath("team") if stg_path is None else stg_path
        team_info_path = stg_path.joinpath("team.json")
        serialized_data = self.model_dump()
        serialized_data["context"] = self.env.context.serialize()

        # 写入 JSON 文件
        write_json_file(team_info_path, serialized_data)

    @classmethod
    def deserialize(cls, stg_path: Path, context: Context = None) -> "Team":
        """反序列化团队数据，恢复团队对象"""
        # 恢复团队信息
        team_info_path = stg_path.joinpath("team.json")
        if not team_info_path.exists():
            raise FileNotFoundError(
                "恢复存储的元数据文件 `team.json` 不存在，请重新开始项目。"
            )

        team_info: dict = read_json_file(team_info_path)
        ctx = context or Context()
        ctx.deserialize(team_info.pop("context", None))
        team = Team(**team_info, context=ctx)
        return team

    def hire(self, roles: list[Role]):
        """雇佣角色进行合作"""
        self.env.add_roles(roles)

    @property
    def cost_manager(self):
        """获取成本管理器"""
        return self.env.context.cost_manager

    def invest(self, investment: float):
        """投资公司，超出最大预算时抛出 NoMoneyException 异常"""
        self.investment = investment
        self.cost_manager.max_budget = investment
        logger.info(f"投资金额：${investment}。")

    def _check_balance(self):
        """检查账户余额，若余额不足则抛出异常"""
        if self.cost_manager.total_cost >= self.cost_manager.max_budget:
            raise NoMoneyException(self.cost_manager.total_cost, f"资金不足: {self.cost_manager.max_budget}")

    def run_project(self, idea, send_to: str = ""):
        """根据用户需求启动项目"""
        self.idea = idea

        # 发布用户需求
        self.env.publish_message(Message(content=idea))

    def start_project(self, idea, send_to: str = ""):
        """
        已废弃：此方法将在未来移除，请使用 `run_project` 方法代替。
        """
        warnings.warn(
            "`start_project` 方法已废弃，将在未来移除。请使用 `run_project` 方法代替。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_project(idea=idea, send_to=send_to)

    @serialize_decorator
    async def run(self, n_round=3, idea="", send_to="", auto_archive=True):
        """运行公司直到目标回合数或资金不足"""
        if idea:
            self.run_project(idea=idea, send_to=send_to)

        while n_round > 0:
            if self.env.is_idle:
                logger.debug("所有角色都处于空闲状态。")
                break
            n_round -= 1
            self._check_balance()  # 检查账户余额
            await self.env.run()  # 执行环境中的任务

            logger.debug(f"剩余回合数 {n_round=}.")
        # 自动归档
        self.env.archive(auto_archive)
        return self.env.history
