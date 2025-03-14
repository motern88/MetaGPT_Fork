#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : werewolf observation/action space and its action definition

from gymnasium import spaces
from pydantic import ConfigDict, Field

from metagpt.base.base_env_space import BaseEnvAction, BaseEnvActionType
from metagpt.environment.werewolf.const import STEP_INSTRUCTIONS


# 定义游戏中行动类型的枚举
class EnvActionType(BaseEnvActionType):
    NONE = 0  # 无动作，只是获取观察
    WOLF_KILL = 1  # 狼杀人
    VOTE_KILL = 2  # 投票杀人
    WITCH_POISON = 3  # 女巫毒死某人
    WITCH_SAVE = 4  # 女巫救人
    GUARD_PROTECT = 5  # 守卫保护某人
    PROGRESS_STEP = 6  # 进度步进


# 定义游戏中的行动对象
class EnvAction(BaseEnvAction):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 动作类型
    action_type: int = Field(default=EnvActionType.NONE, description="动作类型")
    # 执行动作的玩家名称
    player_name: str = Field(default="", description="执行该动作的玩家名称")
    # 被选择执行动作的目标玩家名称
    target_player_name: str = Field(default="", description="目标玩家的名称")


# 获取观察空间（环境的状态信息）
def get_observation_space() -> spaces.Dict:
    # 观察空间包括多个信息，游戏状态、步骤索引、存活玩家、狼人玩家等
    space = spaces.Dict(
        {
            "game_setup": spaces.Text(256),  # 游戏的设置描述
            "step_idx": spaces.Discrete(len(STEP_INSTRUCTIONS)),  # 当前步骤的索引
            "living_players": spaces.Tuple(
                (spaces.Text(16), spaces.Text(16))
            ),  # 存活的玩家名称元组（可以扩展为可变长度）
            "werewolf_players": spaces.Tuple(
                (spaces.Text(16), spaces.Text(16))
            ),  # 狼人的玩家名称元组（可以扩展为可变长度）
            "player_hunted": spaces.Text(16),  # 被猎杀的玩家名称
            "player_current_dead": spaces.Tuple(
                (spaces.Text(16), spaces.Text(16))
            ),  # 当前死亡玩家的名称元组（可以扩展为可变长度）
            "witch_poison_left": spaces.Discrete(2),  # 女巫剩余的毒药数量（0 或 1）
            "witch_antidote_left": spaces.Discrete(2),  # 女巫剩余的解药数量（0 或 1）
            "winner": spaces.Text(16),  # 游戏赢家的名称
            "win_reason": spaces.Text(64),  # 游戏胜利的原因
        }
    )
    return space


# 获取行动空间（玩家可以执行的动作）
def get_action_space() -> spaces.Dict:
    # 动作空间包括动作类型、执行玩家名称和目标玩家名称
    space = spaces.Dict(
        {
            "action_type": spaces.Discrete(len(EnvActionType)),  # 动作类型的离散值
            "player_name": spaces.Text(16),  # 执行动作的玩家名称
            "target_player_name": spaces.Text(16),  # 目标玩家名称（用于执行特定的动作）
        }
    )
    return space
