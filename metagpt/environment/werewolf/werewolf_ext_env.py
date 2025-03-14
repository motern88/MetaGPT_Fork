#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : The werewolf game external environment to integrate with

import random
from collections import Counter
from typing import Any, Callable, Optional

from pydantic import ConfigDict, Field

from metagpt.base.base_env_space import BaseEnvObsParams
from metagpt.environment.base_env import ExtEnv, mark_as_readable, mark_as_writeable
from metagpt.environment.werewolf.const import STEP_INSTRUCTIONS, RoleState, RoleType
from metagpt.environment.werewolf.env_space import EnvAction, EnvActionType
from metagpt.logs import logger


class WerewolfExtEnv(ExtEnv):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 玩家状态字典，保存每个玩家的角色类型和角色状态
    players_state: dict[str, tuple[str, RoleState]] = Field(
        default_factory=dict, description="the player's role type and state by player_name"
    )

    # 游戏回合索引，表示当前的回合
    round_idx: int = Field(default=0)
    # 当前回合的步骤索引
    step_idx: int = Field(default=0)
    eval_step_idx: list[int] = Field(default=[])
    # 每回合的步骤数
    per_round_steps: int = Field(default=len(STEP_INSTRUCTIONS))

    # 游戏的全局状态
    game_setup: str = Field(default="", description="game setup including role and its num")
    special_role_players: list[str] = Field(default=[])
    winner: Optional[str] = Field(default=None)
    win_reason: Optional[str] = Field(default=None)
    witch_poison_left: int = Field(default=1, description="should be 1 or 0")
    witch_antidote_left: int = Field(default=1, description="should be 1 or 0")

    # 游戏当前回合状态，一回合是从闭眼到下次闭眼
    round_hunts: dict[str, str] = Field(default_factory=dict, description="nighttime wolf hunt result")
    round_votes: dict[str, str] = Field(
        default_factory=dict, description="daytime all players vote result, key=voter, value=voted one"
    )
    player_hunted: Optional[str] = Field(default=None)
    player_protected: Optional[str] = Field(default=None)
    is_hunted_player_saved: bool = Field(default=False)
    player_poisoned: Optional[str] = Field(default=None)
    player_current_dead: list[str] = Field(default=[])

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """当前未使用"""
        pass

    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        """当前未使用"""
        pass

    def _get_obs(self):
        """获取当前的观察结果"""
        return {
            "game_setup": self.game_setup,
            "step_idx": self.step_idx,
            "living_players": self.living_players,
            "werewolf_players": self.werewolf_players,  # 目前缺少观察隔离
            "player_hunted": self.player_hunted,
            "player_current_dead": self.player_current_dead,
            "witch_poison_left": self.witch_poison_left,
            "witch_antidote_left": self.witch_antidote_left,
            "winner": self.winner,
            "win_reason": self.win_reason,
        }

    def step(self, action: EnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """处理每步的行动"""
        action_type = action.action_type
        player_name = action.player_name
        target_player_name = action.target_player_name
        if action_type == EnvActionType.WOLF_KILL:
            self.wolf_kill_someone(wolf_name=player_name, player_name=target_player_name)
        elif action_type == EnvActionType.VOTE_KILL:
            self.vote_kill_someone(voter_name=player_name, player_name=target_player_name)
        elif action_type == EnvActionType.WITCH_POISON:
            self.witch_poison_someone(witch_name=player_name, player_name=target_player_name)
        elif action_type == EnvActionType.WITCH_SAVE:
            self.witch_save_someone(witch_name=player_name, player_name=target_player_name)
        elif action_type == EnvActionType.GUARD_PROTECT:
            self.guard_protect_someone(guard_name=player_name, player_name=target_player_name)
        elif action_type == EnvActionType.PROGRESS_STEP:
            self.progress_step()
        elif action_type == EnvActionType.NONE:
            pass
        else:
            raise ValueError(f"不支持的 action_type: {action_type}")

        # 更新游戏状态
        self.update_game_states()
        # 检查游戏是否结束
        terminated = self._check_game_finish()
        # 获取当前观察结果
        obs = self._get_obs()
        return obs, 1.0, terminated, False, {}

    def _check_game_finish(self) -> bool:
        """检查游戏是否结束，若结束返回 True，否则返回 False"""
        terminated = False
        # 获取剩余的狼人、村民和特殊角色玩家
        living_werewolf = [p for p in self.werewolf_players if p in self.living_players]
        living_villagers = [p for p in self.villager_players if p in self.living_players]
        living_special_roles = [p for p in self.special_role_players if p in self.living_players]

        # 判断游戏结束条件
        if not living_werewolf:
            self.winner = "好人"
            self.win_reason = "狼人全死"
            terminated = True
        elif not living_villagers or not living_special_roles:
            self.winner = "狼人"
            self.win_reason = "村民全死" if not living_villagers else "特殊角色全死"
            terminated = True
        return terminated

    @property
    def living_players(self) -> list[str]:
        """返回存活的玩家列表"""
        player_names = []
        for name, roletype_state in self.players_state.items():
            if roletype_state[1] in [RoleState.ALIVE, RoleState.SAVED]:
                player_names.append(name)
        return player_names

    def _role_type_players(self, role_type: str) -> list[str]:
        """返回特定角色类型的玩家名称"""
        player_names = []
        for name, roletype_state in self.players_state.items():
            if role_type in roletype_state[0]:
                player_names.append(name)
        return player_names

    @property
    def werewolf_players(self) -> list[str]:
        """返回狼人玩家的名称列表"""
        player_names = self._role_type_players(role_type=RoleType.WEREWOLF.value)
        return player_names

    @property
    def villager_players(self) -> list[str]:
        """返回村民玩家的名称列表"""
        player_names = self._role_type_players(role_type=RoleType.VILLAGER.value)
        return player_names

    def _init_players_state(self, players: list["Role"]):
        """初始化玩家状态"""
        for play in players:
            self.players_state[play.name] = (play.profile, RoleState.ALIVE)

        # 初始化特殊角色玩家
        self.special_role_players = [
            p for p in self.living_players if p not in self.werewolf_players + self.villager_players
        ]

    def init_game_setup(
            self,
            role_uniq_objs: list[object],
            num_villager: int = 2,
            num_werewolf: int = 2,
            shuffle=True,
            add_human=False,
            use_reflection=True,
            use_experience=False,
            use_memory_selection=False,
            new_experience_version="",
            prepare_human_player=Callable,
    ) -> tuple[str, list]:
        """初始化游戏，分配不同角色的数量"""
        role_objs = []
        # 根据角色类型和数量分配角色
        for role_obj in role_uniq_objs:
            if RoleType.VILLAGER.value in str(role_obj):
                role_objs.extend([role_obj] * num_villager)
            elif RoleType.WEREWOLF.value in str(role_obj):
                role_objs.extend([role_obj] * num_werewolf)
            else:
                role_objs.append(role_obj)
        if shuffle:
            random.shuffle(role_objs)
        # 如果需要增加人类玩家，随机选择一个角色并分配
        if add_human:
            assigned_role_idx = random.randint(0, len(role_objs) - 1)
            assigned_role = role_objs[assigned_role_idx]
            role_objs[assigned_role_idx] = prepare_human_player(assigned_role)  # TODO

        # 创建玩家对象
        players = [
            role(
                name=f"Player{i + 1}",
                use_reflection=use_reflection,
                use_experience=use_experience,
                use_memory_selection=use_memory_selection,
                new_experience_version=new_experience_version,
            )
            for i, role in enumerate(role_objs)
        ]

        if add_human:
            logger.info(f"你被分配到 {players[assigned_role_idx].name}({players[assigned_role_idx].profile})")

        # 记录游戏的设置
        game_setup = ["游戏设置:"] + [f"{player.name}: {player.profile}," for player in players]
        self.game_setup = "\n".join(game_setup)

        # 初始化玩家状态
        self._init_players_state(players)

        return self.game_setup, players

    def _update_players_state(self, player_names: list[str], state: RoleState = RoleState.KILLED):
        """更新玩家的状态"""
        for player_name in player_names:
            if player_name in self.players_state:
                roletype_state = self.players_state[player_name]
                self.players_state[player_name] = (roletype_state[0], state)

    def _check_valid_role(self, player_name: str, role_type: str) -> bool:
        """检查玩家角色是否合法"""
        roletype_state = self.players_state.get(player_name)
        return True if roletype_state and role_type in roletype_state[0] else False

    def _check_player_continue(self, player_name: str, particular_step: int = -1) -> bool:
        """检查玩家是否可以继续进行某个操作"""
        step_idx = self.step_idx % self.per_round_steps
        if particular_step > 0 and step_idx != particular_step:  # 特定步骤
            return False
        if player_name not in self.living_players:
            return False
        return True

    @mark_as_readable
    def curr_step_instruction(self) -> dict:
        """返回当前步骤的指令"""
        step_idx = self.step_idx % len(STEP_INSTRUCTIONS)
        instruction = STEP_INSTRUCTIONS[step_idx]
        self.step_idx += 1
        return instruction

    @mark_as_writeable
    def progress_step(self):
        """推进到下一步骤"""
        self.step_idx += 1

    @mark_as_readable
    def get_players_state(self, player_names: list[str]) -> dict[str, RoleState]:
        """获取指定玩家的状态"""
        players_state = {
            player_name: self.players_state[player_name][1]  # 只返回角色状态
            for player_name in player_names
            if player_name in self.players_state
        }
        return players_state

    @mark_as_writeable
    def vote_kill_someone(self, voter_name: str, player_name: str = None):
        """玩家在白天投票投死人
        player_name: 如果为 None，视为弃权
        """
        if not self._check_player_continue(voter_name, particular_step=18):  # 18=步骤编号
            return

        self.round_votes[voter_name] = player_name
        # 检查所有活着的玩家是否完成投票，然后决定死亡玩家
        if list(self.round_votes.keys()) == self.living_players:
            voted_all = list(self.round_votes.values())  # TODO 处理平票的情况，检查谁先被投票
            voted_all = [item for item in voted_all if item]
            self.player_current_dead = [Counter(voted_all).most_common()[0][0]]
            self._update_players_state(self.player_current_dead)

    @mark_as_writeable
    def wolf_kill_someone(self, wolf_name: str, player_name: str):
        """狼人杀人
        只有狼人才能执行此操作
        """
        if not self._check_valid_role(wolf_name, RoleType.WEREWOLF.value):
            return
        if not self._check_player_continue(wolf_name, particular_step=6):  # 5=步骤编号
            return

        self.round_hunts[wolf_name] = player_name
        # 如果所有活着的狼人都完成了猎杀操作，决定猎杀玩家
        # if list(self.round_hunts.keys()) == living_werewolf:
        #     hunted_all = list(self.round_hunts.values())
        #     self.player_hunted = Counter(hunted_all).most_common()[0][0]
        self.player_hunted = player_name

    def _witch_poison_or_save_someone(
            self, witch_name: str, player_name: str = None, state: RoleState = RoleState.POISONED
    ):
        """女巫施毒或治疗某个玩家"""
        if not self._check_valid_role(witch_name, RoleType.WITCH.value):
            return
        if not self._check_player_continue(player_name):
            return

        assert state in [RoleState.POISONED, RoleState.SAVED]
        self._update_players_state([player_name], state)
        if state == RoleState.POISONED:
            self.player_poisoned = player_name
            self.witch_poison_left -= 1
        else:
            self.is_hunted_player_saved = True
            self.witch_antidote_left -= 1

    @mark_as_writeable
    def witch_poison_someone(self, witch_name: str, player_name: str = None):
        """女巫施毒"""
        self._witch_poison_or_save_someone(witch_name, player_name, RoleState.POISONED)

    @mark_as_writeable
    def witch_save_someone(self, witch_name: str, player_name: str = None):
        """女巫拯救"""
        self._witch_poison_or_save_someone(witch_name, player_name, RoleState.SAVED)

    @mark_as_writeable
    def guard_protect_someone(self, guard_name: str, player_name: str = None):
        """守卫保护某个玩家"""
        if not self._check_valid_role(guard_name, RoleType.GUARD.value):
            return
        if not self._check_player_continue(player_name):
            return
        self.player_protected = player_name

    @mark_as_writeable
    def update_game_states(self):
        """更新游戏状态"""
        step_idx = self.step_idx % self.per_round_steps
        if step_idx not in [15, 18] or self.step_idx in self.eval_step_idx:
            return
        else:
            self.eval_step_idx.append(self.step_idx)  # 记录评估，避免在同一步骤中重复评估

        if step_idx == 15:  # 步骤编号
            # 夜晚结束：在所有特殊角色行动后，处理整个夜晚
            self.player_current_dead = []  # 重置

            if self.player_hunted != self.player_protected and not self.is_hunted_player_saved:
                self.player_current_dead.append(self.player_hunted)
            if self.player_poisoned:
                self.player_current_dead.append(self.player_poisoned)

            self._update_players_state(self.player_current_dead)
            # 重置
            self.player_hunted = None
            self.player_protected = None
            self.is_hunted_player_saved = False
            self.player_poisoned = None
        elif step_idx == 18:
            # 使用 vote_kill_someone 更新
            pass
