#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from enum import Enum

from metagpt.const import MESSAGE_ROUTE_TO_ALL


class RoleType(Enum):
    VILLAGER = "Villager"  # 村民
    WEREWOLF = "Werewolf"  # 狼人
    GUARD = "Guard"  # 守卫
    SEER = "Seer"  # 预言家
    WITCH = "Witch"  # 女巫
    MODERATOR = "Moderator"  # 主持人


class RoleState(Enum):
    ALIVE = "alive"  # 角色存活
    DEAD = "dead"  # 被杀或被毒死
    KILLED = "killed"  # 被狼人或投票杀死
    POISONED = "poisoned"  # 被毒死
    SAVED = "saved"  # 被解药救活
    PROTECTED = "projected"  # 被守卫保护


class RoleActionRes(Enum):
    SAVE = "save"  # 拯救
    PASS = "pass"  # 跳过当前动作


empty_set = set()  # 空集合

# 主持人按顺序宣布每个步骤的规则
STEP_INSTRUCTIONS = {
    0: {
        "content": "天黑了，大家闭上眼睛。我会在晚上与您/您的团队悄悄交谈。",
        "send_to": {RoleType.MODERATOR.value},  # 主持人继续发言
        "restricted_to": empty_set,
    },
    1: {
        "content": "守卫，请睁开眼睛！",
        "send_to": {RoleType.MODERATOR.value},  # 主持人继续发言
        "restricted_to": empty_set,
    },
    2: {
        "content": """守卫，请告诉我今晚你要保护谁？
请从以下存活的玩家中选择一位，或者你可以跳过。例如：保护...""",
        "send_to": {RoleType.GUARD.value},  # 发送给守卫
        "restricted_to": {RoleType.MODERATOR.value, RoleType.GUARD.value},  # 仅限主持人和守卫
    },
    3: {"content": "守卫，请闭上眼睛", "send_to": {RoleType.MODERATOR.value}, "restricted_to": empty_set},
    4: {
        "content": "狼人，请睁开眼睛！",
        "send_to": {RoleType.MODERATOR.value},
        "restricted_to": empty_set,
    },
    5: {
        "content": """狼人，我悄悄告诉你们 {werewolf_players} 是
所有 {werewolf_num} 只狼人！记住你们是队友，其他玩家不是狼人。
请从以下存活的玩家中选择一位进行杀害：
{living_players}。例如：杀...""",
        "send_to": {RoleType.WEREWOLF.value},
        "restricted_to": {RoleType.MODERATOR.value, RoleType.WEREWOLF.value},  # 仅限主持人和狼人
    },
    6: {"content": "狼人，请闭上眼睛", "send_to": {RoleType.MODERATOR.value}, "restricted_to": empty_set},
    7: {"content": "女巫，请睁开眼睛！", "send_to": {RoleType.MODERATOR.value}, "restricted_to": empty_set},
    8: {
        "content": """女巫，今晚 {player_hunted} 被狼人杀害。
你有一瓶解药，是否想救他/她？如果是，请说“救”，否则说“跳过”。""",
        "send_to": {RoleType.WITCH.value},
        "restricted_to": {RoleType.MODERATOR.value, RoleType.WITCH.value},  # 仅限主持人和女巫
    },
    9: {
        "content": """女巫，你也有一瓶毒药，是否想用它毒死某个存活的玩家？
从以下存活的玩家中选择一位。输入“毒死PlayerX”，替换PlayerX为实际玩家名字，若不想使用，输入“跳过”。""",
        "send_to": {RoleType.WITCH.value},
        "restricted_to": {RoleType.MODERATOR.value, RoleType.WITCH.value},  # 仅限主持人和女巫
    },
    10: {"content": "女巫，请闭上眼睛", "send_to": {RoleType.MODERATOR.value}, "restricted_to": empty_set},
    11: {"content": "预言家，请睁开眼睛！", "send_to": {RoleType.MODERATOR.value}, "restricted_to": empty_set},
    12: {
        "content": """预言家，今晚你可以查看一位玩家的身份。
你要查看谁的身份？从以下存活玩家中选择一位：{living_players}。""",
        "send_to": {RoleType.SEER.value},
        "restricted_to": {RoleType.MODERATOR.value, RoleType.SEER.value},  # 仅限主持人和预言家
    },
    13: {"content": "预言家，请闭上眼睛", "send_to": {RoleType.MODERATOR.value}, "restricted_to": empty_set},
    # 第一天白天
    14: {
        "content": """天亮了，除了被杀害的玩家，所有人都醒来了。""",
        "send_to": {RoleType.MODERATOR.value},
        "restricted_to": empty_set,
    },
    15: {
        "content": "{player_current_dead} 昨晚被杀害！",
        "send_to": {RoleType.MODERATOR.value},
        "restricted_to": empty_set,
    },
    16: {
        "content": """存活的玩家：{living_players}，现在自由发言，分享你的观察与思考，
并决定是否揭示你的身份。""",
        "send_to": {MESSAGE_ROUTE_TO_ALL},  # 发送给所有玩家，白天可以自由发言
        "restricted_to": empty_set,
    },
    17: {
        "content": """现在投票，告诉我你认为谁是狼人。不要提及你的角色。
从以下存活的玩家中选择一位，请说：“我投票淘汰...”""",
        "send_to": {MESSAGE_ROUTE_TO_ALL},
        "restricted_to": empty_set,
    },
    18: {
        "content": """{player_current_dead} 被淘汰。""",
        "send_to": {RoleType.MODERATOR.value},
        "restricted_to": empty_set,
    },
}
