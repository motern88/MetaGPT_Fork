#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from metagpt.const import METAGPT_ROOT

# 对于Minecraft游戏代理（Minecraft Game Agent）
MC_CKPT_DIR = METAGPT_ROOT / "data/minecraft/ckpt"  # Minecraft模型检查点存储目录
MC_LOG_DIR = METAGPT_ROOT / "logs"  # Minecraft日志存储目录

# 默认的热身参数（默认的warmup设置）
MC_DEFAULT_WARMUP = {
    "context": 15,  # 上下文热身时长
    "biome": 10,  # 生物群系热身时长
    "time": 15,  # 时间热身时长
    "nearby_blocks": 0,  # 附近方块的热身时长
    "other_blocks": 10,  # 其他方块的热身时长
    "nearby_entities": 5,  # 附近实体的热身时长
    "health": 15,  # 健康值的热身时长
    "hunger": 15,  # 饥饿值的热身时长
    "position": 0,  # 位置的热身时长
    "equipment": 0,  # 装备的热身时长
    "inventory": 0,  # 背包物品的热身时长
    "optional_inventory_items": 7,  # 可选背包物品的热身时长
    "chests": 0,  # 箱子的热身时长
    "completed_tasks": 0,  # 完成任务的热身时长
    "failed_tasks": 0,  # 失败任务的热身时长
}

# Minecraft代理的学习进度（curriculum）目标顺序
MC_CURRICULUM_OB = [
    "context",  # 上下文
    "biome",  # 生物群系
    "time",  # 时间
    "nearby_blocks",  # 附近方块
    "other_blocks",  # 其他方块
    "nearby_entities",  # 附近实体
    "health",  # 健康值
    "hunger",  # 饥饿值
    "position",  # 位置
    "equipment",  # 装备
    "inventory",  # 背包
    "chests",  # 箱子
    "completed_tasks",  # 完成的任务
    "failed_tasks",  # 失败的任务
]

# 核心背包物品列表，仅显示这些物品在背包中，直到可选物品达到热身要求
MC_CORE_INVENTORY_ITEMS = r".*_log|.*_planks|stick|crafting_table|furnace" \
                          r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe"  # curriculum_agent: 在背包中