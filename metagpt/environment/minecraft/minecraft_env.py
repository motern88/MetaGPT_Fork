#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : MG Minecraft Env
#           refs to `voyager voyager.py`

import json
import re
import time
from typing import Any, Iterable

from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import ConfigDict, Field

from metagpt.config2 import Config
from metagpt.environment.base_env import Environment
from metagpt.environment.minecraft.const import MC_CKPT_DIR
from metagpt.environment.minecraft.minecraft_ext_env import MinecraftExtEnv
from metagpt.logs import logger
from metagpt.utils.common import load_mc_skills_code, read_json_file, write_json_file


class MinecraftEnv(MinecraftExtEnv, Environment):
    """MinecraftEnv，包含不同角色间共享的缓存和信息"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event: dict[str, Any] = Field(default_factory=dict)  # 当前事件
    current_task: str = Field(default="Mine 1 wood log")  # 当前任务
    task_execution_time: float = Field(default=float)  # 任务执行时间
    context: str = Field(default="You can mine one of oak, birch, spruce, jungle, acacia, dark oak, or mangrove logs.")  # 任务背景信息
    code: str = Field(default="")  # 当前代码
    program_code: str = Field(default="")  # 任务使用的程序代码（位于 skill/code/*.js）
    program_name: str = Field(default="")  # 程序名称
    critique: str = Field(default="")  # 任务批评意见
    skills: dict = Field(default_factory=dict)  # 存储技能的字典
    retrieve_skills: list[str] = Field(default_factory=list)  # 获取的技能列表
    event_summary: str = Field(default="")  # 事件总结

    qa_cache: dict[str, str] = Field(default_factory=dict)  # 存储问答缓存
    completed_tasks: list[str] = Field(default_factory=list)  # 已完成任务列表
    failed_tasks: list[str] = Field(default_factory=list)  # 失败任务列表

    skill_desp: str = Field(default="")  # 技能描述

    chest_memory: dict[str, Any] = Field(default_factory=dict)  # 存储箱子相关的记忆（例如：{'(1344, 64, 1381)': 'Unknown'}）
    chest_observation: str = Field(default="")  # 存储箱子的观察结果（例如："Chests: None\n\n"）

    runtime_status: bool = False  # 当前执行状态（成功或失败）

    vectordb: ChromaVectorStore = Field(default_factory=ChromaVectorStore)  # 存储技能的向量数据库

    qa_cache_questions_vectordb: ChromaVectorStore = Field(default_factory=ChromaVectorStore)  # 存储问答缓存问题的向量数据库

    @property
    def progress(self):
        # 返回已完成任务的数量
        return len(self.completed_tasks)

    @property
    def programs(self):
        programs = ""
        if self.code == "":
            return programs  # 如果没有代码，则返回空
        # 遍历技能并将代码添加到程序中
        for skill_name, entry in self.skills.items():
            programs += f"{entry['code']}\n\n"
        # 加载额外的技能代码
        for primitives in load_mc_skills_code():
            programs += f"{primitives}\n\n"
        return programs

    def set_mc_port(self, mc_port):
        # 设置 Minecraft 端口
        super().set_mc_port(mc_port)
        self.set_mc_resume()

    def set_mc_resume(self):
        # 设置 Minecraft 恢复状态
        self.qa_cache_questions_vectordb = ChromaVectorStore(
            collection_name="qa_cache_questions_vectordb",
            persist_dir=f"{MC_CKPT_DIR}/curriculum/vectordb",
        )

        self.vectordb = ChromaVectorStore(
            collection_name="skill_vectordb",
            persist_dir=f"{MC_CKPT_DIR}/skill/vectordb",
        )

        if Config.default().resume:
            logger.info(f"从 {MC_CKPT_DIR}/action 加载 Action Developer")
            self.chest_memory = read_json_file(f"{MC_CKPT_DIR}/action/chest_memory.json")

            logger.info(f"从 {MC_CKPT_DIR}/curriculum 加载 Curriculum Agent")
            self.completed_tasks = read_json_file(f"{MC_CKPT_DIR}/curriculum/completed_tasks.json")
            self.failed_tasks = read_json_file(f"{MC_CKPT_DIR}/curriculum/failed_tasks.json")

            logger.info(f"从 {MC_CKPT_DIR}/skill 加载 Skill Manager")
            self.skills = read_json_file(f"{MC_CKPT_DIR}/skill/skills.json")

            logger.info(f"从 {MC_CKPT_DIR}/curriculum 加载 Qa Cache")
            self.qa_cache = read_json_file(f"{MC_CKPT_DIR}/curriculum/qa_cache.json")

            # 如果技能向量数据库为空，则初始化
            if self.vectordb._collection.count() == 0:
                logger.info(self.vectordb._collection.count())
                skill_desps = [skill["description"] for program_name, skill in self.skills.items()]
                program_names = [program_name for program_name, skill in self.skills.items()]
                metadatas = [{"name": program_name} for program_name in program_names]
                self.vectordb.add_texts(
                    texts=skill_desps,
                    ids=program_names,
                    metadatas=metadatas,
                )
                self.vectordb.persist()

            # 如果问答缓存向量数据库为空，则初始化
            logger.info(self.qa_cache_questions_vectordb._collection.count())
            if self.qa_cache_questions_vectordb._collection.count() == 0:
                questions = [question for question, answer in self.qa_cache.items()]
                self.qa_cache_questions_vectordb.add_texts(texts=questions)
                self.qa_cache_questions_vectordb.persist()

                logger.info(
                    f"INIT_CHECK: 当前 vectordb 中有 {self.vectordb._collection.count()} 个技能，skills.json 中有 {len(self.skills)} 个技能。"
                )
                assert self.vectordb._collection.count() == len(self.skills), (
                    f"技能管理器的 vectordb 与 skills.json 不一致。\n"
                    f"vectordb 中有 {self.vectordb._collection.count()} 个技能，而 skills.json 中有 {len(self.skills)} 个技能。\n"
                    f"初始化时是否设置了 resume=False？\n"
                    f"您可能需要手动删除 vectordb 目录以重新运行。"
                )

                logger.info(
                    f"INIT_CHECK: 当前 qa_cache_questions_vectordb 中有 {self.qa_cache_questions_vectordb._collection.count()} 个问题，qa_cache.json 中有 {len(self.qa_cache)} 个问题。"
                )
                assert self.qa_cache_questions_vectordb._collection.count() == len(self.qa_cache), (
                    f"课程代理的 qa 缓存问题 vectordb 与 qa_cache.json 不一致。\n"
                    f"vectordb 中有 {self.qa_cache_questions_vectordb._collection.count()} 个问题，而 qa_cache.json 中有 {len(self.qa_cache)} 个问题。\n"
                    f"初始化时是否设置了 resume=False？\n"
                    f"您可能需要手动删除 qa 缓存问题的 vectordb 目录以重新运行。"
                )

    def register_roles(self, roles: Iterable["Minecraft"]):
        # 注册角色并为每个角色设置内存
        for role in roles:
            role.set_memory(self)

    def update_event(self, event: dict):
        # 更新事件
        if self.event == event:
            return
        self.event = event
        self.update_chest_memory(event)
        self.update_chest_observation()

    def update_task(self, task: str):
        # 更新当前任务
        self.current_task = task

    def update_context(self, context: str):
        # 更新上下文
        self.context = context

    def update_program_code(self, program_code: str):
        # 更新程序代码
        self.program_code = program_code

    def update_code(self, code: str):
        # 更新代码
        self.code = code

    def update_program_name(self, program_name: str):
        # 更新程序名称
        self.program_name = program_name

    def update_critique(self, critique: str):
        # 更新批评意见
        self.critique = critique

    def append_skill(self, skill: dict):
        # 添加技能到技能列表
        self.skills[self.program_name] = skill

    def update_retrieve_skills(self, retrieve_skills: list):
        # 更新获取的技能列表
        self.retrieve_skills = retrieve_skills

    def update_skill_desp(self, skill_desp: str):
        # 更新技能描述
        self.skill_desp = skill_desp

    async def update_qa_cache(self, qa_cache: dict):
        # 异步更新问答缓存
        self.qa_cache = qa_cache

    def update_chest_memory(self, events: dict):
        """
        输入: events: 字典
        结果: 更新 chest_memory 并保存到 json 文件
        """
        nearbyChests = events[-1][1]["nearbyChests"]
        for position, chest in nearbyChests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    logger.info(f"Action Developer 移除箱子 {position}: {chest}")
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    logger.info(f"Action Developer 保存箱子 {position}: {chest}")
                    self.chest_memory[position] = chest

        write_json_file(f"{MC_CKPT_DIR}/action/chest_memory.json", self.chest_memory)

    def update_chest_observation(self):
        """
        更新 chest_memory 到 chest_observation。
        参考: https://github.com/MineDojo/Voyager/blob/main/voyager/agents/action.py
        """

        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: 空")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: 未知物品")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            self.chest_observation = f"箱子:\n{chests}\n\n"
        else:
            self.chest_observation = "箱子: 无\n\n"

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = r"I cannot make \w+ because there is no crafting table nearby"
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                self.event_summary = re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                self.event_summary = "附近的工作台"
            elif re.match(mine_pattern, message):
                self.event_summary = re.match(mine_pattern, message).groups()[0]
            else:
                self.event_summary = ""
            return self.event_summary

        chatlog = set()
        for event_type, event in events:
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        self.event_summary = "我还需要 " + ", ".join(chatlog) + "。" if chatlog else ""

    def reset_block_info(self):
        # 撤销上一步的所有放置事件
        pass

    def update_exploration_progress(self, success: bool):
        """
        将任务拆分为已完成的任务或失败的任务
        参数: info = {
            "task": self.task,
            "success": success,
            "conversations": self.conversations,
        }
        """
        self.runtime_status = success
        task = self.current_task
        if task.startswith("Deposit useless items into the chest at"):
            return
        if success:
            logger.info(f"完成任务 {task}.")
            self.completed_tasks.append(task)
        else:
            logger.info(f"任务 {task} 完成失败。跳过到下一个任务。")
            self.failed_tasks.append(task)
            # 如果任务失败，更新事件！
            # 撤销上一步的所有放置事件
            blocks = []
            positions = []
            for event_type, event in self.event:
                if event_type == "onSave" and event["onSave"].endswith("_placed"):
                    block = event["onSave"].split("_placed")[0]
                    position = event["status"]["position"]
                    blocks.append(block)
                    positions.append(position)
            new_events = self._step(
                f"await givePlacedItemBack(bot, {json.dumps(blocks)}, {json.dumps(positions)})",
                programs=self.programs,
            )
            self.event[-1][1]["inventory"] = new_events[-1][1]["inventory"]
            self.event[-1][1]["voxels"] = new_events[-1][1]["voxels"]

        self.save_sorted_tasks()

    def save_sorted_tasks(self):
        updated_completed_tasks = []
        # 记录重复的失败任务
        updated_failed_tasks = self.failed_tasks
        # 去重但保留顺序
        for task in self.completed_tasks:
            if task not in updated_completed_tasks:
                updated_completed_tasks.append(task)

        # 从失败任务中移除已完成的任务
        for task in updated_completed_tasks:
            while task in updated_failed_tasks:
                updated_failed_tasks.remove(task)

        self.completed_tasks = updated_completed_tasks
        self.failed_tasks = updated_failed_tasks

        # 保存到 json 文件
        write_json_file(f"{MC_CKPT_DIR}/curriculum/completed_tasks.json", self.completed_tasks)
        write_json_file(f"{MC_CKPT_DIR}/curriculum/failed_tasks.json", self.failed_tasks)

    async def on_event_retrieve(self, *args):
        """
        获取 Minecraft 事件。

        返回:
            list: 一个包含 Minecraft 事件的列表。

            抛出:
                异常: 如果获取事件时出现问题。
        """
        try:
            self._reset(
                options={
                    "mode": "soft",
                    "wait_ticks": 20,
                }
            )
            # difficulty = "easy" if len(self.completed_tasks) > 15 else "peaceful"
            difficulty = "peaceful"

            events = self._step("bot.chat(`/time set ${getNextTime()}`);\n" + f"bot.chat('/difficulty {difficulty}');")
            self.update_event(events)
            return events
        except Exception as e:
            time.sleep(3)  # 等待 mineflayer 退出
            # 这里重置机器人状态
            events = self._reset(
                options={
                    "mode": "hard",
                    "wait_ticks": 20,
                    "inventory": self.event[-1][1]["inventory"],
                    "equipment": self.event[-1][1]["status"]["equipment"],
                    "position": self.event[-1][1]["status"]["position"],
                }
            )
            self.update_event(events)
            logger.error(f"获取 Minecraft 事件失败: {str(e)}")
            return events

    async def on_event_execute(self, *args):
        """
        执行 Minecraft 事件。

        该函数用于获取 Minecraft 环境中的事件。请检查 'voyager/env/bridge.py step()' 中的实现，
        以捕获游戏中生成的事件。

        返回:
            list: 一个包含 Minecraft 事件的列表。

            抛出:
                异常: 如果执行事件时出现问题。
        """
        try:
            events = self._step(
                code=self.code,
                programs=self.programs,
            )
            self.update_event(events)
            return events
        except Exception as e:
            time.sleep(3)  # 等待 mineflayer 退出
            # 这里重置机器人状态
            events = self._reset(
                options={
                    "mode": "hard",
                    "wait_ticks": 20,
                    "inventory": self.event[-1][1]["inventory"],
                    "equipment": self.event[-1][1]["status"]["equipment"],
                    "position": self.event[-1][1]["status"]["position"],
                }
            )
            self.update_event(events)
            logger.error(f"执行 Minecraft 事件失败: {str(e)}")
            return events