#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/5 01:44
@Author  : alexanderwu
@File    : skill_manager.py
@Modified By: mashenquan, 2023/8/20. Remove useless `llm`
"""
from metagpt.actions import Action
from metagpt.const import PROMPT_PATH
from metagpt.document_store.chromadb_store import ChromaStore
from metagpt.logs import logger

Skill = Action


class SkillManager:
    """用于管理所有技能的类"""

    def __init__(self):
        self._store = ChromaStore("skill_manager")  # 创建用于存储技能的 ChromaStore
        self._skills: dict[str:Skill] = {}  # 存储技能的字典

    def add_skill(self, skill: Skill):
        """
        添加技能，将技能添加到技能池并存入可搜索的存储中。
        :param skill: Skill 实例，表示要添加的技能
        """
        self._skills[skill.name] = skill  # 在技能字典中添加技能
        self._store.add(skill.desc, {"name": skill.name, "desc": skill.desc}, skill.name)  # 将技能信息存入存储

    def del_skill(self, skill_name: str):
        """
        删除技能，从技能池和可搜索存储中移除该技能。
        :param skill_name: 需要删除的技能名称
        """
        self._skills.pop(skill_name)  # 从技能字典中删除技能
        self._store.delete(skill_name)  # 从存储中删除技能

    def get_skill(self, skill_name: str) -> Skill:
        """
        通过技能名称获取特定技能。
        :param skill_name: 技能名称
        :return: 该名称对应的技能对象，如果不存在则返回 None
        """
        return self._skills.get(skill_name)

    def retrieve_skill(self, desc: str, n_results: int = 2) -> list[Skill]:
        """
        通过搜索引擎获取相关技能。
        :param desc: 技能描述
        :param n_results: 需要返回的技能数量（默认返回 2 个）
        :return: 匹配到的多个技能
        """
        return self._store.search(desc, n_results=n_results)["ids"][0]

    def retrieve_skill_scored(self, desc: str, n_results: int = 2) -> dict:
        """
        通过搜索引擎获取相关技能，并返回匹配分数。
        :param desc: 技能描述
        :param n_results: 需要返回的技能数量（默认返回 2 个）
        :return: 一个包含技能及其匹配分数的字典
        """
        return self._store.search(desc, n_results=n_results)

    def generate_skill_desc(self, skill: Skill) -> str:
        """
        生成每个技能的描述性文本。
        :param skill: 需要生成描述的技能
        """
        path = PROMPT_PATH / "generate_skill.md"  # 读取技能描述的模板文件
        text = path.read_text()
        logger.info(text)  # 记录生成的文本

if __name__ == "__main__":
    manager = SkillManager()
    manager.generate_skill_desc(Action())  # 生成一个 Action 技能的描述
