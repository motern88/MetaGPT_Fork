#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : skill_loader.py
@Desc    : Skill YAML Configuration Loader.
"""
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from metagpt.context import Context
from metagpt.utils.common import aread


class Example(BaseModel):
    """示例类，包含一个问题（ask）和对应的答案（answer）。"""
    ask: str
    answer: str


class Returns(BaseModel):
    """返回值类，定义返回数据的类型和格式。"""
    type: str
    format: Optional[str] = None


class Parameter(BaseModel):
    """参数类，描述技能所需的参数类型及其说明。"""
    type: str
    description: str = None


class Skill(BaseModel):
    """技能类，描述一个具体的技能，包括名称、描述、ID、前提条件、参数、示例及返回值类型。"""
    name: str  # 技能名称
    description: str = None  # 技能描述
    id: str = None  # 技能唯一标识
    x_prerequisite: Dict = Field(default=None, alias="x-prerequisite")  # 技能的前提条件
    parameters: Dict[str, Parameter] = None  # 技能所需的参数
    examples: List[Example]  # 技能示例
    returns: Returns  # 技能返回值类型

    @property
    def arguments(self) -> Dict:
        """获取技能参数的描述信息。

        :return: 返回参数名称及其描述的字典
        """
        if not self.parameters:
            return {}
        ret = {}
        for k, v in self.parameters.items():
            ret[k] = v.description if v.description else ""
        return ret


class Entity(BaseModel):
    """实体类，包含实体名称和其拥有的技能列表。"""
    name: str = None  # 实体名称
    skills: List[Skill]  # 该实体拥有的技能列表


class Components(BaseModel):
    """组件类（目前为空，预留扩展）。"""
    pass


class SkillsDeclaration(BaseModel):
    """技能声明类，包含技能 API 版本、实体列表及可选的组件信息。"""
    skillapi: str  # 技能 API 版本
    entities: Dict[str, Entity]  # 所有实体的映射
    components: Components = None  # 组件信息（可选）

    @staticmethod
    async def load(skill_yaml_file_name: Path = None) -> "SkillsDeclaration":
        """异步加载技能声明文件。

        :param skill_yaml_file_name: 技能声明 YAML 文件的路径
        :return: 解析后的 SkillsDeclaration 对象
        """
        if not skill_yaml_file_name:
            skill_yaml_file_name = Path(__file__).parent.parent.parent / "docs/.well-known/skills.yaml"
        data = await aread(filename=skill_yaml_file_name)  # 读取 YAML 文件
        skill_data = yaml.safe_load(data)  # 解析 YAML 数据
        return SkillsDeclaration(**skill_data)

    def get_skill_list(self, entity_name: str = "Assistant", context: Context = None) -> Dict:
        """根据技能描述返回技能名称列表。

        :param entity_name: 目标实体的名称，默认为 "Assistant"
        :param context: 上下文对象（包含代理技能）
        :return: 以技能描述为键，技能名称为值的字典
        """
        entity = self.entities.get(entity_name)
        if not entity:
            return {}

        # 获取代理可用的技能列表
        ctx = context or Context()
        agent_skills = ctx.kwargs.agent_skills
        if not agent_skills:
            return {}

        class _AgentSkill(BaseModel):
            """代理技能类，仅包含技能名称。"""
            name: str

        # 获取代理允许使用的技能名称
        names = [_AgentSkill(**i).name for i in agent_skills]
        return {s.description: s.name for s in entity.skills if s.name in names}

    def get_skill(self, name, entity_name: str = "Assistant") -> Skill:
        """根据技能名称获取对应的技能对象。

        :param name: 技能名称
        :param entity_name: 目标实体的名称，默认为 "Assistant"
        :return: 目标技能对象，若不存在则返回 None
        """
        entity = self.entities.get(entity_name)
        if not entity:
            return None
        for sk in entity.skills:
            if sk.name == name:
                return sk
