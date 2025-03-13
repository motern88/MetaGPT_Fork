#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : architect.py
"""
from pydantic import Field

from metagpt.actions.design_api import WriteDesign
from metagpt.actions.write_prd import WritePRD
from metagpt.prompts.di.architect import ARCHITECT_EXAMPLE, ARCHITECT_INSTRUCTION
from metagpt.roles.di.role_zero import RoleZero
from metagpt.tools.libs.terminal import Terminal


class Architect(RoleZero):
    """
    代表软件开发过程中架构师角色。

    属性：
        name (str): 架构师的姓名。
        profile (str): 角色简介，默认为 'Architect'。
        goal (str): 架构师的主要目标或责任。
        constraints (str): 架构师的约束条件或指导原则。
    """

    # 架构师的姓名
    name: str = "Bob"
    # 角色简介，默认为 "Architect"
    profile: str = "Architect"
    # 架构师的目标：设计一个简洁、可用且完整的软件系统，输出系统设计
    goal: str = "design a concise, usable, complete software system. output the system design."
    # 约束条件：确保架构简洁，并使用合适的开源库，使用与用户需求相同的编程语言
    constraints: str = (
        "make sure the architecture is simple enough and use  appropriate open source "
        "libraries. Use same language as user requirement"
    )
    # 初始化终端对象（用于执行命令）
    terminal: Terminal = Field(default_factory=Terminal, exclude=True)
    # 初始化指令
    instruction: str = ARCHITECT_INSTRUCTION
    # 工具列表，包含可用的工具（如编辑器、终端命令等）
    tools: list[str] = [
        "Editor:write,read,similarity_search",  # 编辑器相关操作
        "RoleZero",  # RoleZero 类
        "Terminal:run_command",  # 终端命令执行
    ]

    # 构造函数，初始化架构师角色的相关设置
    def __init__(self, **kwargs) -> None:
        # 调用父类构造函数进行初始化
        super().__init__(**kwargs)

        # NOTE: 以下初始化设置仅在 self.use_fixed_sop 设置为 True 时生效
        self.enable_memory = False  # 禁用内存功能（可能不需要记住上下文）

        # 初始化架构师角色特定的动作
        self.set_actions([WriteDesign])  # 设置该角色的可执行动作为 'WriteDesign'

        # 设置架构师应该关注的事件或动作
        self._watch({WritePRD})  # 关注 'WritePRD' 事件

    # 获取架构师角色的经验示例
    def _retrieve_experience(self) -> str:
        return ARCHITECT_EXAMPLE  # 返回一个架构师的示例经验

    # 更新工具执行映射
    def _update_tool_execution(self):
        # 将 'Terminal.run_command' 映射到终端命令执行方法
        self.tool_execution_map.update({"Terminal.run_command": self.terminal.run_command})