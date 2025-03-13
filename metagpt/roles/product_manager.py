#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : product_manager.py
@Modified By: liushaojie, 2024/10/17.
"""
from metagpt.actions import UserRequirement, WritePRD
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.actions.search_enhanced_qa import SearchEnhancedQA
from metagpt.prompts.product_manager import PRODUCT_MANAGER_INSTRUCTION
from metagpt.roles.di.role_zero import RoleZero
from metagpt.roles.role import RoleReactMode
from metagpt.tools.libs.browser import Browser
from metagpt.tools.libs.editor import Editor
from metagpt.utils.common import any_to_name, any_to_str, tool2name
from metagpt.utils.git_repository import GitRepository


class ProductManager(RoleZero):
    """
    表示一个产品经理角色，负责产品开发和管理。

    属性：
        name (str): 产品经理的名字。
        profile (str): 角色简介，默认为 '产品经理'。
        goal (str): 产品经理的目标。
        constraints (str): 产品经理的约束或限制。
    """

    name: str = "Alice"  # 产品经理的名字
    profile: str = "Product Manager"  # 角色简介
    goal: str = "创建产品需求文档或进行市场调研/竞争产品调研。"  # 角色目标
    constraints: str = "使用与用户需求相同的语言，以确保无缝沟通"  # 角色约束
    instruction: str = PRODUCT_MANAGER_INSTRUCTION  # 产品经理的指令，默认为定义的常量
    tools: list[str] = ["RoleZero", Browser.__name__, Editor.__name__, SearchEnhancedQA.__name__]  # 产品经理可用的工具

    todo_action: str = any_to_name(WritePRD)  # 待办事项动作，默认为 'WritePRD'（撰写产品需求文档）

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # 调用父类初始化
        if self.use_fixed_sop:
            # 如果使用固定的标准操作程序（SOP），禁用记忆功能并设置特定的行动
            self.enable_memory = False
            self.set_actions([PrepareDocuments(send_to=any_to_str(self)), WritePRD])  # 设置行动为准备文档和撰写产品需求文档
            self._watch([UserRequirement, PrepareDocuments])  # 监视用户需求和准备文档的变化
            self.rc.react_mode = RoleReactMode.BY_ORDER  # 设置反应模式为按顺序反应

    def _update_tool_execution(self):
        wp = WritePRD()  # 创建 WritePRD 实例
        # 更新工具执行映射，指定 WritePRD 工具的运行方法
        self.tool_execution_map.update(tool2name(WritePRD, ["run"], wp.run))

    async def _think(self) -> bool:
        """决定接下来要做什么"""
        if not self.use_fixed_sop:
            return await super()._think()  # 如果没有使用固定 SOP，调用父类的 _think 方法

        # 如果当前项目路径是一个 Git 仓库，并且 Git 尚未重新初始化
        if GitRepository.is_git_dir(self.config.project_path) and not self.config.git_reinit:
            self._set_state(1)  # 设置状态为 1
        else:
            self._set_state(0)  # 否则设置状态为 0
            self.config.git_reinit = False  # 禁止 Git 重新初始化
            self.todo_action = any_to_name(WritePRD)  # 设置待办事项为撰写产品需求文档
        return bool(self.rc.todo)  # 如果有待办事项，返回 True
