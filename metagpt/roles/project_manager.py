#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 15:04
@Author  : alexanderwu
@File    : project_manager.py
"""
from metagpt.actions import WriteTasks
from metagpt.actions.design_api import WriteDesign
from metagpt.roles.di.role_zero import RoleZero


class ProjectManager(RoleZero):
    """
    表示项目经理角色，负责监督项目执行和团队效率。

    属性：
        name (str): 项目经理的名字。
        profile (str): 角色简介，默认为 '项目经理'。
        goal (str): 项目经理的目标。
        constraints (str): 项目经理的约束或限制。
    """

    name: str = "Eve"  # 项目经理的名字
    profile: str = "Project Manager"  # 角色简介
    goal: str = (
        "根据产品需求文档（PRD）/技术设计分解任务，生成任务列表，并分析任务依赖关系，"
        "从先决模块开始执行"
    )  # 项目经理的目标
    constraints: str = "使用与用户需求相同的语言"  # 角色约束

    instruction: str = """使用 WriteTasks 工具编写项目任务列表"""  # 使用 WriteTasks 工具编写任务列表
    max_react_loop: int = 1  # FIXME: 读取和编辑文件需要更多步骤，稍后考虑
    tools: list[str] = ["Editor:write,read,similarity_search", "RoleZero", "WriteTasks"]  # 项目经理可用的工具

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # 调用父类初始化
        # NOTE: 以下初始化设置仅在 self.use_fixed_sop 设置为 True 时生效
        self.enable_memory = False  # 禁用记忆功能
        self.set_actions([WriteTasks])  # 设置项目经理的行动为编写任务列表
        self._watch([WriteDesign])  # 监视写设计文档的操作

    def _update_tool_execution(self):
        wt = WriteTasks()  # 创建 WriteTasks 实例
        # 更新工具执行映射，指定 WriteTasks 工具的运行方法
        self.tool_execution_map.update(
            {
                "WriteTasks.run": wt.run,  # 将 WriteTasks 的 run 方法映射为工具执行方法
                "WriteTasks": wt.run,  # 别名，映射为相同的方法
            }
        )
