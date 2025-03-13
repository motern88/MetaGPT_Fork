#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/14 15:28
@Author  : alexanderwu
@File    : project_management_an.py
"""
from typing import List, Optional

from metagpt.actions.action_node import ActionNode

# 定义所需的Python包
REQUIRED_PACKAGES = ActionNode(
    key="Required packages",  # 包的关键字
    expected_type=Optional[List[str]],  # 期望的类型为可选的字符串列表
    instruction="Provide required packages The response language should correspond to the context and requirements.",  # 指导说明
    example=["flask==1.1.2", "bcrypt==3.2.0"],  # 示例包
)

# 定义所需的其他语言的第三方包
REQUIRED_OTHER_LANGUAGE_PACKAGES = ActionNode(
    key="Required Other language third-party packages",  # 关键字
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="List down the required packages for languages other than Python.",  # 指导说明
    example=["No third-party dependencies required"],  # 示例内容
)

# 逻辑分析：列出需实现的类/方法/函数及依赖关系分析
LOGIC_ANALYSIS = ActionNode(
    key="Logic Analysis",  # 关键字
    expected_type=List[List[str]],  # 期望的类型为字符串列表的列表
    instruction="Provide a list of files with the classes/methods/functions to be implemented, "
                "including dependency analysis and imports."
                "Ensure consistency between System Design and Logic Analysis; the files must match exactly. "
                "If the file is written in Vue or React, use Tailwind CSS for styling.",  # 指导说明
    example=[  # 示例内容
        ["game.py", "Contains Game class and ... functions"],
        ["main.py", "Contains main function, from game import Game"],
    ],
)

# 细化的逻辑分析：更新和扩展逻辑分析，包括文件依赖、影响等
REFINED_LOGIC_ANALYSIS = ActionNode(
    key="Refined Logic Analysis",  # 关键字
    expected_type=List[List[str]],  # 期望的类型为字符串列表的列表
    instruction="Review and refine the logic analysis by merging the Legacy Content and Incremental Content. "
                "Provide a comprehensive list of files with classes/methods/functions to be implemented or modified incrementally. "
                "Include dependency analysis, consider potential impacts on existing code, and document necessary imports.",  # 指导说明
    example=[  # 示例内容
        ["game.py", "Contains Game class and ... functions"],
        ["main.py", "Contains main function, from game import Game"],
        ["new_feature.py", "Introduces NewFeature class and related functions"],
        ["utils.py", "Modifies existing utility functions to support incremental changes"],
    ],
)

# 任务列表：列出任务并按照依赖顺序排列
TASK_LIST = ActionNode(
    key="Task list",  # 关键字
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Break down the tasks into a list of filenames, prioritized by dependency order.",  # 指导说明
    example=["game.py", "main.py"],  # 示例内容
)

# 细化的任务列表：更新和优化合并后的任务列表，确保任务的合理排序
REFINED_TASK_LIST = ActionNode(
    key="Refined Task list",  # 关键字
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Review and refine the combined task list after the merger of Legacy Content and Incremental Content, "
                "and consistent with Refined File List. Ensure that tasks are organized in a logical and prioritized order, "
                "considering dependencies for a streamlined and efficient development process.",  # 指导说明
    example=["new_feature.py", "utils", "game.py", "main.py"],  # 示例内容
)

# 完整的API规格：使用OpenAPI 3.0描述所有前后端可能使用的API
FULL_API_SPEC = ActionNode(
    key="Full API spec",  # 关键字
    expected_type=str,  # 期望的类型为字符串
    instruction="Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end "
                "and back-end communication is not required, leave it blank.",  # 指导说明
    example="openapi: 3.0.0 ...",  # 示例内容
)

# 共享知识：描述项目中的共用函数或配置变量
SHARED_KNOWLEDGE = ActionNode(
    key="Shared Knowledge",  # 关键字
    expected_type=str,  # 期望的类型为字符串
    instruction="Detail any shared knowledge, like common utility functions or configuration variables.",  # 指导说明
    example="`game.py` contains functions shared across the project.",  # 示例内容
)

# 细化的共享知识：更新和扩展共享知识，反映新引入的元素
REFINED_SHARED_KNOWLEDGE = ActionNode(
    key="Refined Shared Knowledge",  # 关键字
    expected_type=str,  # 期望的类型为字符串
    instruction="Update and expand shared knowledge to reflect any new elements introduced. This includes common "
                "utility functions, configuration variables for team collaboration. Retain content that is not related to "
                "incremental development but important for consistency and clarity.",  # 指导说明
    example="`new_module.py` enhances shared utility functions for improved code reusability and collaboration.",  # 示例内容
)

# 不清楚的项目管理事项：提到项目管理中的任何不明确问题并尝试澄清
ANYTHING_UNCLEAR_PM = ActionNode(
    key="Anything UNCLEAR",  # 关键字
    expected_type=str,  # 期望的类型为字符串
    instruction="Mention any unclear aspects in the project management context and try to clarify them.",  # 指导说明
    example="Clarification needed on how to start and initialize third-party libraries.",  # 示例内容
)

# 定义所有节点
NODES = [
    REQUIRED_PACKAGES,
    REQUIRED_OTHER_LANGUAGE_PACKAGES,
    LOGIC_ANALYSIS,
    TASK_LIST,
    FULL_API_SPEC,
    SHARED_KNOWLEDGE,
    ANYTHING_UNCLEAR_PM,
]

# 细化的节点：包含更多细化后的节点
REFINED_NODES = [
    REQUIRED_PACKAGES,
    REQUIRED_OTHER_LANGUAGE_PACKAGES,
    REFINED_LOGIC_ANALYSIS,
    REFINED_TASK_LIST,
    FULL_API_SPEC,
    REFINED_SHARED_KNOWLEDGE,
    ANYTHING_UNCLEAR_PM,
]

# 项目管理节点：包括所有任务管理相关的节点
PM_NODE = ActionNode.from_children("PM_NODE", NODES)

# 细化的项目管理节点：包括更多细化后的任务管理相关的节点
REFINED_PM_NODE = ActionNode.from_children("REFINED_PM_NODE", REFINED_NODES)
