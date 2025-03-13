#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/12 22:24
@Author  : alexanderwu
@File    : design_api_an.py
"""
from typing import List, Optional

from metagpt.actions.action_node import ActionNode
from metagpt.utils.mermaid import MMC1, MMC2

# 定义“实施方法”的 ActionNode
IMPLEMENTATION_APPROACH = ActionNode(
    key="Implementation approach",  # 节点的关键字
    expected_type=str,  # 期望的类型是字符串
    instruction="分析需求中的难点，选择合适的开源框架。",  # 对该节点的描述性指令
    example="我们将会...",  # 示例
)

# 定义“精炼实施方法”的 ActionNode
REFINED_IMPLEMENTATION_APPROACH = ActionNode(
    key="Refined Implementation Approach",  # 节点的关键字
    expected_type=str,  # 期望的类型是字符串
    instruction="更新并扩展原始的实施方法，以反映由于增量开发带来的挑战和需求变化。"
    "概述实施过程中的步骤并提供详细的策略。",  # 对该节点的描述性指令
    example="我们将优化...",  # 示例
)

# 定义“项目名称”的 ActionNode
PROJECT_NAME = ActionNode(
    key="Project name", expected_type=str, instruction="项目名称（用下划线分隔）", example="game_2048"
)

# 定义“文件列表”的 ActionNode
FILE_LIST = ActionNode(
    key="File list",  # 节点的关键字
    expected_type=List[str],  # 期望的类型是字符串列表
    instruction="仅需相对路径。根据编程语言指定正确的入口文件：JavaScript 使用 main.js，Python 使用 main.py，"
    "其他语言也相应指定。",  # 对该节点的描述性指令
    example=["a.js", "b.py", "c.css", "d.html"],  # 示例
)

# 定义“精炼文件列表”的 ActionNode
REFINED_FILE_LIST = ActionNode(
    key="Refined File list",  # 节点的关键字
    expected_type=List[str],  # 期望的类型是字符串列表
    instruction="更新并扩展原始文件列表，仅包含相对路径。最多可以添加 2 个文件。"
    "确保精炼后的文件列表反映了项目结构的变化。",  # 对该节点的描述性指令
    example=["main.py", "game.py", "new_feature.py"],  # 示例
)

# 可选项：由于非 Python 项目类图的成功复现较低，故该部分为可选
DATA_STRUCTURES_AND_INTERFACES = ActionNode(
    key="Data structures and interfaces",  # 节点的关键字
    expected_type=Optional[str],  # 期望的类型是可选的字符串
    instruction="使用 mermaid 类图代码语法，包含类、方法（__init__ 等）和函数，并且带有类型注解，"
    "清楚地标明类之间的关系，遵循 PEP8 标准。数据结构应非常详细，API 设计应全面且完整。",  # 对该节点的描述性指令
    example=MMC1,  # 示例
)

# 定义“精炼数据结构和接口”的 ActionNode
REFINED_DATA_STRUCTURES_AND_INTERFACES = ActionNode(
    key="Refined Data structures and interfaces",  # 节点的关键字
    expected_type=str,  # 期望的类型是字符串
    instruction="更新并扩展现有的 mermaid 类图代码语法，纳入新的类、方法（包括 __init__）和函数，"
    "并附上精准的类型注解。明确类之间的新增关系，确保代码清晰且遵循 PEP8 标准。"
    "保留与增量开发无关但对一致性和清晰性重要的内容。",  # 对该节点的描述性指令
    example=MMC1,  # 示例
)

# 定义“程序调用流程”的 ActionNode
PROGRAM_CALL_FLOW = ActionNode(
    key="Program call flow",  # 节点的关键字
    expected_type=Optional[str],  # 期望的类型是可选的字符串
    instruction="使用 sequenceDiagram 代码语法，全面且详细，准确覆盖每个对象的 CRUD 和初始化，"
    "确保语法正确，且准确使用已定义的类和 API。",  # 对该节点的描述性指令
    example=MMC2,  # 示例
)

# 定义“精炼程序调用流程”的 ActionNode
REFINED_PROGRAM_CALL_FLOW = ActionNode(
    key="Refined Program call flow",  # 节点的关键字
    expected_type=str,  # 期望的类型是字符串
    instruction="扩展现有的 sequenceDiagram 代码语法，详细说明每个对象的 CRUD 和初始化，"
    "确保语法正确，并准确反映增量开发带来的变化。保留与增量开发无关但对一致性和清晰性重要的内容。",  # 对该节点的描述性指令
    example=MMC2,  # 示例
)

# 定义“任何不清楚的地方”的 ActionNode
ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",  # 节点的关键字
    expected_type=str,  # 期望的类型是字符串
    instruction="提及不清楚的项目方面，然后尝试澄清它。",  # 对该节点的描述性指令
    example="需要澄清关于第三方 API 集成的内容，...",  # 示例
)

# 定义包含所有节点的设计 API 节点
NODES = [
    IMPLEMENTATION_APPROACH,  # 实施方法
    # PROJECT_NAME,  # 项目名称
    FILE_LIST,  # 文件列表
    DATA_STRUCTURES_AND_INTERFACES,  # 数据结构和接口
    PROGRAM_CALL_FLOW,  # 程序调用流程
    ANYTHING_UNCLEAR,  # 不清楚的地方
]

# 定义包含精炼节点的设计 API 节点
REFINED_NODES = [
    REFINED_IMPLEMENTATION_APPROACH,  # 精炼实施方法
    REFINED_FILE_LIST,  # 精炼文件列表
    REFINED_DATA_STRUCTURES_AND_INTERFACES,  # 精炼数据结构和接口
    REFINED_PROGRAM_CALL_FLOW,  # 精炼程序调用流程
    ANYTHING_UNCLEAR,  # 不清楚的地方
]

# 创建设计 API 节点
DESIGN_API_NODE = ActionNode.from_children("DesignAPI", NODES)

# 创建精炼设计 API 节点
REFINED_DESIGN_NODE = ActionNode.from_children("RefinedDesignAPI", REFINED_NODES)