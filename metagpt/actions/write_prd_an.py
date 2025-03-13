#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/14 11:40
@Author  : alexanderwu
@File    : write_prd_an.py
"""
from typing import List, Union

from metagpt.actions.action_node import ActionNode

LANGUAGE = ActionNode(
    key="Language",  # 项目的语言
    expected_type=str,  # 期望的类型为字符串
    instruction="Provide the language used in the project, typically matching the user's requirement language.",  # 项目使用的语言，通常与用户需求语言匹配
    example="en_us",  # 示例：英文 (美国)
)

PROGRAMMING_LANGUAGE = ActionNode(
    key="Programming Language",  # 使用的编程语言
    expected_type=str,  # 期望的类型为字符串
    instruction="Mainstream programming language. If not specified in the requirements, use Vite, React, MUI, Tailwind CSS.",  # 常用的编程语言。如果在需求中没有指定，使用 Vite, React, MUI, Tailwind CSS
    example="Vite, React, MUI, Tailwind CSS",  # 示例：Vite, React, MUI, Tailwind CSS
)

ORIGINAL_REQUIREMENTS = ActionNode(
    key="Original Requirements",  # 用户的原始需求
    expected_type=str,  # 期望的类型为字符串
    instruction="Place the original user's requirements here.",  # 放置用户的原始需求
    example="Create a 2048 game",  # 示例：创建一个2048游戏
)

REFINED_REQUIREMENTS = ActionNode(
    key="Refined Requirements",  # 精炼后的需求
    expected_type=str,  # 期望的类型为字符串
    instruction="Place the New user's original requirements here.",  # 放置精炼后的用户需求
    example="Create a 2048 game with a new feature that ...",  # 示例：创建一个带有新功能的2048游戏
)

PROJECT_NAME = ActionNode(
    key="Project Name",  # 项目名称
    expected_type=str,  # 期望的类型为字符串
    instruction='According to the content of "Original Requirements," name the project using snake case style , '
    "like 'game_2048' or 'simple_crm.",  # 根据原始需求命名项目，采用蛇形命名风格（如：game_2048 或 simple_crm）
    example="game_2048",  # 示例：game_2048
)

PRODUCT_GOALS = ActionNode(
    key="Product Goals",  # 产品目标
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Provide up to three clear, orthogonal product goals.",  # 提供最多三个清晰且独立的产品目标
    example=["Create an engaging user experience", "Improve accessibility, be responsive", "More beautiful UI"],  # 示例：创建吸引用户的体验、提高可访问性、更加美观的UI
)

REFINED_PRODUCT_GOALS = ActionNode(
    key="Refined Product Goals",  # 精炼后的产品目标
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Update and expand the original product goals to reflect the evolving needs due to incremental "
    "development. Ensure that the refined goals align with the current project direction and contribute to its success.",  # 更新并扩展原始的产品目标，以反映增量开发带来的不断变化的需求
    example=[
        "Enhance user engagement through new features",  # 示例：通过新功能增强用户互动
        "Optimize performance for scalability",  # 示例：优化性能以实现可扩展性
        "Integrate innovative UI enhancements",  # 示例：整合创新的UI提升
    ],
)

USER_STORIES = ActionNode(
    key="User Stories",  # 用户故事
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Provide up to 3 to 5 scenario-based user stories.",  # 提供最多3到5个基于场景的用户故事
    example=[
        "As a player, I want to be able to choose difficulty levels",  # 示例：作为玩家，我想能够选择难度级别
        "As a player, I want to see my score after each game",  # 示例：作为玩家，我想在每局游戏后查看我的得分
        "As a player, I want to get restart button when I lose",  # 示例：作为玩家，我希望在失败时看到一个重新开始的按钮
        "As a player, I want to see beautiful UI that make me feel good",  # 示例：作为玩家，我希望看到让人愉悦的美观UI
        "As a player, I want to play game via mobile phone",  # 示例：作为玩家，我希望能通过手机玩游戏
    ],
)

REFINED_USER_STORIES = ActionNode(
    key="Refined User Stories",  # 精炼后的用户故事
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Update and expand the original scenario-based user stories to reflect the evolving needs due to "
    "incremental development. Ensure that the refined user stories capture incremental features and improvements.",  # 更新并扩展原始的基于场景的用户故事，以反映增量开发带来的不断变化的需求
    example=[
        "As a player, I want to choose difficulty levels to challenge my skills",  # 示例：作为玩家，我希望能选择难度级别，以挑战我的技能
        "As a player, I want a visually appealing score display after each game for a better gaming experience",  # 示例：作为玩家，我希望在每局游戏后看到一个美观的得分显示，以提供更好的游戏体验
        "As a player, I want a convenient restart button displayed when I lose to quickly start a new game",  # 示例：作为玩家，我希望在失败时看到一个方便的重新开始按钮，以便快速开始新的一局
        "As a player, I want an enhanced and aesthetically pleasing UI to elevate the overall gaming experience",  # 示例：作为玩家，我希望看到一个经过优化的、美观的UI，提升整体的游戏体验
        "As a player, I want the ability to play the game seamlessly on my mobile phone for on-the-go entertainment",  # 示例：作为玩家，我希望能无缝地在手机上玩游戏，随时随地享受娱乐
    ],
)

COMPETITIVE_ANALYSIS = ActionNode(
    key="Competitive Analysis",  # 竞争分析
    expected_type=List[str],  # 期望的类型为字符串列表
    instruction="Provide 5 to 7 competitive products.",  # 提供5到7个竞争产品
    example=[
        "2048 Game A: Simple interface, lacks responsive features",  # 示例：2048游戏A：简单的界面，缺乏响应式特性
        "play2048.co: Beautiful and responsive UI with my best score shown",  # 示例：play2048.co：美观且响应式的UI，显示我的最佳得分
        "2048game.com: Responsive UI with my best score shown, but many ads",  # 示例：2048game.com：响应式UI，显示我的最佳得分，但有很多广告
    ],
)

COMPETITIVE_QUADRANT_CHART = ActionNode(
    key="Competitive Quadrant Chart",  # 竞争象限图
    expected_type=str,  # 期望的类型为字符串
    instruction="Use mermaid quadrantChart syntax. Distribute scores evenly between 0 and 1",  # 使用mermaid quadrantChart语法，在0到1之间均匀分布分数
    example="""quadrantChart
    title "Reach and engagement of campaigns"
    x-axis "Low Reach" --> "High Reach"
    y-axis "Low Engagement" --> "High Engagement"
    quadrant-1 "We should expand"
    quadrant-2 "Need to promote"
    quadrant-3 "Re-evaluate"
    quadrant-4 "May be improved"
    "Campaign A": [0.3, 0.6]
    "Campaign B": [0.45, 0.23]
    "Campaign C": [0.57, 0.69]
    "Campaign D": [0.78, 0.34]
    "Campaign E": [0.40, 0.34]
    "Campaign F": [0.35, 0.78]
    "Our Target Product": [0.5, 0.6]""",  # 示例：竞争象限图
)

REQUIREMENT_ANALYSIS = ActionNode(
    key="Requirement Analysis",  # 需求分析
    expected_type=str,  # 期望的类型为字符串
    instruction="Provide a detailed analysis of the requirements.",  # 提供需求的详细分析
    example="",  # 示例：空
)

REFINED_REQUIREMENT_ANALYSIS = ActionNode(
    key="Refined Requirement Analysis",  # 精炼后的需求分析
    expected_type=Union[List[str], str],  # 期望的类型为字符串列表或字符串
    instruction="Review and refine the existing requirement analysis into a string list to align with the evolving needs of the project "
    "due to incremental development. Ensure the analysis comprehensively covers the new features and enhancements "
    "required for the refined project scope.",  # 审查并精炼现有的需求分析，将其转化为字符串列表，以适应增量开发带来的不断变化的需求
    example=["Require add ...", "Require modify ..."],  # 示例：需要增加...，需要修改...
)

REQUIREMENT_POOL = ActionNode(
    key="Requirement Pool",  # 需求池
    expected_type=List[List[str]],  # 期望的类型为字符串列表的列表
    instruction="List down the top-5 requirements with their priority (P0, P1, P2).",  # 列出前5个需求及其优先级（P0，P1，P2）
    example=[["P0", "The main code ..."], ["P0", "The game algorithm ..."]],  # 示例：[["P0", "主要代码..."], ["P0", "游戏算法..."]]
)

# 精炼需求池，列出最重要的 5 至 7 个需求，以及其优先级（P0, P1, P2）
REFINED_REQUIREMENT_POOL = ActionNode(
    key="Refined Requirement Pool",
    expected_type=List[List[str]],
    instruction="列出最重要的 5 至 7 个需求，并附上它们的优先级（P0, P1, P2）。 "
    "包括传统内容和增量内容，并保留与增量开发无关的内容。",
    example=[["P0", "主要代码..."], ["P0", "游戏算法..."]],
)

# UI设计草图，提供一个简单的UI元素、功能、样式和布局描述
UI_DESIGN_DRAFT = ActionNode(
    key="UI Design draft",
    expected_type=str,
    instruction="提供一个简要的UI元素、功能、样式和布局的描述。",
    example="基本功能描述，风格和布局简单。",
)

# 不清楚的部分，提到项目中不清楚的任何方面并尝试澄清
ANYTHING_UNCLEAR = ActionNode(
    key="Anything UNCLEAR",
    expected_type=str,
    instruction="提到项目中任何不清楚的地方，并尽量澄清。",
    example="目前项目的所有方面都很清楚。",
)

# 问题类型，回答BUG或需求。如果是bug修复，回答BUG，否则回答需求。
ISSUE_TYPE = ActionNode(
    key="issue_type",
    expected_type=str,
    instruction="回答BUG/REQUIREMENT。如果是bug修复，回答BUG，否则回答需求。",
    example="BUG",
)

# 是否相关，回答YES/NO。如果该需求与旧的PRD相关，回答YES，否则回答NO。
IS_RELATIVE = ActionNode(
    key="is_relative",
    expected_type=str,
    instruction="回答YES/NO。如果该需求与旧的PRD相关，回答YES，否则回答NO。",
    example="YES",
)

# 理由，解释从问题到答案的推理过程。
REASON = ActionNode(
    key="reason", expected_type=str, instruction="解释从问题到答案的推理过程", example="..."
)

# 定义了一组节点，用于生成 PRD（产品需求文档）
NODES = [
    LANGUAGE,  # 语言
    PROGRAMMING_LANGUAGE,  # 编程语言
    ORIGINAL_REQUIREMENTS,  # 原始需求
    PROJECT_NAME,  # 项目名称
    PRODUCT_GOALS,  # 产品目标
    USER_STORIES,  # 用户故事
    COMPETITIVE_ANALYSIS,  # 竞争分析
    COMPETITIVE_QUADRANT_CHART,  # 竞争四象限图
    REQUIREMENT_ANALYSIS,  # 需求分析
    REQUIREMENT_POOL,  # 需求池
    UI_DESIGN_DRAFT,  # UI设计草图
    ANYTHING_UNCLEAR,  # 不清楚的部分
]

# 定义了精炼后的节点集合，用于生成精炼后的PRD
REFINED_NODES = [
    LANGUAGE,  # 语言
    PROGRAMMING_LANGUAGE,  # 编程语言
    REFINED_REQUIREMENTS,  # 精炼需求
    PROJECT_NAME,  # 项目名称
    REFINED_PRODUCT_GOALS,  # 精炼产品目标
    REFINED_USER_STORIES,  # 精炼用户故事
    COMPETITIVE_ANALYSIS,  # 竞争分析
    COMPETITIVE_QUADRANT_CHART,  # 竞争四象限图
    REFINED_REQUIREMENT_ANALYSIS,  # 精炼需求分析
    REFINED_REQUIREMENT_POOL,  # 精炼需求池
    UI_DESIGN_DRAFT,  # UI设计草图
    ANYTHING_UNCLEAR,  # 不清楚的部分
]

# 根据定义的节点创建PRD和精炼PRD节点
WRITE_PRD_NODE = ActionNode.from_children("WritePRD", NODES)
REFINED_PRD_NODE = ActionNode.from_children("RefinedPRD", REFINED_NODES)

# 问题类型节点
WP_ISSUE_TYPE_NODE = ActionNode.from_children("WP_ISSUE_TYPE", [ISSUE_TYPE, REASON])

# 是否相关节点
WP_IS_RELATIVE_NODE = ActionNode.from_children("WP_IS_RELATIVE", [IS_RELATIVE, REASON])