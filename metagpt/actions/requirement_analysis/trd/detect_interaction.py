#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : detect_interaction.py
@Desc    : The implementation of Chapter 2.1.6 of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import general_after_log, to_markdown_code_block


@register_tool(include_functions=["run"])
class DetectInteraction(Action):
    """DetectInteraction 处理以下情况：
    1. 给定用户需求的自然语言文本，从中识别交互事件及其参与者。
    """

    @retry(
        wait=wait_random_exponential(min=1, max=20),  # 在 1 到 20 秒之间随机指数回退等待时间
        stop=stop_after_attempt(6),  # 最多重试 6 次
        after=general_after_log(logger),  # 失败后记录日志
    )
    async def run(
        self,
        *,
        user_requirements: str,  # 用户需求的自然语言文本
        use_case_actors: str,  # 用例中的参与者描述
        legacy_interaction_events: str,  # 先前识别出的交互事件版本
        evaluation_conclusion: str,  # 外部评估关于交互事件的结论
    ) -> str:
        """
        识别用户需求中的交互事件及其参与者。

        参数:
            user_requirements (str): 用户需求的自然语言描述。
            use_case_actors (str): 用例中涉及的参与者信息。
            legacy_interaction_events (str): 之前版本中识别出的交互事件。
            evaluation_conclusion (str): 外部评估对交互事件的反馈和结论。

        返回:
            str: 总结交互事件及其参与者的信息。

        示例:
            >>> detect_interaction = DetectInteraction()
            >>> user_requirements = "用户需求 1. ..."
            >>> use_case_actors = "- 角色: 游戏玩家;\\n- 系统: 贪吃蛇游戏; \\n- 外部系统: 游戏中心;"
            >>> previous_version_interaction_events = "['interaction ...', ...]"
            >>> evaluation_conclusion = "问题: ..."
            >>> interaction_events = await detect_interaction.run(
            >>>    user_requirements=user_requirements,
            >>>    use_case_actors=use_case_actors,
            >>>    legacy_interaction_events=previous_version_interaction_events,
            >>>    evaluation_conclusion=evaluation_conclusion,
            >>> )
            >>> print(interaction_events)
            "['interaction ...', ...]"
        """
        msg = PROMPT.format(
            use_case_actors=use_case_actors,
            original_user_requirements=to_markdown_code_block(val=user_requirements),
            previous_version_of_interaction_events=legacy_interaction_events,
            the_evaluation_conclusion_of_previous_version_of_trd=evaluation_conclusion,
        )
        return await self.llm.aask(msg=msg)


PROMPT = """
## 角色、系统、外部系统
{use_case_actors}

## 用户需求
{original_user_requirements}

## 先前的交互事件
{previous_version_of_interaction_events}

## 评估结论
{the_evaluation_conclusion_of_previous_version_of_trd}

---
你是一个用于捕捉交互事件的工具。
- "角色、系统、外部系统" 提供了交互事件的可能参与者；
- "先前的交互事件" 是你之前输出的交互事件内容；
- "评估结论" 可能涉及 "用户需求" 中的部分描述，并对 "先前的交互事件" 的内容提出了一些问题；
- 你的任务是从 "用户需求" 内容中逐字捕捉交互事件，包括：
  1. 识别交互的参与者（每个交互事件最多只能有 2 个参与者；如果有多个参与者，说明多个事件被合并，应进一步拆分）；
  2. 确定交互事件的发起者，并识别发起者输入的数据；
  3. 根据 "用户需求" 识别交互事件的最终返回数据。

你可以检查 "用户需求" 描述中的数据流，看看是否有缺失的交互事件；
返回一个 Markdown JSON 对象列表，每个对象包含：
- "name" 键：交互事件的名称；
- "participants" 键：包含两个参与者名称的字符串列表；
- "initiator" 键：包含发起交互事件的参与者名称；
- "input" 键：包含输入数据的自然语言描述；
"""
