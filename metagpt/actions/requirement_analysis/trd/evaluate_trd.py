#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : evaluate_trd.py
@Desc    : The implementation of Chapter 2.1.6~2.1.7 of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""

from metagpt.actions.requirement_analysis import EvaluateAction, EvaluationData
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import to_markdown_code_block


@register_tool(include_functions=["run"])
class EvaluateTRD(EvaluateAction):
    """EvaluateTRD 处理以下情况：
    1. 给定一个 TRD（技术需求文档），评估其质量并返回结论。
    """

    async def run(
        self,
        *,
        user_requirements: str,
        use_case_actors: str,
        trd: str,
        interaction_events: str,
        legacy_user_requirements_interaction_events: str = "",
    ) -> EvaluationData:
        """
        根据用户需求、用例参与者、交互事件以及可选的外部交互事件来评估给定的 TRD。

        参数：
            user_requirements (str): 用户提供的需求。
            use_case_actors (str): 用例涉及的参与者。
            trd (str): 需要评估的 TRD（技术需求文档）。
            interaction_events (str): 相关的交互事件，包括用户需求与 TRD 的交互事件。
            legacy_user_requirements_interaction_events (str, 可选): 先前与用户需求相关的外部交互事件，默认为空字符串。

        返回：
            EvaluationData: TRD 评估的结论。

        示例：
            >>> evaluate_trd = EvaluateTRD()
            >>> user_requirements = "用户需求 1. ..."
            >>> use_case_actors = "- 角色: 游戏玩家;\\n- 系统: 贪吃蛇游戏; \\n- 外部系统: 游戏中心;"
            >>> trd = "## TRD\\n..."
            >>> interaction_events = "['交互事件 ...', ...]"
            >>> evaluation_conclusion = "问题: ..."
            >>> legacy_user_requirements_interaction_events = ["用户需求 1. ...", ...]
            >>> evaluation = await evaluate_trd.run(
            >>>    user_requirements=user_requirements,
            >>>    use_case_actors=use_case_actors,
            >>>    trd=trd,
            >>>    interaction_events=interaction_events,
            >>>    legacy_user_requirements_interaction_events=str(legacy_user_requirements_interaction_events),
            >>> )
            >>> is_pass = evaluation.is_pass
            >>> print(is_pass)
            True
            >>> evaluation_conclusion = evaluation.conclusion
            >>> print(evaluation_conclusion)
            ## 结论\n balabalabala...
        """
        prompt = PROMPT.format(
            use_case_actors=use_case_actors,
            user_requirements=to_markdown_code_block(val=user_requirements),
            trd=to_markdown_code_block(val=trd),
            legacy_user_requirements_interaction_events=legacy_user_requirements_interaction_events,
            interaction_events=interaction_events,
        )
        return await self._vote(prompt)


PROMPT = """
## 角色、系统、外部系统
{use_case_actors}

## 用户需求
{user_requirements}

## TRD 设计
{trd}

## 外部交互事件
{legacy_user_requirements_interaction_events}

## 交互事件
{legacy_user_requirements_interaction_events}
{interaction_events}

---
你是一个用于评估 TRD 设计的工具。
- "角色、系统、外部系统" 提供了交互事件中所有可能的参与者；
- "用户需求" 提供了原始需求描述，任何未在此描述中提及的部分将由其他模块处理，因此请勿凭空捏造需求；
- "外部交互事件" 由外部模块提供供你使用，其内容与 "交互事件" 部分相关，"外部交互事件" 的内容可视为无问题；
- "外部交互事件" 提供了一些已识别的交互事件及其参与者，这些内容基于 "用户需求" 的部分内容；
- "交互事件" 提供了一些已识别的交互事件及其参与者，这些内容基于 "用户需求" 的全部内容；
- "TRD 设计" 详细描述了实现原始需求的步骤，它结合了 "交互事件" 中的交互，并补充了其他步骤以形成完整的上下游数据流；
- 为了整合完整的数据流，"TRD 设计" 可以包含未在 "用户需求" 中明确描述的步骤，但不能与 "用户需求" 中已明确描述的内容冲突。

你的任务：
1. 确定 "交互事件" 中的交互步骤与 "TRD 设计" 中的哪些步骤相对应，并提供理由。
2. 找出 "TRD 设计" 和 "交互事件" 中哪些部分与 "用户需求" 描述不符，并详细说明原因。
3. 如果 "用户需求" 描述的内容被拆分成多个步骤出现在 "TRD 设计" 和 "交互事件" 中，只要不冲突，就可以视为符合 "用户需求"。
4. "用户需求" 描述的内容可能存在遗漏，"TRD 设计" 和 "交互事件" 允许补充额外的步骤，但不能与 "用户需求" 冲突。
5. 如果 "TRD 设计" 中涉及与外部系统的交互，你必须明确指定使用的外部接口 ID，并且该接口的输入输出参数必须与交互事件的输入输出数据完全匹配。
6. 评估 "交互事件" 的步骤顺序是否会引发性能或成本问题，并详细说明理由。
7. 确保 "TRD 设计" 的每个步骤都有明确的输入数据，这些输入数据应来自前序步骤的输出，或者由 "角色、系统、外部系统" 提供，不应出现无来源数据。

返回一个 Markdown 格式的 JSON 对象，包含：
- "issues"：包含 "TRD 设计" 中需要解决的问题的字符串列表，每个问题必须有详细描述和理由；
- "conclusion"：评估结论；
- "correspondence_between"：字符串列表，描述 "交互事件" 与 "TRD 设计" 步骤的对应关系；
- "misalignment"：字符串列表，描述与 "用户需求" 不符的部分；
- "is_pass"：如果 "TRD 设计" 没有问题，则值为 `true`。
"""
