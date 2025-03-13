#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : write_trd.py
@Desc    : The implementation of Chapter 2.1.6~2.1.7 of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import general_after_log, to_markdown_code_block


@register_tool(include_functions=["run"])
class WriteTRD(Action):
    """WriteTRD 处理以下情况：
    1. 给定一些新的用户需求，写出一份新的技术需求文档（TRD）。
    2. 给定一些增量的用户需求，更新旧版的技术需求文档（TRD）。
    """

    async def run(
        self,
        *,
        user_requirements: str = "",
        use_case_actors: str,
        available_external_interfaces: str,
        evaluation_conclusion: str = "",
        interaction_events: str,
        previous_version_trd: str = "",
        legacy_user_requirements: str = "",
        legacy_user_requirements_trd: str = "",
        legacy_user_requirements_interaction_events: str = "",
    ) -> str:
        """
        根据用户需求处理技术需求文档（TRD）的编写或更新。

        参数：
            user_requirements (str): 新的/增量的用户需求。
            use_case_actors (str): 用例中涉及的参与者描述。
            available_external_interfaces (str): 可用的外部接口列表。
            evaluation_conclusion (str, 可选): 评估您编写的 TRD 的结论，默认为空字符串。
            interaction_events (str): 与您处理的用户需求相关的交互事件。
            previous_version_trd (str, 可选): 您编写的 TRD 的上一版本，用于更新。
            legacy_user_requirements (str, 可选): 外部对象处理的现有用户需求供您使用，默认为空字符串。
            legacy_user_requirements_trd (str, 可选): 与现有用户需求相关的 TRD，由外部对象处理，供您使用，默认为空字符串。
            legacy_user_requirements_interaction_events (str, 可选): 与现有用户需求相关的交互事件，由外部对象处理，供您使用，默认为空字符串。

        返回：
            str: 由您编写的新的或更新后的 TRD。

        示例：
            >>> # 给定新的用户需求，编写一份新的 TRD。
            >>> user_requirements = "编写一个 '贪吃蛇游戏' 的 TRD。"
            >>> use_case_actors = "- 参与者：游戏玩家;\\n- 系统：贪吃蛇游戏; \\n- 外部系统：游戏中心;"
            >>> available_external_interfaces = "`CompressExternalInterfaces.run` 返回的可用外部接口是 ..."
            >>> previous_version_trd = "TRD ..." # 如果有的话，上一版本的 TRD。
            >>> evaluation_conclusion = "结论 ..." # 如果有的话，`EvaluateTRD.run` 返回的结论。
            >>> interaction_events = "交互事件 ..." # `DetectInteraction.run` 返回的交互事件。
            >>> write_trd = WriteTRD()
            >>> new_version_trd = await write_trd.run(
            >>>     user_requirements=user_requirements,
            >>>     use_case_actors=use_case_actors,
            >>>     available_external_interfaces=available_external_interfaces,
            >>>     evaluation_conclusion=evaluation_conclusion,
            >>>     interaction_events=interaction_events,
            >>>     previous_version_trd=previous_version_trd,
            >>> )
            >>> print(new_version_trd)
            ## 技术需求文档\n ...

            >>> # 给定增量需求，更新旧版 TRD。
            >>> legacy_user_requirements = ["用户需求 1. ...", "用户需求 2. ...", ...]
            >>> legacy_user_requirements_trd = "## 技术需求文档\\n ..." # 在集成更多用户需求之前的 TRD。
            >>> legacy_user_requirements_interaction_events = ["用户需求 1 的交互事件列表 ...", "用户需求 2 的交互事件列表 ...", ...]
            >>> use_case_actors = "- 参与者：游戏玩家;\\n- 系统：贪吃蛇游戏; \\n- 外部系统：游戏中心;"
            >>> available_external_interfaces = "`CompressExternalInterfaces.run` 返回的可用外部接口是 ..."
            >>> increment_requirements = "增量的用户需求是 ..."
            >>> evaluation_conclusion = "结论 ..." # 如果有的话，`EvaluateTRD.run` 返回的结论。
            >>> previous_version_trd = "TRD ..." # 如果有的话，上一版本的 TRD。
            >>> write_trd = WriteTRD()
            >>> new_version_trd = await write_trd.run(
            >>>     user_requirements=increment_requirements,
            >>>     use_case_actors=use_case_actors,
            >>>     available_external_interfaces=available_external_interfaces,
            >>>     evaluation_conclusion=evaluation_conclusion,
            >>>     interaction_events=interaction_events,
            >>>     previous_version_trd=previous_version_trd,
            >>>     legacy_user_requirements=str(legacy_user_requirements),
            >>>     legacy_user_requirements_trd=legacy_user_requirements_trd,
            >>>     legacy_user_requirements_interaction_events=str(legacy_user_requirements_interaction_events),
            >>> )
            >>> print(new_version_trd)
            ## 技术需求文档\n ...
        """
        if legacy_user_requirements:
            return await self._write_incremental_trd(
                use_case_actors=use_case_actors,
                legacy_user_requirements=legacy_user_requirements,
                available_external_interfaces=available_external_interfaces,
                legacy_user_requirements_trd=legacy_user_requirements_trd,
                legacy_user_requirements_interaction_events=legacy_user_requirements_interaction_events,
                incremental_user_requirements=user_requirements,
                previous_version_trd=previous_version_trd,
                evaluation_conclusion=evaluation_conclusion,
                incremental_user_requirements_interaction_events=interaction_events,
            )
        return await self._write_new_trd(
            use_case_actors=use_case_actors,
            original_user_requirement=user_requirements,
            available_external_interfaces=available_external_interfaces,
            legacy_trd=previous_version_trd,
            evaluation_conclusion=evaluation_conclusion,
            interaction_events=interaction_events,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _write_new_trd(
        self,
        *,
        use_case_actors: str,
        original_user_requirement: str,
        available_external_interfaces: str,
        legacy_trd: str,
        evaluation_conclusion: str,
        interaction_events: str,
    ) -> str:
        prompt = NEW_PROMPT.format(
            use_case_actors=use_case_actors,
            original_user_requirement=to_markdown_code_block(val=original_user_requirement),
            available_external_interfaces=available_external_interfaces,
            legacy_trd=to_markdown_code_block(val=legacy_trd),
            evaluation_conclusion=evaluation_conclusion,
            interaction_events=interaction_events,
        )
        return await self.llm.aask(prompt)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _write_incremental_trd(
        self,
        *,
        use_case_actors: str,
        legacy_user_requirements: str,
        available_external_interfaces: str,
        legacy_user_requirements_trd: str,
        legacy_user_requirements_interaction_events: str,
        incremental_user_requirements: str,
        previous_version_trd: str,
        evaluation_conclusion: str,
        incremental_user_requirements_interaction_events: str,
    ):
        prompt = INCREMENTAL_PROMPT.format(
            use_case_actors=use_case_actors,
            legacy_user_requirements=to_markdown_code_block(val=legacy_user_requirements),
            available_external_interfaces=available_external_interfaces,
            legacy_user_requirements_trd=to_markdown_code_block(val=legacy_user_requirements_trd),
            legacy_user_requirements_interaction_events=legacy_user_requirements_interaction_events,
            incremental_user_requirements=to_markdown_code_block(val=incremental_user_requirements),
            previous_version_trd=to_markdown_code_block(val=previous_version_trd),
            evaluation_conclusion=evaluation_conclusion,
            incremental_user_requirements_interaction_events=incremental_user_requirements_interaction_events,
        )
        return await self.llm.aask(prompt)


NEW_PROMPT = """
## 参与者、系统、外部系统
{use_case_actors}

## 用户需求
{original_user_requirement}

## 可用的外部接口
{available_external_interfaces}

## 旧版技术需求文档（Legacy TRD）
{legacy_trd}

## 评估结论
{evaluation_conclusion}

## 交互事件
{interaction_events}

---
你是一个技术需求文档（TRD）生成器。
"参与者、系统、外部系统"部分提供了UML用例图中出现的参与者和系统的说明；
"可用的外部接口"部分提供了每个步骤的候选步骤，以及每个步骤的输入和输出；
"用户需求"部分提供了原始需求描述，任何未提到的部分将由其他模块处理，所以不要编造需求；
"旧版技术需求文档"部分提供了基于"用户需求"生成的旧版TRD，可以作为新版TRD的参考；
"评估结论"部分提供了对旧版TRD的评估总结，可以作为新版TRD的参考；
"交互事件"部分提供了根据"用户需求"识别的交互事件及其交互参与者；
1. "用户需求"中描述了哪些输入和输出？
2. 为了实现"用户需求"中描述的输入和输出，至少需要多少步骤？每个步骤涉及哪些来自"参与者、系统、外部系统"部分的参与者？每个步骤的输入和输出是什么？这些输出在哪里使用，例如，作为哪个接口的输入或在哪些需求中需要等？
3. 输出完整的技术需求文档（TRD）：
  3.1. 在描述中，使用"参与者、系统、外部系统"部分定义的参与者和系统来描述交互者；
  3.2. 内容应包含"用户需求"的原始文本；
  3.3. 在TRD中，每个步骤最多涉及两个参与者。如果有超过两个参与者，步骤需要进一步拆分；
  3.4. 在TRD中，每个步骤必须包括详细的描述、输入、输出、参与者、发起者和步骤存在的理由。理由应参考原始文本来解释，例如，指定哪个接口需要此步骤的输出作为参数，或在需求中说明此步骤的必要性；
  3.5. 在TRD中，如果需要调用外部系统的接口，必须明确指定你要调用的外部系统的接口ID；
"""

INCREMENTAL_PROMPT = """
## 参与者、系统、外部系统
{use_case_actors}

## 旧版用户需求
{legacy_user_requirements}

## 可用的外部接口
{available_external_interfaces}

## 旧版用户需求的技术需求文档（TRD）
{legacy_user_requirements_trd}

## 旧版用户需求的交互事件
{legacy_user_requirements_interaction_events}

## 增量需求
{incremental_user_requirements}

## 旧版技术需求文档（Legacy TRD）
{previous_version_trd}

## 评估结论
{evaluation_conclusion}

## 交互事件
{incremental_user_requirements_interaction_events}

---
你是一个技术需求文档（TRD）生成器。
"参与者、系统、外部系统"部分提供了UML用例图中出现的参与者和系统的说明；
"可用的外部接口"部分提供了每个步骤的候选步骤，以及每个步骤的输入和输出；
"旧版用户需求"部分提供了由其他模块处理的原始需求描述供你使用；
"旧版用户需求的技术需求文档"部分是由其他模块基于"旧版用户需求"生成的TRD供你使用；
"旧版用户需求的交互事件"部分是由其他模块基于"旧版用户需求"生成的交互事件供你使用；
"增量需求"部分提供了需要你处理的原始需求描述，任何未提到的部分将由其他模块处理，所以不要编造需求；
"旧版技术需求文档"部分提供了你之前生成的基于"增量需求"的旧版TRD，可以作为新版TRD的参考；
"评估结论"部分提供了你生成的旧版TRD的评估总结，发现的问题可以作为新版TRD的参考；
"交互事件"部分提供了根据"增量需求"识别的交互事件及其交互参与者；
1. "增量需求"中描述了哪些输入和输出？
2. 为了实现"增量需求"中描述的输入和输出，至少需要多少步骤？每个步骤涉及哪些来自"参与者、系统、外部系统"部分的参与者？每个步骤的输入和输出是什么？这些输出在哪里使用，例如，作为哪个接口的输入或在哪些需求中需要等？
3. 输出完整的技术需求文档（TRD）：
  3.1. 在描述中，使用"参与者、系统、外部系统"部分定义的参与者和系统来描述交互者；
  3.2. 内容应包含"用户需求"的原始文本；
  3.3. 在TRD中，每个步骤最多涉及两个参与者。如果有超过两个参与者，步骤需要进一步拆分；
  3.4. 在TRD中，每个步骤必须包括详细的描述、输入、输出、参与者、发起者和步骤存在的理由。理由应参考原始文本来解释，例如，指定哪个接口需要此步骤的输出作为参数，或在需求中说明此步骤的必要性；
  3.5. 在TRD中，如果需要调用外部系统的接口，必须明确指定你要调用的外部系统的接口ID；
"""
