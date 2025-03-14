#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from metagpt.actions.requirement_analysis.framework import (
    EvaluateFramework,
    WriteFramework,
    save_framework,
)
from metagpt.actions.requirement_analysis.trd import (
    CompressExternalInterfaces,
    DetectInteraction,
    EvaluateTRD,
    WriteTRD,
)
from metagpt.const import ASSISTANT_ALIAS, TEST_DATA_PATH
from metagpt.context import Context
from metagpt.logs import ToolLogItem, log_tool_output, logger
from metagpt.utils.common import aread
from metagpt.utils.cost_manager import CostManager


async def import_git_repo(url: str) -> Path:
    """
    从 Git 网址导入项目，并格式化为 MetaGPT 项目格式，以支持增量需求追加。

    参数:
        url (str): Git 项目 URL，例如 "https://github.com/geekan/MetaGPT.git"。

    返回:
        Path: 格式化后的项目路径。

    示例:
        >>> git_url = "https://github.com/geekan/MetaGPT.git"
        >>> formatted_project_path = await import_git_repo(git_url)
        >>> print("格式化后的项目路径:", formatted_project_path)
        /PATH/TO/THE/FORMMATTED/PROJECT
    """
    from metagpt.actions.import_repo import ImportRepo
    from metagpt.context import Context

    log_tool_output(
        output=[ToolLogItem(name=ASSISTANT_ALIAS, value=import_git_repo.__name__)], tool_name=import_git_repo.__name__
    )

    ctx = Context()
    action = ImportRepo(repo_path=url, context=ctx)
    await action.run()

    outputs = [ToolLogItem(name="MetaGPT Project", value=str(ctx.repo.workdir))]
    log_tool_output(output=outputs, tool_name=import_git_repo.__name__)

    return ctx.repo.workdir


async def extract_external_interfaces(acknowledge: str) -> str:
    """
    提取并压缩关于外部系统接口的信息。

    参数:
        acknowledge (str): 包含外部系统接口详情的文本信息。

    返回:
        str: 压缩后的外部系统接口信息。

    示例:
        >>> acknowledge = "## 接口信息\\n..."
        >>> external_interfaces = await extract_external_interfaces(acknowledge=acknowledge)
        >>> print(external_interfaces)
        ```json\n[\n{\n"id": 1,\n"inputs": {..."
    """
    compress_acknowledge = CompressExternalInterfaces()
    return await compress_acknowledge.run(acknowledge=acknowledge)


async def mock_asearch_acknowledgement(use_case_actors: str):
    return await aread(filename=TEST_DATA_PATH / "requirements/1.acknowledge.md")


async def write_trd(
    use_case_actors: str,
    user_requirements: str,
    investment: float = 10,
    context: Optional[Context] = None,
) -> str:
    """
    生成技术需求文档 (TRD)。

    参数:
        user_requirements (str): 用户的新需求或增量需求。
        use_case_actors (str): 参与该用例的角色描述。
        investment (float): 预算，当预算超支时停止优化 TRD。
        context (Context, 可选): 上下文配置，默认 None。

    返回:
        str: 生成的 TRD。

    示例:
        >>> user_requirements = "编写一个'贪吃蛇游戏'的 TRD。"
        >>> use_case_actors = "- 角色: 游戏玩家;\\n- 系统: 贪吃蛇游戏; \\n- 外部系统: 游戏中心;"
        >>> investment = 10.0
        >>> trd = await write_trd(
        >>>     user_requirements=user_requirements,
        >>>     use_case_actors=use_case_actors,
        >>>     investment=investment,
        >>> )
        >>> print(trd)
        ## 技术需求文档 (TRD)\\n ...
    """
    context = context or Context(cost_manager=CostManager(max_budget=investment))
    compress_acknowledge = CompressExternalInterfaces()
    acknowledgement = await mock_asearch_acknowledgement(use_case_actors)  # Replaced by acknowledgement_repo later.
    external_interfaces = await compress_acknowledge.run(acknowledge=acknowledgement)
    detect_interaction = DetectInteraction(context=context)
    w_trd = WriteTRD(context=context)
    evaluate_trd = EvaluateTRD(context=context)
    is_pass = False
    evaluation_conclusion = ""
    interaction_events = ""
    trd = ""
    while not is_pass and (context.cost_manager.total_cost < context.cost_manager.max_budget):
        interaction_events = await detect_interaction.run(
            user_requirements=user_requirements,
            use_case_actors=use_case_actors,
            legacy_interaction_events=interaction_events,
            evaluation_conclusion=evaluation_conclusion,
        )
        trd = await w_trd.run(
            user_requirements=user_requirements,
            use_case_actors=use_case_actors,
            available_external_interfaces=external_interfaces,
            evaluation_conclusion=evaluation_conclusion,
            interaction_events=interaction_events,
            previous_version_trd=trd,
        )
        evaluation = await evaluate_trd.run(
            user_requirements=user_requirements,
            use_case_actors=use_case_actors,
            trd=trd,
            interaction_events=interaction_events,
        )
        is_pass = evaluation.is_pass
        evaluation_conclusion = evaluation.conclusion

    return trd


async def write_framework(
    use_case_actors: str,
    trd: str,
    additional_technical_requirements: str,
    output_dir: Optional[str] = "",
    investment: float = 20.0,
    context: Optional[Context] = None,
    max_loop: int = 20,
) -> str:
    """
    生成软件框架。

    参数:
        use_case_actors (str): 用例相关的角色描述。
        trd (str): 技术需求文档 (TRD)。
        additional_technical_requirements (str): 额外的技术需求。
        output_dir (str, 可选): 软件框架保存路径，默认空字符串。
        investment (float): 预算，超出预算则停止优化。
        context (Context, 可选): 上下文配置，默认 None。
        max_loop (int, 可选): 用于限制循环次数，避免成本统计失效，默认 20。

    返回:
        str: 生成的软件框架路径信息。

    示例:
        >>> use_case_actors = "- 角色: 游戏玩家;\\n- 系统: 贪吃蛇游戏; \\n- 外部系统: 游戏中心;"
        >>> trd = "## TRD\\n..."
        >>> additional_technical_requirements = "使用 Java 语言, ..."
        >>> investment = 15.0
        >>> framework = await write_framework(
        >>>    use_case_actors=use_case_actors,
        >>>    trd=trd,
        >>>    additional_technical_requirements=additional_technical_requirements,
        >>>    investment=investment,
        >>> )
        >>> print(framework)
        [{"path":"balabala", "filename":"...", ...
    """
    context = context or Context(cost_manager=CostManager(max_budget=investment))
    write_framework = WriteFramework(context=context)
    evaluate_framework = EvaluateFramework(context=context)
    is_pass = False
    framework = ""
    evaluation_conclusion = ""
    acknowledgement = await mock_asearch_acknowledgement(use_case_actors)  # Replaced by acknowledgement_repo later.
    loop_count = 0
    output_dir = (
        Path(output_dir)
        if output_dir
        else context.config.workspace.path / (datetime.now().strftime("%Y%m%d%H%M%ST") + uuid.uuid4().hex[0:8])
    )
    file_list = []
    while not is_pass and (context.cost_manager.total_cost < context.cost_manager.max_budget):
        try:
            framework = await write_framework.run(
                use_case_actors=use_case_actors,
                trd=trd,
                acknowledge=acknowledgement,
                legacy_output=framework,
                evaluation_conclusion=evaluation_conclusion,
                additional_technical_requirements=additional_technical_requirements,
            )
        except Exception as e:
            logger.info(f"{e}")
            break
        evaluation = await evaluate_framework.run(
            use_case_actors=use_case_actors,
            trd=trd,
            acknowledge=acknowledgement,
            legacy_output=framework,
            additional_technical_requirements=additional_technical_requirements,
        )
        is_pass = evaluation.is_pass
        evaluation_conclusion = evaluation.conclusion
        loop_count += 1
        logger.info(f"Loop {loop_count}")
        if context.cost_manager.total_cost < 1 and loop_count > max_loop:
            break
        file_list = await save_framework(dir_data=framework, trd=trd, output_dir=output_dir)
        logger.info(f"Output:\n{file_list}")

    return "## Software Framework" + "".join([f"\n- {i}" for i in file_list])


async def write_trd_and_framework(
    use_case_actors: str,
    user_requirements: str,
    additional_technical_requirements: str,
    investment: float = 50.0,
    output_dir: Optional[str] = "",
    context: Optional[Context] = None,
) -> str:
    """
    生成 TRD 并基于 TRD 生成软件框架。

    参数:
        use_case_actors (str): 参与用例的角色描述。
        user_requirements (str): 用户需求描述。
        additional_technical_requirements (str): 额外的技术需求。
        investment (float): 预算，超出预算则停止优化。
        output_dir (str, 可选): 生成的软件框架的保存路径，默认空字符串。
        context (Context, 可选): 上下文配置，默认 None。

    返回:
        str: 生成的软件框架信息。

    示例:
        >>> use_case_actors = "- 角色: 游戏玩家;\\n- 系统: 贪吃蛇游戏; \\n- 外部系统: 游戏中心;"
        >>> user_requirements = "编写一个'贪吃蛇游戏'的 TRD。"
        >>> additional_technical_requirements = "使用 Python 语言..."
        >>> framework = await write_trd_and_framework(
        >>>    use_case_actors=use_case_actors,
        >>>    user_requirements=user_requirements,
        >>>    additional_technical_requirements=additional_technical_requirements,
        >>>    investment=50.0,
        >>> )
        >>> print(framework)
        ## 软件框架\\n - /path/to/generated/framework
    """
    context = context or Context(cost_manager=CostManager(max_budget=investment))
    trd = await write_trd(use_case_actors=use_case_actors, user_requirements=user_requirements, context=context)
    return await write_framework(
        use_case_actors=use_case_actors,
        trd=trd,
        additional_technical_requirements=additional_technical_requirements,
        output_dir=output_dir,
        context=context,
    )
