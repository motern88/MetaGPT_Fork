#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : write_framework.py
@Desc    : The implementation of Chapter 2.1.8 of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""
import json

from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import general_after_log, to_markdown_code_block


@register_tool(include_functions=["run"])
class WriteFramework(Action):
    """WriteFramework 处理以下情况：
    1. 给定 TRD（技术需求文档），生成软件框架。
    """

    async def run(
        self,
        *,
        use_case_actors: str,  # 用例参与者（包括角色、系统、外部系统）
        trd: str,  # 技术需求文档（Technical Requirements Document）
        acknowledge: str,  # 外部系统接口信息
        legacy_output: str,  # 之前生成的软件框架（用于增量改进）
        evaluation_conclusion: str,  # 评估结论（用于改进软件框架）
        additional_technical_requirements: str,  # 额外的技术要求
    ) -> str:
        """
        根据提供的 TRD 和相关信息生成软件框架。

        参数:
            use_case_actors (str): 参与用例的角色描述。
            trd (str): 技术需求文档，详细描述软件需求。
            acknowledge (str): 需要使用的外部接口信息。
            legacy_output (str): 之前 `WriteFramework.run` 生成的旧软件框架。
            evaluation_conclusion (str): 需求评估后的改进建议。
            additional_technical_requirements (str): 其他技术要求。

        返回:
            str: 生成的软件框架，以 JSON 格式返回文件列表。

        示例:
            >>> write_framework = WriteFramework()
            >>> use_case_actors = "- Actor: game player;\\n- System: snake game;\\n- External System: game center;"
            >>> trd = "## TRD\\n..."
            >>> acknowledge = "## Interfaces\\n..."
            >>> legacy_output = '{"path":"balabala", "filename":"...", ...'
            >>> evaluation_conclusion = "需要优化模块A的结构..."
            >>> constraint = "使用 Java 语言..."
            >>> framework = await write_framework.run(
            >>>    use_case_actors=use_case_actors,
            >>>    trd=trd,
            >>>    acknowledge=acknowledge,
            >>>    legacy_output=framework,
            >>>    evaluation_conclusion=evaluation_conclusion,
            >>>    additional_technical_requirements=constraint,
            >>> )
            >>> print(framework)
            {"path":"balabala", "filename":"...", ...}
        """
        # 提取 TRD 中使用的外部接口，仅保留相关部分
        acknowledge = await self._extract_external_interfaces(trd=trd, knowledge=acknowledge)

        # 生成最终提示词
        prompt = PROMPT.format(
            use_case_actors=use_case_actors,
            trd=to_markdown_code_block(val=trd),
            acknowledge=to_markdown_code_block(val=acknowledge),
            legacy_output=to_markdown_code_block(val=legacy_output),
            evaluation_conclusion=evaluation_conclusion,
            additional_technical_requirements=to_markdown_code_block(val=additional_technical_requirements),
        )

        # 生成软件框架
        return await self._write(prompt)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _write(self, prompt: str) -> str:
        """调用 LLM 生成软件框架"""
        rsp = await self.llm.aask(prompt)

        # 解析 JSON 格式输出
        tags = ["```json", "```"]
        bix = rsp.find(tags[0])
        eix = rsp.rfind(tags[1])
        if bix >= 0:
            rsp = rsp[bix : eix + len(tags[1])]

        # 去除代码块标记，提取 JSON 数据
        json_data = rsp.removeprefix("```json").removesuffix("```")

        # 验证 JSON 格式
        json.loads(json_data)

        return json_data

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _extract_external_interfaces(self, trd: str, knowledge: str) -> str:
        """提取 TRD 中使用的外部接口，并删除无关内容"""
        prompt = f"## TRD\n{to_markdown_code_block(val=trd)}\n\n## Knowledge\n{to_markdown_code_block(val=knowledge)}\n"

        rsp = await self.llm.aask(
            prompt,
            system_msgs=[
                "你是一个负责清理文章杂质的工具；你可以去除与文章无关的内容。",
                '找出 "TRD" 中使用的接口，并从 "Knowledge" 中删除未被使用的接口，返回精简后的 "Knowledge"。',
            ],
        )
        return rsp



# 生成软件框架的提示模板
PROMPT = """
## Actor, System, External System
{use_case_actors}

## TRD
{trd}

## Acknowledge
{acknowledge}

## Legacy Outputs
{legacy_output}

## Evaluation Conclusion
{evaluation_conclusion}

## Additional Technical Requirements
{additional_technical_requirements}

---
你是一个根据 TRD 生成软件框架代码的工具。

- "Actor, System, External System" 说明了 UML 用例图中涉及的角色和系统；
- "Acknowledge" 包含外部系统接口描述，仅当 "TRD" 需要时才实现；
- "Legacy Outputs" 是上次生成的代码，可在此基础上优化；
- "Evaluation Conclusion" 指出了需要改进的部分；
- "Additional Technical Requirements" 规定了代码必须满足的附加技术要求。

生成的框架应包括：
- `README.md`：
  - 项目目录结构图；
  - 类、接口、函数与 "TRD" 之间的对应关系；
  - 必要的前置条件；
  - 安装、配置、使用说明；
- `CLASS.md`：基于 "TRD" 的 PlantUML 类图；
- `SEQUENCE.md`：基于 "TRD" 的 PlantUML 序列图；
- 源代码文件（实现 "TRD" 和 "Additional Technical Requirements"，代码不需要注释）；
- 必要的配置文件。

返回 JSON 格式，包含：
- `"path"`：文件路径；
- `"filename"`：文件名；
- `"content"`：文件内容；
"""
