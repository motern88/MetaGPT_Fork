#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/13
@Author  : mashenquan
@File    : evaluate_framework.py
@Desc    : The implementation of Chapter 2.1.8 of RFC243. https://deepwisdom.feishu.cn/wiki/QobGwPkImijoyukBUKHcrYetnBb
"""

from metagpt.actions.requirement_analysis import EvaluateAction, EvaluationData
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import to_markdown_code_block


# 注册工具，并仅包含 `run` 方法
@register_tool(include_functions=["run"])
class EvaluateFramework(EvaluateAction):
    """WriteFramework 处理以下情况：
    1. 给定技术需求文档 (TRD) 和基于 TRD 生成的软件框架，评估该软件框架的质量。
    """

    async def run(
        self,
        *,
        use_case_actors: str,  # 用例中的角色
        trd: str,  # 技术需求文档（Technical Requirements Document, TRD）
        acknowledge: str,  # 相关的接口或外部系统信息
        legacy_output: str,  # 之前版本的软件框架
        additional_technical_requirements: str,  # 额外的技术要求
    ) -> EvaluationData:
        """
        根据提供的 TRD 和相关参数，对软件框架进行评估。

        参数：
            use_case_actors (str): 用例涉及的角色描述。
            trd (str): 技术需求文档 (TRD)，描述软件框架的需求。
            acknowledge (str): 外部系统的接口信息或相关的确认信息。
            legacy_output (str): 之前 `WriteFramework` 生成的旧版软件框架。
            additional_technical_requirements (str): 需要在评估时考虑的额外技术要求。

        返回：
            EvaluationData: 评估结果数据对象。

        示例：
            >>> evaluate_framework = EvaluateFramework()
            >>> use_case_actors = "- Actor: 游戏玩家;\\n- System: 贪吃蛇游戏; \\n- External System: 游戏中心;"
            >>> trd = "## TRD\\n..."
            >>> acknowledge = "## 接口\\n..."
            >>> framework = '{"path":"balabala", "filename":"...", ...'
            >>> constraint = "使用 Java 语言，..."
            >>> evaluation = await evaluate_framework.run(
            >>>     use_case_actors=use_case_actors,
            >>>     trd=trd,
            >>>     acknowledge=acknowledge,
            >>>     legacy_output=framework,
            >>>     additional_technical_requirements=constraint,
            >>> )
            >>> is_pass = evaluation.is_pass
            >>> print(is_pass)
            True
            >>> evaluation_conclusion = evaluation.conclusion
            >>> print(evaluation_conclusion)
            Balabala...
        """
        # 格式化提示词，将输入参数转换为 Markdown 代码块格式
        prompt = PROMPT.format(
            use_case_actors=use_case_actors,
            trd=to_markdown_code_block(val=trd),
            acknowledge=to_markdown_code_block(val=acknowledge),
            legacy_output=to_markdown_code_block(val=legacy_output),
            additional_technical_requirements=to_markdown_code_block(val=additional_technical_requirements),
        )
        return await self._vote(prompt)  # 运行评估


# 评估软件框架的提示词
PROMPT = """
## 角色、系统、外部系统
{use_case_actors}

## 旧版 TRD
{trd}

## 相关接口信息
{acknowledge}

## 旧版软件框架
{legacy_output}

## 额外技术要求
{additional_technical_requirements}

---
你是一个工具，用于根据 TRD 评估框架代码的质量；
你需要参考 "旧版 TRD" 章节的内容，检查 "旧版软件框架" 是否存在错误或遗漏；
"角色、系统、外部系统" 提供了 UML 用例图中涉及的角色和系统信息；
如果 "旧版 TRD" 缺少外部系统的信息，可以在 "相关接口信息" 章节中找到；
在 "相关接口信息" 章节中定义的接口，哪些在 "旧版 TRD" 中被使用了？
除非某个接口在 "旧版 TRD" 中使用，否则不要实现 "相关接口信息" 章节中的接口；
可以通过 ID 或 URL 判断接口是否相同；
"旧版 TRD" 中未提及的部分会由其他 TRD 处理，因此 "旧版 TRD" 中未出现的流程视为已准备就绪；
"额外技术要求" 指定了软件框架代码必须满足的附加技术要求；
检查代码中使用的外部系统接口参数是否符合 "相关接口信息" 中的规范；
是否缺少必要的配置文件？

返回一个 Markdown 格式的 JSON 对象，包含：
- `"issues"`: 字符串列表，列出 "旧版软件框架" 存在的问题（如果有），每个问题都要有详细描述和原因；
- `"conclusion"`: 评估结论；
- `"misalignment"`: 字符串列表，描述 "旧版 TRD" 与代码之间的不匹配之处；
- `"is_pass"`: 布尔值，如果 "旧版软件框架" 没有发现任何问题，则为 `true`；
"""
