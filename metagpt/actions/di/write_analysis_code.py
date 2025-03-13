# -*- encoding: utf-8 -*-
"""
@Date    :   2023/11/20 13:19:39
@Author  :   orange-crow
@File    :   write_analysis_code.py
"""
from __future__ import annotations

from metagpt.actions import Action
from metagpt.prompts.di.write_analysis_code import (
    CHECK_DATA_PROMPT,
    DEBUG_REFLECTION_EXAMPLE,
    INTERPRETER_SYSTEM_MSG,
    REFLECTION_PROMPT,
    REFLECTION_SYSTEM_MSG,
    STRUCTUAL_PROMPT,
)
from metagpt.schema import Message, Plan
from metagpt.utils.common import CodeParser, remove_comments


class WriteAnalysisCode(Action):
    async def _debug_with_reflection(self, context: list[Message], working_memory: list[Message]):
        # 创建反思提示，格式化为包含上下文和工作记忆的信息
        reflection_prompt = REFLECTION_PROMPT.format(
            debug_example=DEBUG_REFLECTION_EXAMPLE,
            context=context,
            previous_impl=working_memory,
        )

        # 调用 LLM（大语言模型）进行反思分析
        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        # 使用 CodeParser 解析返回的代码
        reflection = CodeParser.parse_code(block=None, text=rsp)
        # 返回改进后的实现（反思）
        return reflection

    async def run(
        self,
        user_requirement: str,
        plan_status: str = "",
        tool_info: str = "",
        working_memory: list[Message] = None,
        use_reflection: bool = False,
        memory: list[Message] = None,
        **kwargs,
    ) -> str:
        # 结构化的提示，包含用户需求、计划状态和工具信息
        structual_prompt = STRUCTUAL_PROMPT.format(
            user_requirement=user_requirement,
            plan_status=plan_status,
            tool_info=tool_info,
        )

        # 如果工作记忆和内存为空，则初始化为空列表
        working_memory = working_memory or []
        memory = memory or []
        context = self.llm.format_msg(memory + [Message(content=structual_prompt, role="user")] + working_memory)

        # 根据是否需要反思，选择不同的处理方式
        if use_reflection:
            # 如果需要反思，调用 _debug_with_reflection 方法
            code = await self._debug_with_reflection(context=context, working_memory=working_memory)
        else:
            # 否则直接调用 LLM 进行分析
            rsp = await self.llm.aask(context, system_msgs=[INTERPRETER_SYSTEM_MSG], **kwargs)
            # 解析返回的代码
            code = CodeParser.parse_code(text=rsp, lang="python")

        return code


class CheckData(Action):
    async def run(self, plan: Plan) -> dict:
        # 获取已完成的任务列表
        finished_tasks = plan.get_finished_tasks()
        # 去除每个任务代码中的注释
        code_written = [remove_comments(task.code) for task in finished_tasks]
        # 将所有代码拼接成一个字符串
        code_written = "\n\n".join(code_written)
        # 创建检查数据的提示
        prompt = CHECK_DATA_PROMPT.format(code_written=code_written)
        # 调用 LLM 获取回应
        rsp = await self._aask(prompt)
        # 解析返回的代码
        code = CodeParser.parse_code(text=rsp)
        return code
