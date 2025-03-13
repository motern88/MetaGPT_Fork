from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from metagpt.actions.di.execute_nb_code import ExecuteNbCode
from metagpt.actions.di.write_analysis_code import CheckData, WriteAnalysisCode
from metagpt.logs import logger
from metagpt.prompts.di.data_analyst import (
    CODE_STATUS,
    EXTRA_INSTRUCTION,
    TASK_TYPE_DESC,
)
from metagpt.prompts.di.role_zero import ROLE_INSTRUCTION
from metagpt.prompts.di.write_analysis_code import DATA_INFO
from metagpt.roles.di.role_zero import RoleZero
from metagpt.schema import Message, TaskResult
from metagpt.strategy.experience_retriever import ExpRetriever, KeywordExpRetriever
from metagpt.strategy.task_type import TaskType
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.tools.tool_registry import register_tool


# 定义 DataAnalyst 角色
@register_tool(include_functions=["write_and_exec_code"])
class DataAnalyst(RoleZero):
    name: str = "David"  # 角色名称
    profile: str = "DataAnalyst"  # 角色类型
    goal: str = "负责数据分析、机器学习、深度学习、网页浏览、爬取、搜索、终端操作、文档分析等任务。"  # 目标
    instruction: str = ROLE_INSTRUCTION + EXTRA_INSTRUCTION  # 角色指令
    task_type_desc: str = TASK_TYPE_DESC  # 任务类型描述

    # 可用工具列表
    tools: list[str] = [
        "Plan",
        "DataAnalyst",
        "RoleZero",
        "Browser",
        "Editor:write,read,similarity_search",
        "SearchEnhancedQA",
    ]
    # 自定义工具
    custom_tools: list[str] = ["web scraping", "Terminal", "Editor:write,read,similarity_search"]
    custom_tool_recommender: ToolRecommender = None  # 自定义工具推荐器
    experience_retriever: Annotated[ExpRetriever, Field(exclude=True)] = KeywordExpRetriever()  # 经验检索器

    use_reflection: bool = True  # 是否使用反思机制
    write_code: WriteAnalysisCode = Field(default_factory=WriteAnalysisCode, exclude=True)  # 代码编写工具
    execute_code: ExecuteNbCode = Field(default_factory=ExecuteNbCode, exclude=True)  # 代码执行工具

    # 校验方法：设置自定义工具推荐器
    @model_validator(mode="after")
    def set_custom_tool(self):
        if self.custom_tools and not self.custom_tool_recommender:
            self.custom_tool_recommender = BM25ToolRecommender(tools=self.custom_tools, force=True)

    # 更新工具执行映射
    def _update_tool_execution(self):
        self.tool_execution_map.update(
            {
                "DataAnalyst.write_and_exec_code": self.write_and_exec_code,
            }
        )

    # 编写并执行代码
    async def write_and_exec_code(self, instruction: str = ""):
        """
        为当前任务编写代码并在交互式 notebook 环境中执行。

        参数:
            instruction (可选, str): 任务的额外提示或注意事项，必须非常简洁，可以为空。默认为 ""。
        """
        if self.planner.plan:
            logger.info(f"当前任务 {self.planner.plan.current_task}")

        counter = 0  # 计数器
        success = False  # 代码执行是否成功
        await self.execute_code.init_code()  # 初始化代码执行环境

        # 获取计划信息
        if self.planner.current_task:
            # 移除任务结果以节省 token（因为已存入内存）
            plan_status = self.planner.get_plan_status(exclude=["task_result"])
            plan_status += f"\n额外任务指令: {instruction}"
        else:
            return "当前没有任务。请使用 Plan.append_task 添加任务。"

        # 获取工具信息
        if self.custom_tool_recommender:
            plan = self.planner.plan
            fixed = ["Terminal"] if "Terminal" in self.custom_tools else None
            tool_info = await self.custom_tool_recommender.get_recommended_tool_info(fixed=fixed, plan=plan)
        else:
            tool_info = ""

        # 检查数据
        await self._check_data()

        while not success and counter < 3:
            ### 编写代码 ###
            logger.info("开始编写代码")
            use_reflection = counter > 0 and self.use_reflection  # 仅在首次尝试失败后使用反思机制

            code = await self.write_code.run(
                user_requirement=self.planner.plan.goal,  # 用户需求
                plan_status=plan_status,  # 计划状态
                tool_info=tool_info,  # 工具信息
                working_memory=self.rc.working_memory.get(),  # 当前工作记忆
                use_reflection=use_reflection,  # 是否使用反思
                memory=self.rc.memory.get(self.memory_k),  # 历史记忆
            )
            self.rc.working_memory.add(Message(content=code, role="assistant", cause_by=WriteAnalysisCode))

            ### 执行代码 ###
            result, success = await self.execute_code.run(code)
            print(result)

            self.rc.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))

            ### 处理执行结果 ###
            counter += 1
            if success:
                task_result = TaskResult(code=code, result=result, is_success=success)
                self.planner.current_task.update_task_result(task_result)

        status = "成功" if success else "失败"
        output = CODE_STATUS.format(code=code, status=status, result=result)
        if success:
            output += "代码已成功执行。"
        self.rc.working_memory.clear()
        return output

    # 检查数据是否更新
    async def _check_data(self):
        if not self.planner.plan.get_finished_tasks() or self.planner.plan.current_task.task_type not in [
            TaskType.DATA_PREPROCESS.type_name,
            TaskType.FEATURE_ENGINEERING.type_name,
            TaskType.MODEL_TRAIN.type_name,
        ]:
            return
        logger.info("检查更新后的数据")
        code = await CheckData().run(self.planner.plan)
        if not code.strip():
            return
        result, success = await self.execute_code.run(code)
        if success:
            print(result)
            data_info = DATA_INFO.format(info=result)
            self.rc.working_memory.add(Message(content=data_info, role="user", cause_by=CheckData))

    async def _run_special_command(self, cmd) -> str:
        """需要特别检查或解析的命令。"""
        # TODO: 与 Engineer2._run_special_command 重复，考虑去重

        # 在结束之前完成当前任务
        command_output = ""

        # 如果命令是 "end" 且计划尚未完成，则结束所有任务
        if cmd["command_name"] == "end" and not self.planner.plan.is_plan_finished():
            self.planner.plan.finish_all_tasks()  # 完成所有任务
            command_output += "所有任务已完成。\n"  # 输出任务完成信息

        # 调用父类的 _run_special_command 方法并返回输出
        command_output += await super()._run_special_command(cmd)

        return command_output  # 返回最终命令输出
