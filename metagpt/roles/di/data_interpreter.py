from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, model_validator

# from metagpt.actions.di.ask_review import ReviewConst
from metagpt.actions.di.execute_nb_code import ExecuteNbCode
from metagpt.actions.di.write_analysis_code import CheckData, WriteAnalysisCode
from metagpt.logs import logger
from metagpt.prompts.di.write_analysis_code import DATA_INFO
from metagpt.roles import Role
from metagpt.schema import Message, Task, TaskResult
from metagpt.strategy.task_type import TaskType
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.utils.common import CodeParser
from metagpt.utils.report import ThoughtReporter

REACT_THINK_PROMPT = """
# User Requirement
{user_requirement}
# Context
{context}

Output a json following the format:
```json
{{
    "thoughts": str = "Thoughts on current situation, reflect on how you should proceed to fulfill the user requirement",
    "state": bool = "Decide whether you need to take more actions to complete the user requirement. Return true if you think so. Return false if you think the requirement has been completely fulfilled."
}}
```
"""


class DataInterpreter(Role):
    name: str = "David"  # 角色名称
    profile: str = "DataInterpreter"  # 角色简介
    auto_run: bool = True  # 是否自动运行
    use_plan: bool = True  # 是否使用计划
    use_reflection: bool = False  # 是否使用反思机制
    execute_code: ExecuteNbCode = Field(default_factory=ExecuteNbCode, exclude=True)  # 执行代码的对象
    tools: list[str] = []  # 使用的工具列表，特殊符号 ["<all>"] 表示使用所有注册的工具
    tool_recommender: ToolRecommender = None  # 工具推荐器
    react_mode: Literal["plan_and_act", "react"] = "plan_and_act"  # 反应模式，可以是“plan_and_act”或“react”
    max_react_loop: int = 10  # 反应模式下最大循环次数
    user_requirement: str = ""  # 用户需求

    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "Interpreter":
        """设置计划和工具推荐器"""
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop, auto_run=self.auto_run)
        self.use_plan = (
            self.react_mode == "plan_and_act"
        )  # 根据反应模式设置是否使用计划
        if self.tools and not self.tool_recommender:
            self.tool_recommender = BM25ToolRecommender(tools=self.tools)  # 如果没有工具推荐器，则使用BM25推荐器
        self.set_actions([WriteAnalysisCode])  # 设置行动
        self._set_state(0)  # 设置状态
        return self

    @property
    def working_memory(self):
        """获取工作记忆"""
        return self.rc.working_memory

    async def _think(self) -> bool:
        """在 'react' 模式下非常有用，使用LLM来决定是否以及做什么下一步行动"""
        self.user_requirement = self.get_memories()[-1].content  # 获取最新的用户需求
        context = self.working_memory.get()  # 获取工作记忆

        if not context:
            # 如果没有上下文，则需要立刻执行一个行动
            self.working_memory.add(self.get_memories()[0])  # 将用户需求添加到工作记忆
            self._set_state(0)
            return True

        # 生成反应的提示文本
        prompt = REACT_THINK_PROMPT.format(user_requirement=self.user_requirement, context=context)
        async with ThoughtReporter(enable_llm_stream=True):
            rsp = await self.llm.aask(prompt)  # 请求LLM生成回应
        rsp_dict = json.loads(CodeParser.parse_code(text=rsp))  # 解析LLM的回应
        self.working_memory.add(Message(content=rsp_dict["thoughts"], role="assistant"))  # 将LLM的思考内容加入工作记忆
        need_action = rsp_dict["state"]  # 判断是否需要执行行动
        self._set_state(0) if need_action else self._set_state(-1)  # 根据是否需要行动设置状态

        return need_action  # 返回是否需要执行行动

    async def _act(self) -> Message:
        """在 'react' 模式下非常有用，返回符合 Role._act 接口的消息"""
        code, _, _ = await self._write_and_exec_code()  # 写入并执行代码
        return Message(content=code, role="assistant", sent_from=self._setting, cause_by=WriteAnalysisCode)

    async def _plan_and_act(self) -> Message:
        """在 'plan_and_act' 模式下，首先执行计划然后执行行动"""
        self._set_state(0)
        try:
            rsp = await super()._plan_and_act()  # 调用父类的方法执行计划并行动
            await self.execute_code.terminate()  # 终止执行代码
            return rsp
        except Exception as e:
            await self.execute_code.terminate()  # 捕获异常并终止代码执行
            raise e

    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """在 'plan_and_act' 模式下，将任务结果封装在 TaskResult 中供审查和确认"""
        code, result, is_success = await self._write_and_exec_code()  # 写入并执行代码
        task_result = TaskResult(code=code, result=result, is_success=is_success)  # 包装任务结果
        return task_result

    async def _write_and_exec_code(self, max_retry: int = 3):
        """写入并执行代码，最多尝试指定次数"""
        counter = 0
        success = False

        # 获取计划信息
        plan_status = self.planner.get_plan_status() if self.use_plan else ""

        # 获取工具信息
        if self.tool_recommender:
            context = (
                self.working_memory.get()[-1].content if self.working_memory.get() else ""
            )  # 获取最新的思考内容作为上下文
            plan = self.planner.plan if self.use_plan else None
            tool_info = await self.tool_recommender.get_recommended_tool_info(context=context, plan=plan)  # 获取推荐的工具信息
        else:
            tool_info = ""

        # 检查数据
        await self._check_data()

        while not success and counter < max_retry:
            ### 写入代码 ###
            code, cause_by = await self._write_code(counter, plan_status, tool_info)

            self.working_memory.add(Message(content=code, role="assistant", cause_by=cause_by))  # 将生成的代码添加到工作记忆

            ### 执行代码 ###
            result, success = await self.execute_code.run(code)
            print(result)

            self.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))  # 将执行结果添加到工作记忆

            ### 处理执行结果 ###
            counter += 1

        return code, result, success  # 返回生成的代码、结果和成功标志

    async def _write_code(
        self,
        counter: int,
        plan_status: str = "",
        tool_info: str = "",
    ):
        """写入代码，使用反思机制仅在第一次之后"""
        todo = self.rc.todo  # todo是WriteAnalysisCode
        logger.info(f"准备进行 {todo.name}")
        use_reflection = counter > 0 and self.use_reflection  # 第一次尝试之后才使用反思机制

        code = await todo.run(
            user_requirement=self.user_requirement,
            plan_status=plan_status,
            tool_info=tool_info,
            working_memory=self.working_memory.get(),
            use_reflection=use_reflection,
        )  # 执行写代码的任务

        return code, todo  # 返回生成的代码和任务对象

    async def _check_data(self):
        """检查数据是否需要更新"""
        if (
            not self.use_plan
            or not self.planner.plan.get_finished_tasks()
            or self.planner.plan.current_task.task_type
            not in [
                TaskType.DATA_PREPROCESS.type_name,
                TaskType.FEATURE_ENGINEERING.type_name,
                TaskType.MODEL_TRAIN.type_name,
            ]
        ):
            return
        logger.info("检查更新的数据")
        code = await CheckData().run(self.planner.plan)  # 获取检查数据的代码
        if not code.strip():
            return
        result, success = await self.execute_code.run(code)  # 执行代码检查数据
        if success:
            print(result)
            data_info = DATA_INFO.format(info=result)  # 格式化数据结果信息
            self.working_memory.add(Message(content=data_info, role="user", cause_by=CheckData))  # 将数据结果加入工作记忆