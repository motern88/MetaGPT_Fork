from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field

from metagpt.actions.di.ask_review import AskReview, ReviewConst
from metagpt.actions.di.write_plan import (
    WritePlan,
    precheck_update_plan_from_rsp,
    update_plan_from_rsp,
)
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.schema import Message, Plan, Task, TaskResult
from metagpt.strategy.task_type import TaskType
from metagpt.utils.common import remove_comments

STRUCTURAL_CONTEXT = """
## User Requirement
{user_requirement}
## Context
{context}
## Current Plan
{tasks}
## Current Task
{current_task}
"""

PLAN_STATUS = """
## Finished Tasks
### code
```python
{code_written}
```

### execution result
{task_results}

## Current Task
{current_task}

## Finished Section of Current Task
### code
```python
{current_task_code}
```
### execution result
{current_task_result}

## Task Guidance
Write code for the incomplete sections of 'Current Task'. And avoid duplicating code from 'Finished Tasks' and 'Finished Section of Current Task', such as repeated import of packages, reading data, etc.
Specifically, {guidance}
"""


class Planner(BaseModel):
    """规划器类，负责处理任务的计划、更新、确认和执行过程。"""

    plan: Plan  # 计划对象
    working_memory: Memory = Field(
        default_factory=Memory
    )  # 工作内存，用于处理每个任务，任务完成后会清除
    auto_run: bool = False  # 是否自动运行

    def __init__(self, goal: str = "", plan: Plan = None, **kwargs):
        """初始化规划器，若没有指定计划，则根据目标创建一个新的计划。

        参数:
            goal (str): 任务目标。
            plan (Plan): 已存在的计划。
        """
        plan = plan or Plan(goal=goal)
        super().__init__(plan=plan, **kwargs)

    @property
    def current_task(self):
        """返回当前任务"""
        return self.plan.current_task

    @property
    def current_task_id(self):
        """返回当前任务的ID"""
        return self.plan.current_task_id

    async def update_plan(self, goal: str = "", max_tasks: int = 3, max_retries: int = 3):
        """更新计划，重新生成新的任务计划。

        参数:
            goal (str): 新的目标。
            max_tasks (int): 最大任务数。
            max_retries (int): 最大重试次数。
        """
        if goal:
            self.plan = Plan(goal=goal)

        plan_confirmed = False
        while not plan_confirmed:
            context = self.get_useful_memories()
            rsp = await WritePlan().run(context, max_tasks=max_tasks)
            self.working_memory.add(Message(content=rsp, role="assistant", cause_by=WritePlan))

            # 在请求确认之前预检查生成的计划
            is_plan_valid, error = precheck_update_plan_from_rsp(rsp, self.plan)
            if not is_plan_valid and max_retries > 0:
                error_msg = f"生成的计划无效，错误信息: {error}，尝试重新生成计划，记得生成整个计划或仅生成更改的任务"
                logger.warning(error_msg)
                self.working_memory.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
                max_retries -= 1
                continue

            _, plan_confirmed = await self.ask_review(trigger=ReviewConst.TASK_REVIEW_TRIGGER)

        update_plan_from_rsp(rsp=rsp, current_plan=self.plan)

        self.working_memory.clear()

    async def process_task_result(self, task_result: TaskResult):
        """处理任务执行结果。

        参数:
            task_result (TaskResult): 任务结果。
        """
        # 请求确认，用户可以拒绝并更改计划中的任务
        review, task_result_confirmed = await self.ask_review(task_result)

        if task_result_confirmed:
            # 确认任务完成，并记录进展
            await self.confirm_task(self.current_task, task_result, review)

        elif "redo" in review:
            # 如果任务需要重做，用户可能给出了重做请求
            pass  # 如果任务重做，跳过确认结果

        else:
            # 根据用户反馈更新计划，处理已更改的任务
            await self.update_plan()

    async def ask_review(
        self,
        task_result: TaskResult = None,
        auto_run: bool = None,
        trigger: str = ReviewConst.TASK_REVIEW_TRIGGER,
        review_context_len: int = 5,
    ):
        """请求任务结果审查，审查者需要确认或要求更改。

        参数:
            task_result (TaskResult): 任务执行结果。
            auto_run (bool): 是否自动运行，默认使用类中的设置。
            trigger (str): 触发器，用于任务审查。
            review_context_len (int): 审查时上下文的长度。

        返回:
            tuple: 审查结果和确认标志。
        """
        auto_run = auto_run if auto_run is not None else self.auto_run
        if not auto_run:
            context = self.get_useful_memories()
            review, confirmed = await AskReview().run(
                context=context[-review_context_len:], plan=self.plan, trigger=trigger
            )
            if not confirmed:
                self.working_memory.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        confirmed = task_result.is_success if task_result else True
        return "", confirmed

    async def confirm_task(self, task: Task, task_result: TaskResult, review: str):
        """确认任务并根据任务结果更新计划。

        参数:
            task (Task): 当前任务。
            task_result (TaskResult): 任务结果。
            review (str): 审查反馈。
        """
        task.update_task_result(task_result=task_result)
        self.plan.finish_current_task()
        self.working_memory.clear()

        confirmed_and_more = (
            ReviewConst.CONTINUE_WORDS[0] in review.lower() and review.lower() not in ReviewConst.CONTINUE_WORDS[0]
        )  # "confirm, ... (更多内容，比如更改下游任务)"
        if confirmed_and_more:
            self.working_memory.add(Message(content=review, role="user", cause_by=AskReview))
            await self.update_plan()

    def get_useful_memories(self, task_exclude_field=None) -> list[Message]:
        """仅获取有用的记忆，减少上下文长度并提高性能"""
        user_requirement = self.plan.goal
        context = self.plan.context
        tasks = [task.dict(exclude=task_exclude_field) for task in self.plan.tasks]
        tasks = json.dumps(tasks, indent=4, ensure_ascii=False)
        current_task = self.plan.current_task.json() if self.plan.current_task else {}
        context = STRUCTURAL_CONTEXT.format(
            user_requirement=user_requirement, context=context, tasks=tasks, current_task=current_task
        )
        context_msg = [Message(content=context, role="user")]

        return context_msg + self.working_memory.get()

    def get_plan_status(self, exclude: List[str] = None) -> str:
        """准备并返回计划状态的组成部分"""
        exclude = exclude or []
        exclude_prompt = "omit here"
        finished_tasks = self.plan.get_finished_tasks()
        code_written = [remove_comments(task.code) for task in finished_tasks]
        code_written = "\n\n".join(code_written)
        task_results = [task.result for task in finished_tasks]
        task_results = "\n\n".join(task_results)
        task_type_name = self.current_task.task_type
        task_type = TaskType.get_type(task_type_name)
        guidance = task_type.guidance if task_type else ""

        # 将各个部分组合成一个完整的计划状态
        prompt = PLAN_STATUS.format(
            code_written=code_written if "code" not in exclude else exclude_prompt,
            task_results=task_results if "task_result" not in exclude else exclude_prompt,
            current_task=self.current_task.instruction,
            current_task_code=self.current_task.code if "code" not in exclude else exclude_prompt,
            current_task_result=self.current_task.result if "task_result" not in exclude else exclude_prompt,
            guidance=guidance,
        )

        return prompt
