# -*- encoding: utf-8 -*-
"""
@Date    :   2023/11/20 11:24:03
@Author  :   orange-crow
@File    :   plan.py
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Tuple

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.schema import Message, Plan, Task
from metagpt.strategy.task_type import TaskType
from metagpt.utils.common import CodeParser

PROMPT_TEMPLATE: str = """
# Context:
{context}
# Available Task Types:
{task_type_desc}
# Task:
Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan consists of one to {max_tasks} tasks.
If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. Give the whole plan unless instructed to modify only one task of the plan.
If you encounter errors on the current task, revise and output the current single task only.
Output a list of jsons following the format:
```json
[
    {{
        "task_id": str = "unique identifier for a task in plan, can be an ordinal",
        "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
        "instruction": "what you should do in this task, one short phrase or sentence.",
        "task_type": "type of this task, should be one of Available Task Types.",
    }},
    ...
]
```
"""


class WritePlan(Action):
    async def run(self, context: list[Message], max_tasks: int = 5) -> str:
        # 生成任务类型的描述，每种任务类型的名称和描述
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])

        # 创建 prompt，传入上下文、最大任务数和任务类型描述
        prompt = PROMPT_TEMPLATE.format(
            context="\n".join([str(ct) for ct in context]), max_tasks=max_tasks, task_type_desc=task_type_desc
        )

        # 调用 LLM 获取响应
        rsp = await self._aask(prompt)

        # 解析返回的代码
        rsp = CodeParser.parse_code(text=rsp)

        return rsp


def update_plan_from_rsp(rsp: str, current_plan: Plan):
    # 解析响应内容为 JSON 对象
    rsp = json.loads(rsp)

    # 根据响应内容创建任务对象列表
    tasks = [Task(**task_config) for task_config in rsp]

    if len(tasks) == 1 or tasks[0].dependent_task_ids:
        # 如果任务列表中只有一个任务，或者第一个任务有依赖任务
        if tasks[0].dependent_task_ids and len(tasks) > 1:
            # 如果第一个任务有依赖任务并且生成了多个任务，说明生成的任务不是完整的计划
            # 在这种情况下，只支持一次更新一个任务
            logger.warning(
                "当前计划只会接受第一个生成的任务，如果生成的任务不完整"
            )

        # 处理单个任务的情况
        if current_plan.has_task_id(tasks[0].task_id):
            # 如果当前计划已有任务，则替换现有任务
            current_plan.replace_task(
                tasks[0].task_id, tasks[0].dependent_task_ids, tasks[0].instruction, tasks[0].assignee
            )
        else:
            # 如果当前计划没有该任务，则添加任务
            current_plan.append_task(
                tasks[0].task_id, tasks[0].dependent_task_ids, tasks[0].instruction, tasks[0].assignee
            )

    else:
        # 如果有多个任务，直接将任务添加到当前计划中
        current_plan.add_tasks(tasks)


def precheck_update_plan_from_rsp(rsp: str, current_plan: Plan) -> Tuple[bool, str]:
    # 复制当前计划，避免直接修改原计划
    temp_plan = deepcopy(current_plan)
    try:
        # 尝试从响应更新计划
        update_plan_from_rsp(rsp, temp_plan)
        return True, ""  # 更新成功，返回成功标志
    except Exception as e:
        return False, str(e)  # 更新失败，返回失败标志和错误信息