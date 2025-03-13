from __future__ import annotations

from typing import Tuple

from metagpt.actions import Action
from metagpt.logs import get_human_input, logger
from metagpt.schema import Message, Plan


class ReviewConst:
    # 定义触发审查的关键词
    TASK_REVIEW_TRIGGER = "task"  # 任务审查触发关键词
    CODE_REVIEW_TRIGGER = "code"  # 代码审查触发关键词
    CONTINUE_WORDS = ["confirm", "continue", "c", "yes", "y"]  # 确认继续的关键词
    CHANGE_WORDS = ["change"]  # 更改的关键词
    EXIT_WORDS = ["exit"]  # 退出的关键词
    TASK_REVIEW_INSTRUCTION = (
        f"如果您想更改、添加、删除任务或合并任务，请说 '{CHANGE_WORDS[0]} task task_id 或当前任务，...(需要更改的内容)' "
        f"如果您确认当前任务的输出并希望继续，请输入：{CONTINUE_WORDS[0]}"
    )  # 任务审查指令
    CODE_REVIEW_INSTRUCTION = (
        f"如果您希望重写代码，请说 '{CHANGE_WORDS[0]} ...(您的修改建议)' "
        f"如果您希望保持原样，请输入：{CONTINUE_WORDS[0]} 或 {CONTINUE_WORDS[1]}"
    )  # 代码审查指令
    EXIT_INSTRUCTION = f"如果您想终止该过程，请输入：{EXIT_WORDS[0]}"  # 退出指令


class AskReview(Action):
    async def run(
        self, context: list[Message] = [], plan: Plan = None, trigger: str = ReviewConst.TASK_REVIEW_TRIGGER
    ) -> Tuple[str, bool]:
        # 如果提供了计划，输出当前整体计划
        if plan:
            logger.info("当前整体计划：")
            logger.info(
                "\n".join(
                    [f"{task.task_id}: {task.instruction}, 是否完成: {task.is_finished}" for task in plan.tasks]
                )
            )

        # 输出最新的上下文
        logger.info("最近的上下文：")
        latest_action = context[-1].cause_by if context and context[-1].cause_by else ""
        review_instruction = (
            ReviewConst.TASK_REVIEW_INSTRUCTION
            if trigger == ReviewConst.TASK_REVIEW_TRIGGER
            else ReviewConst.CODE_REVIEW_INSTRUCTION
        )  # 根据触发类型选择任务或代码审查指令
        prompt = (
            f"这是一个<{trigger}>审查。请审查来自 {latest_action} 的输出\n"
            f"{review_instruction}\n"
            f"{ReviewConst.EXIT_INSTRUCTION}\n"
            "请输入您的审查内容：\n"
        )

        rsp = await get_human_input(prompt)  # 获取用户输入

        # 如果输入是退出指令，则退出
        if rsp.lower() in ReviewConst.EXIT_WORDS:
            exit()

        # 确认是否继续的条件：输入内容必须是“confirm”、“continue”、“c”、“yes”或“y”，或包含“confirm”的句子
        confirmed = rsp.lower() in ReviewConst.CONTINUE_WORDS or ReviewConst.CONTINUE_WORDS[0] in rsp.lower()

        return rsp, confirmed  # 返回用户输入和是否确认继续
