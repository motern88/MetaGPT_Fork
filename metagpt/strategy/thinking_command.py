from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from metagpt.environment.mgx.mgx_env import MGXEnv
from metagpt.memory import Memory
from metagpt.roles import Role
from metagpt.schema import Message


class CommandDef(BaseModel):
    """定义命令的结构，包括名称、调用签名和描述"""
    name: str  # 命令名称
    signature: str = ""  # 命令的调用格式
    desc: str = ""  # 命令的描述


class Command(Enum):
    """定义可用的命令集，分为任务规划命令、环境交互命令和通用命令"""

    # 任务规划相关的命令
    APPEND_TASK = CommandDef(
        name="append_task",
        signature="append_task(task_id: str, dependent_task_ids: list[str], instruction: str, assignee: str)",
        desc="在现有任务序列末尾追加一个新的任务 task_id。如果 dependent_task_ids 非空，则新任务依赖于这些任务。",
    )
    RESET_TASK = CommandDef(
        name="reset_task",
        signature="reset_task(task_id: str)",
        desc="根据 task_id 重置任务（将 Task.is_finished 设置为 False，并请求重新执行）。同时重置所有依赖该任务的其他任务。",
    )
    REPLACE_TASK = CommandDef(
        name="replace_task",
        signature="replace_task(task_id: str, new_dependent_task_ids: list[str], new_instruction: str, new_assignee: str)",
        desc="替换指定的任务（可以是当前任务），并重置所有依赖该任务的任务。",
    )
    FINISH_CURRENT_TASK = CommandDef(
        name="finish_current_task",
        signature="finish_current_task()",
        desc="标记当前任务为完成（Task.is_finished=True），并将当前任务切换到下一个任务。",
    )

    # 与环境交互的命令
    PUBLISH_MESSAGE = CommandDef(
        name="publish_message",
        signature="publish_message(content: str, send_to: str)",
        desc="向团队成员发布消息，send_to 参数应填入成员名称。你可以复制原始内容或增加上游的额外信息，"
             "确保必要信息（如路径、链接、环境、编程语言、框架、需求、约束等）完整传递。",
    )
    REPLY_TO_HUMAN = CommandDef(
        name="reply_to_human",
        signature="reply_to_human(content: str)",
        desc="向人类用户回复消息。适用于当你已清楚问题并有明确答案时。",
    )
    ASK_HUMAN = CommandDef(
        name="ask_human",
        signature="ask_human(question: str)",
        desc="当无法完成当前任务或遇到不确定情况时，使用此命令。"
             "应包含对当前情况的简要描述，并以清晰简洁的提问结尾。",
    )

    # 通用命令
    PASS = CommandDef(
        name="pass",
        signature="pass",
        desc="跳过操作，不执行任何动作。适用于计划无需更新，或消息无需转发，"
             "或者想等待更多信息再做决策的情况。",
    )

    @property
    def cmd_name(self):
        """获取命令名称"""
        return self.value.name


def prepare_command_prompt(commands: list[Command]) -> str:
    """
    生成命令提示信息，包含所有可用命令的描述

    :param commands: 需要展示的命令列表
    :return: 格式化的命令描述字符串
    """
    command_prompt = ""
    for i, command in enumerate(commands):
        command_prompt += f"{i+1}. {command.value.signature}:\n{command.value.desc}\n\n"
    return command_prompt


async def run_env_command(role: Role, cmd: list[dict], role_memory: Memory = None):
    """
    执行环境交互相关的命令，如发送消息、向用户提问等

    :param role: 当前执行命令的角色
    :param cmd: 命令字典，包含 command_name 和 args
    :param role_memory: 角色的记忆存储（可选）
    """
    if not isinstance(role.rc.env, MGXEnv):
        return

    if cmd["command_name"] == Command.PUBLISH_MESSAGE.cmd_name:
        role.publish_message(Message(**cmd["args"]))

    elif cmd["command_name"] == Command.ASK_HUMAN.cmd_name:
        # TODO: 角色记忆的操作不应出现在这里，考虑将其移动到 Role 类中
        role.rc.working_memory.add(Message(content=cmd["args"]["question"], role="assistant"))
        human_rsp = await role.rc.env.ask_human(sent_from=role, **cmd["args"])
        role.rc.working_memory.add(Message(content=human_rsp, role="user"))

    elif cmd["command_name"] == Command.REPLY_TO_HUMAN.cmd_name:
        # TODO: 需要考虑此消息是否应存入记忆
        await role.rc.env.reply_to_human(sent_from=role, **cmd["args"])


def run_plan_command(role: Role, cmd: list[dict]):
    """
    执行任务规划相关的命令，如添加任务、重置任务、完成任务等

    :param role: 当前执行命令的角色
    :param cmd: 命令字典，包含 command_name 和 args
    """
    if cmd["command_name"] == Command.APPEND_TASK.cmd_name:
        role.planner.plan.append_task(**cmd["args"])

    elif cmd["command_name"] == Command.RESET_TASK.cmd_name:
        role.planner.plan.reset_task(**cmd["args"])

    elif cmd["command_name"] == Command.REPLACE_TASK.cmd_name:
        role.planner.plan.replace_task(**cmd["args"])

    elif cmd["command_name"] == Command.FINISH_CURRENT_TASK.cmd_name:
        if role.planner.plan.is_plan_finished():
            return

        # 如果当前任务有结果，存入任务信息
        if role.task_result:
            role.planner.plan.current_task.update_task_result(task_result=role.task_result)

        # 标记当前任务为完成并切换到下一个任务
        role.planner.plan.finish_current_task()
        role.rc.working_memory.clear()


async def run_commands(role: Role, cmds: list[dict], role_memory: Memory = None):
    """
    依次执行多个命令，包括环境交互和任务规划

    :param role: 当前执行命令的角色
    :param cmds: 命令列表，每个命令为一个字典
    :param role_memory: 角色的记忆存储（可选）
    """
    print(*cmds, sep="\n")

    for cmd in cmds:
        await run_env_command(role, cmd, role_memory)  # 先执行环境交互命令
        run_plan_command(role, cmd)  # 然后执行任务规划命令

    # 如果任务计划已经完成，则更新角色状态
    if role.planner.plan.is_plan_finished():
        role._set_state(-1)
