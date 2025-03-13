from __future__ import annotations

from typing import Annotated

from pydantic import Field

from metagpt.actions.di.run_command import RunCommand
from metagpt.const import TEAMLEADER_NAME
from metagpt.prompts.di.role_zero import QUICK_THINK_TAG
from metagpt.prompts.di.team_leader import (
    FINISH_CURRENT_TASK_CMD,
    TL_INFO,
    TL_INSTRUCTION,
    TL_THOUGHT_GUIDANCE,
)
from metagpt.roles.di.role_zero import RoleZero
from metagpt.schema import AIMessage, Message, UserMessage
from metagpt.strategy.experience_retriever import ExpRetriever, SimpleExpRetriever
from metagpt.tools.tool_registry import register_tool


@register_tool(include_functions=["publish_team_message"])
class TeamLeader(RoleZero):
    name: str = TEAMLEADER_NAME  # 设置 TeamLeader 的名称
    profile: str = "Team Leader"  # 角色简介：团队负责人
    goal: str = "Manage a team to assist users"  # 目标：管理团队帮助用户
    thought_guidance: str = TL_THOUGHT_GUIDANCE  # 思考指导：帮助团队负责人思考
    max_react_loop: int = 3  # 最大反应循环次数：团队负责人每次只能反应最多三次

    tools: list[str] = ["Plan", "RoleZero", "TeamLeader"]  # 可用的工具：计划、RoleZero、TeamLeader

    experience_retriever: Annotated[ExpRetriever, Field(exclude=True)] = SimpleExpRetriever()  # 经验检索器，用于获取经验

    use_summary: bool = False  # 是否使用总结

    def _update_tool_execution(self):
        """
        更新工具执行映射，将工具名称映射到实际执行函数。
        """
        self.tool_execution_map.update(
            {
                "TeamLeader.publish_team_message": self.publish_team_message,  # 将 publish_team_message 映射到对应的函数
                "TeamLeader.publish_message": self.publish_team_message,  # 别名，使用 publish_message 时调用 publish_team_message
            }
        )

    def _get_team_info(self) -> str:
        """
        获取团队信息，返回所有角色的名称、简介和目标。
        """
        if not self.rc.env:  # 如果环境不存在，返回空字符串
            return ""
        team_info = ""
        for role in self.rc.env.roles.values():  # 遍历环境中的所有角色
            team_info += f"{role.name}: {role.profile}, {role.goal}\n"  # 将每个角色的信息拼接为字符串
        return team_info  # 返回团队信息字符串

    def _get_prefix(self) -> str:
        """
        获取前缀信息，结合角色信息和团队信息
        """
        role_info = super()._get_prefix()  # 获取角色信息的前缀
        team_info = self._get_team_info()  # 获取团队信息
        return TL_INFO.format(role_info=role_info, team_info=team_info)  # 格式化并返回包含角色和团队信息的前缀

    async def _think(self) -> bool:
        """
        思考方法：格式化团队信息并调用父类的思考方法
        """
        self.instruction = TL_INSTRUCTION.format(team_info=self._get_team_info())  # 格式化指令
        return await super()._think()  # 调用父类的思考方法并返回结果

    def publish_message(self, msg: Message, send_to="no one"):
        """
        发布消息，重写 Role.publish_message。如果在 Role.run 内部调用，默认不发送给任何人；
        如果动态调用，则发送给指定的角色。
        """
        if not msg:  # 如果没有消息，直接返回
            return
        if not self.rc.env:  # 如果环境不存在，不能发布消息
            return
        if msg.cause_by != QUICK_THINK_TAG:  # 如果消息不是快速思考触发的
            msg.send_to = send_to  # 设置发送目标
        self.rc.env.publish_message(msg, publicer=self.profile)  # 在环境中发布消息

    def publish_team_message(self, content: str, send_to: str):
        """
        向团队成员发布消息，使用成员名称填充 send_to 参数。
        你可能会复制完整的原始内容或添加上游的附加信息。这将使团队成员开始工作。
        不要遗漏任何必要的信息，如路径、链接、环境、编程语言、框架、需求、约束等，因为你是团队成员唯一的信息来源。
        """
        self._set_state(-1)  # 每次发布消息时，暂停等待响应
        if send_to == self.name:  # 如果发送目标是自己，避免发送消息给自己
            return
        # 指定外部的 send_to 参数，覆盖默认的 "no one" 值。使用 UserMessage，因为来自自己的消息像是用户请求其他人处理。
        self.publish_message(
            UserMessage(content=content, sent_from=self.name, send_to=send_to, cause_by=RunCommand), send_to=send_to
        )

    def finish_current_task(self):
        """
        完成当前任务并更新内存，标记任务完成。
        """
        self.planner.plan.finish_current_task()  # 完成当前任务
        self.rc.memory.add(AIMessage(content=FINISH_CURRENT_TASK_CMD))  # 将任务完成的指令添加到内存中
