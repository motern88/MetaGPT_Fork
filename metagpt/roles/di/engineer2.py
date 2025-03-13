from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from metagpt.logs import logger

# from metagpt.actions.write_code_review import ValidateAndRewriteCode
from metagpt.prompts.di.engineer2 import (
    CURRENT_STATE,
    ENGINEER2_INSTRUCTION,
    WRITE_CODE_PROMPT,
    WRITE_CODE_SYSTEM_PROMPT,
)
from metagpt.roles.di.role_zero import RoleZero
from metagpt.schema import UserMessage
from metagpt.strategy.experience_retriever import ENGINEER_EXAMPLE
from metagpt.tools.libs.cr import CodeReview
from metagpt.tools.libs.deployer import Deployer
from metagpt.tools.libs.git import git_create_pull
from metagpt.tools.libs.image_getter import ImageGetter
from metagpt.tools.libs.terminal import Terminal
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import CodeParser, awrite
from metagpt.utils.report import EditorReporter


@register_tool(include_functions=["write_new_code"])
class Engineer2(RoleZero):
    name: str = "Alex"
    profile: str = "Engineer"
    goal: str = "负责游戏、应用程序、Web开发和部署。"
    instruction: str = ENGINEER2_INSTRUCTION
    terminal: Terminal = Field(default_factory=Terminal, exclude=True)
    deployer: Deployer = Field(default_factory=Deployer, exclude=True)
    tools: list[str] = [
        "Plan",
        "Editor",
        "RoleZero",
        "Terminal:run_command",
        "Browser:goto,scroll",
        "git_create_pull",
        "SearchEnhancedQA",
        "Engineer2",
        "CodeReview",
        "ImageGetter",
        "Deployer",
    ]
    # SWE Agent 参数
    run_eval: bool = False  # 是否启用评估模式
    output_diff: str = ""  # 输出的差异
    max_react_loop: int = 40  # 最大反应循环次数

    async def _think(self) -> bool:
        await self._format_instruction()  # 格式化指令
        res = await super()._think()  # 调用父类的思考方法
        return res

    async def _format_instruction(self):
        """
        显示当前终端和编辑器的状态。
        这些信息将动态添加到命令提示符中。
        """
        current_directory = (await self.terminal.run_command("pwd")).strip()  # 获取当前目录
        self.editor._set_workdir(current_directory)  # 设置编辑器的工作目录
        state = {
            "editor_open_file": self.editor.current_file,
            "current_directory": current_directory,
        }
        self.cmd_prompt_current_state = CURRENT_STATE.format(**state).strip()  # 格式化当前状态

    def _update_tool_execution(self):
        # 验证和重写代码
        cr = CodeReview()
        image_getter = ImageGetter()
        self.exclusive_tool_commands.append("Engineer2.write_new_code")
        if self.run_eval is True:
            # 启用评估模式时更新工具映射
            self.tool_execution_map.update(
                {
                    "git_create_pull": git_create_pull,
                    "Engineer2.write_new_code": self.write_new_code,
                    "ImageGetter.get_image": image_getter.get_image,
                    "CodeReview.review": cr.review,
                    "CodeReview.fix": cr.fix,
                    "Terminal.run_command": self._eval_terminal_run,
                    "RoleZero.ask_human": self._end,
                    "RoleZero.reply_to_human": self._end,
                    "Deployer.deploy_to_public": self._deploy_to_public,
                }
            )
        else:
            # 默认工具映射
            self.tool_execution_map.update(
                {
                    "git_create_pull": git_create_pull,
                    "Engineer2.write_new_code": self.write_new_code,
                    "ImageGetter.get_image": image_getter.get_image,
                    "CodeReview.review": cr.review,
                    "CodeReview.fix": cr.fix,
                    "Terminal.run_command": self.terminal.run_command,
                    "Deployer.deploy_to_public": self._deploy_to_public,
                }
            )

    def _retrieve_experience(self) -> str:
        return ENGINEER_EXAMPLE  # 返回工程师的经验

    async def write_new_code(self, path: str, file_description: str = "") -> str:
        """编写新的代码文件。

        参数:
            path (str): 要创建的文件的绝对路径。
            file_description (可选，str): 文件内容的简要描述和重要说明，必须非常简洁，可以为空。默认为""。
        """
        # 如果路径不是绝对路径，尝试使用编辑器的工作目录来修正路径。
        path = self.editor._try_fix_path(path)
        plan_status, _ = self._get_plan_status()  # 获取计划状态
        prompt = WRITE_CODE_PROMPT.format(
            user_requirement=self.planner.plan.goal,
            plan_status=plan_status,
            file_path=path,
            file_description=file_description,
            file_name=os.path.basename(path),
        )
        # 有时工程师会重复最后一条命令进行响应。
        # 用手动提示替换最后一条命令，指导工程师编写新代码。
        memory = self.rc.memory.get(self.memory_k)[:-1]
        context = self.llm.format_msg(memory + [UserMessage(content=prompt)])

        async with EditorReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "code", "filename": Path(path).name, "src_path": path}, "meta")
            rsp = await self.llm.aask(context, system_msgs=[WRITE_CODE_SYSTEM_PROMPT])
            code = CodeParser.parse_code(text=rsp)
            await awrite(path, code)
            await reporter.async_report(path, "path")

        # TODO: 考虑添加行号以便编辑。
        return f"文件 {path} 已成功创建，内容如下：\n{code}"

    async def _deploy_to_public(self, dist_dir):
        """修复dist_dir路径为绝对路径后进行部署。
        参数:
            dist_dir (str): Web项目运行构建后的dist目录，必须是绝对路径。
        """
        # 尝试使用编辑器的工作目录修正路径。
        if not Path(dist_dir).is_absolute():
            default_dir = self.editor._try_fix_path(dist_dir)
            if not default_dir.exists():
                raise ValueError("dist_dir 必须是绝对路径。")
            dist_dir = default_dir
        return await self.deployer.deploy_to_public(dist_dir)

    async def _eval_terminal_run(self, cmd):
        """修改命令pull/push/commit为结束命令。"""
        if any([cmd_key_word in cmd for cmd_key_word in ["pull", "push", "commit"]]):
            # 工程师在修复bug后尝试提交代码，从而结束修复过程。
            logger.info(f"Engineer2 使用命令:{cmd}\n当前测试案例已完成。")
            # 设置 self.rc.todo 为 None 来停止工程师。
            self._set_state(-1)
        else:
            command_output = await self.terminal.run_command(cmd)
        return command_output

    async def _end(self):
        if not self.planner.plan.is_plan_finished():
            self.planner.plan.finish_all_tasks()  # 完成所有任务
        return await super()._end()  # 调用父类的结束方法