import json

from pydantic import Field

from metagpt.logs import logger
from metagpt.prompts.di.swe_agent import (
    CURRENT_BASH_STATE,
    MINIMAL_EXAMPLE,
    NEXT_STEP_TEMPLATE,
)
from metagpt.roles.di.role_zero import RoleZero
from metagpt.schema import Message
from metagpt.tools.libs.git import git_create_pull
from metagpt.tools.libs.terminal import Bash


class SWEAgent(RoleZero):
    name: str = "Swen"  # 代理的名称
    profile: str = "Issue Solver"  # 代理的角色简介
    goal: str = "Resolve GitHub issue or bug in any existing codebase"  # 代理的目标：解决 GitHub 问题或代码中的 bug
    _instruction: str = NEXT_STEP_TEMPLATE  # 指令模板
    tools: list[str] = [
        "Bash",  # 使用 Bash 命令
        "Browser:goto,scroll",  # 浏览器工具：跳转、滚动
        "RoleZero",  # 使用 RoleZero 工具
        "git_create_pull",  # 使用 Git 创建拉取请求
    ]
    terminal: Bash = Field(default_factory=Bash, exclude=True)  # 终端实例，排除在序列化外
    output_diff: str = ""  # 存储输出的 Git diff
    max_react_loop: int = 40  # 最大反应循环次数
    run_eval: bool = False  # 是否运行评估

    async def _think(self) -> bool:
        """
        让代理思考：格式化指令并执行父类的思考方法
        """
        await self._format_instruction()  # 格式化指令
        res = await super()._think()  # 调用父类的思考方法
        return res

    def _update_tool_execution(self):
        """
        更新工具执行映射：将 Bash 和 git 创建拉取请求与相应的执行函数绑定
        """
        self.tool_execution_map.update(
            {
                "Bash.run": self.terminal.run,  # 将 Bash 工具的运行映射到终端执行
                "git_create_pull": git_create_pull,  # 将 git 创建拉取请求映射到相应的函数
            }
        )

    async def _format_instruction(self):
        """
        格式化 SWE 代理的指令消息。
        执行终端中的 "state" 命令，解析其输出为 JSON，并使用它来格式化 `_instruction` 模板。
        """
        state_output = await self.terminal.run("state")  # 执行 "state" 命令获取状态输出
        bash_state = json.loads(state_output)  # 将输出解析为 JSON 格式
        self.cmd_prompt_current_state = CURRENT_BASH_STATE.format(**bash_state).strip()  # 格式化当前状态并去除多余空白

    async def _act(self) -> Message:
        """
        执行代理的动作，调用父类的动作方法。
        如果启用评估，则解析命令进行评估。
        """
        message = await super()._act()  # 调用父类的执行方法
        if self.run_eval:  # 如果启用评估
            self._parse_commands_for_eval()  # 解析评估命令
        return message

    async def _parse_commands_for_eval(self):
        """
        解析命令并根据解析结果处理动作。
        如果发现 "submit" 操作，则生成补丁并通过 `git diff` 存储更改。
        将清理后的补丁存储在 `output_diff` 中。如果出现异常，则记录错误信息。
        该函数专门为 SWE 基准评估添加。
        """
        # 如果 todo 切换为 None，表示这是最后一轮反应，SWE-Agent 将停止。使用 git diff 存储任何已做更改。
        if not self.rc.todo:
            from metagpt.tools.swe_agent_commands.swe_agent_utils import extract_patch

            try:
                diff_output = await self.terminal.run("git diff --cached")  # 获取 git diff 输出
                clear_diff = extract_patch(diff_output)  # 提取补丁
                logger.info(f"Diff output: \n{clear_diff}")  # 输出日志
                if clear_diff:  # 如果有清理后的差异
                    self.output_diff = clear_diff  # 存储清理后的补丁
            except Exception as e:
                logger.error(f"Error during submission: {e}")  # 记录提交过程中出现的错误

    def _retrieve_experience(self) -> str:
        """
        获取代理的经验，这里返回一个最小的示例
        """
        return MINIMAL_EXAMPLE  # 返回最小示例
