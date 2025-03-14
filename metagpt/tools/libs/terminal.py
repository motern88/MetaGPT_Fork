import asyncio
import os
import re
from asyncio import Queue
from asyncio.subprocess import PIPE, STDOUT
from typing import Optional

from metagpt.config2 import Config
from metagpt.const import DEFAULT_WORKSPACE_ROOT, SWE_SETUP_PATH
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.report import END_MARKER_VALUE, TerminalReporter


@register_tool()
class Terminal:
    """
    终端工具类，用于执行终端命令。
    如果已有该类的实例，请勿重新初始化。
    如果需要在 Conda 环境中执行命令，建议使用 `execute_in_conda_env` 方法。
    """

    def __init__(self):
        self.shell_command = ["bash"]  # FIXME: 需要考虑 Windows 兼容性
        self.command_terminator = "\n"  # 命令终止符
        self.stdout_queue = Queue(maxsize=1000)  # 存储标准输出的队列
        self.observer = TerminalReporter()  # 观察者对象，用于报告终端输出
        self.process: Optional[asyncio.subprocess.Process] = None  # 终端进程
        # 禁止执行的命令列表，执行这些命令时将被替换为 "true"，并返回建议信息
        self.forbidden_commands = {
            "run dev": "请使用 Deployer.deploy_to_public 代替。",
            "serve ": "请使用 Deployer.deploy_to_public 代替。",
        }

    async def _start_process(self):
        """启动一个持久化的 shell 进程"""
        self.process = await asyncio.create_subprocess_exec(
            *self.shell_command,
            stdin=PIPE,
            stdout=PIPE,
            stderr=STDOUT,
            executable="bash",
            env=os.environ.copy(),
            cwd=DEFAULT_WORKSPACE_ROOT.absolute(),
        )
        await self._check_state()

    async def _check_state(self):
        """检查终端状态，例如当前目录，方便代理程序理解上下文"""
        output = await self.run_command("pwd")
        logger.info("终端当前目录:", output)

    async def run_command(self, cmd: str, daemon=False) -> str:
        """
        执行终端命令，并实时返回执行结果。
        终端会保持状态，例如当前目录，使得连续执行的命令具有上下文。

        参数：
            cmd (str): 要执行的命令。
            daemon (bool): 如果为 True，则命令在后台执行，主程序不会等待其完成。

        返回：
            str: 命令的输出结果。如果 `daemon=True`，则返回空字符串，需调用 `get_stdout_output` 获取输出。
        """
        if self.process is None:
            await self._start_process()

        output = ""
        # 检查并移除禁止的命令
        commands = re.split(r"\s*&&\s*", cmd)
        for cmd_name, reason in self.forbidden_commands.items():
            for index, command in enumerate(commands):
                if cmd_name in command:
                    output += f"无法执行 {command}。{reason}\n"
                    commands[index] = "true"  # 替换为 "true" 以跳过命令
        cmd = " && ".join(commands)

        # 发送命令
        self.process.stdin.write((cmd + self.command_terminator).encode())
        self.process.stdin.write(
            f'echo "{END_MARKER_VALUE}"{self.command_terminator}'.encode()
        )  # 发送结束标记
        await self.process.stdin.drain()
        if daemon:
            asyncio.create_task(self._read_and_process_output(cmd))
        else:
            output += await self._read_and_process_output(cmd)

        return output

    async def execute_in_conda_env(self, cmd: str, env, daemon=False) -> str:
        """
        在指定的 Conda 环境中执行命令，自动激活环境。

        参数：
            cmd (str): 要执行的命令。
            env (str): 需要激活的 Conda 环境名称。
            daemon (bool): 如果为 True，则在后台运行命令。

        返回：
            str: 命令的输出结果。如果 `daemon=True`，则返回空字符串，需调用 `get_stdout_output` 获取输出。
        """
        cmd = f"conda run -n {env} {cmd}"
        return await self.run_command(cmd, daemon=daemon)

    async def get_stdout_output(self) -> str:
        """
        获取后台执行的命令输出。

        返回：
            str: 终端输出的内容。
        """
        output_lines = []
        while not self.stdout_queue.empty():
            line = await self.stdout_queue.get()
            output_lines.append(line)
        return "\n".join(output_lines)

    async def _read_and_process_output(self, cmd, daemon=False) -> str:
        """
        读取并处理终端输出。

        参数：
            cmd (str): 需要执行的命令。
            daemon (bool): 是否在后台运行命令。

        返回：
            str: 命令的输出内容。
        """
        async with self.observer as observer:
            cmd_output = []
            await observer.async_report(cmd + self.command_terminator, "cmd")
            tmp = b""
            while True:
                output = tmp + await self.process.stdout.read(1)
                if not output:
                    continue
                *lines, tmp = output.splitlines(True)
                for line in lines:
                    line = line.decode()
                    ix = line.rfind(END_MARKER_VALUE)
                    if ix >= 0:
                        line = line[:ix]
                        if line:
                            await observer.async_report(line, "output")
                            cmd_output.append(line)
                        return "".join(cmd_output)
                    await observer.async_report(line, "output")
                    cmd_output.append(line)
                    if daemon:
                        await self.stdout_queue.put(line)

    async def close(self):
        """关闭持久化 shell 进程"""
        self.process.stdin.close()
        await self.process.wait()


@register_tool(include_functions=["run"])
class Bash(Terminal):
    """
    继承自 Terminal，用于执行 Bash 命令，并提供自定义 shell 函数。
    """

    def __init__(self):
        """初始化"""
        os.environ["SWE_CMD_WORK_DIR"] = str(Config.default().workspace.path)
        super().__init__()
        self.start_flag = False  # 标记是否已启动

    async def start(self):
        """启动 Bash 终端，并切换到工作目录"""
        await self.run_command(f"cd {Config.default().workspace.path}")
        await self.run_command(f"source {SWE_SETUP_PATH}")

    async def run(self, cmd) -> str:
        """
        执行 Bash 命令。

        参数：
            cmd (str): 要执行的命令。

        返回：
            str: 命令的执行结果。

        该方法允许执行标准 Bash 命令以及环境中的自定义函数，如：

        - `open <path> [<line_number>]`：打开文件，跳转到指定行。
        - `goto <line_number>`：移动窗口至指定行。
        - `scroll_down` / `scroll_up`：向下 / 向上滚动窗口。
        - `create <filename>`：创建并打开新文件。
        - `search_dir_and_preview <search_term> [<dir>]`：在目录中搜索关键词，并显示代码预览。
        - `search_file <search_term> [<file>]`：在文件中搜索关键词。
        - `find_file <file_name> [<dir>]`：在目录中查找指定文件。
        - `edit <start_line>:<end_line> <<EOF ... EOF`：编辑代码文件，符合 PEP8 规范。
        - `submit`：提交代码（只能执行一次）。
        """
        if not self.start_flag:
            await self.start()
            self.start_flag = True

        return await self.run_command(cmd)