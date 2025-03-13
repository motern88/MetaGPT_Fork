# -*- encoding: utf-8 -*-
"""
@Date    :   2023/11/17 14:22:15
@Author  :   orange-crow
@File    :   execute_nb_code.py
"""
from __future__ import annotations

import asyncio
import base64
import re
from typing import Literal, Tuple

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionComplete, CellTimeoutError, DeadKernelError
from nbclient.util import ensure_async
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output, output_from_msg
from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.utils.report import NotebookReporter

INSTALL_KEEPLEN = 500
INI_CODE = """import warnings
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
warnings.filterwarnings('ignore')"""


class RealtimeOutputNotebookClient(NotebookClient):
    """实现实时输出Notebook执行的功能。"""

    def __init__(self, *args, notebook_reporter=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.notebook_reporter = notebook_reporter or NotebookReporter()

    async def _async_poll_output_msg(self, parent_msg_id: str, cell: NotebookNode, cell_index: int) -> None:
        """实现轮询消息并发送输出的功能。"""
        assert self.kc is not None
        while True:
            msg = await ensure_async(self.kc.iopub_channel.get_msg(timeout=None))
            await self._send_msg(msg)

            if msg["parent_header"].get("msg_id") == parent_msg_id:
                try:
                    # 执行完成时会抛出 CellExecutionComplete 异常
                    self.process_message(msg, cell, cell_index)
                except CellExecutionComplete:
                    return

    async def _send_msg(self, msg: dict):
        """发送消息，根据消息类型进行处理。"""
        msg_type = msg.get("header", {}).get("msg_type")
        if msg_type not in ["stream", "error", "execute_result"]:
            return

        await self.notebook_reporter.async_report(output_from_msg(msg), "content")


class ExecuteNbCode(Action):
    """执行Notebook代码块，返回结果到LLM并显示结果。"""

    nb: NotebookNode
    nb_client: RealtimeOutputNotebookClient = None
    console: Console
    interaction: str
    timeout: int = 600

    def __init__(self, nb=nbformat.v4.new_notebook(), timeout=600):
        super().__init__(
            nb=nb,
            timeout=timeout,
            console=Console(),
            interaction=("ipython" if self.is_ipython() else "terminal"),
        )
        self.reporter = NotebookReporter()
        self.set_nb_client()
        self.init_called = False

    async def init_code(self):
        """初始化代码，只会调用一次。"""
        if not self.init_called:
            await self.run(INI_CODE)
            self.init_called = True

    def set_nb_client(self):
        """设置Notebook客户端，用于执行Notebook代码。"""
        self.nb_client = RealtimeOutputNotebookClient(
            self.nb,
            timeout=self.timeout,
            resources={"metadata": {"path": self.config.workspace.path}},
            notebook_reporter=self.reporter,
            coalesce_streams=True,
        )

    async def build(self):
        """构建并启动Notebook客户端。"""
        if self.nb_client.kc is None or not await self.nb_client.kc.is_alive():
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()

    async def terminate(self):
        """终止Notebook客户端，清理资源。"""
        if self.nb_client.km is not None and await self.nb_client.km.is_alive():
            await self.nb_client.km.shutdown_kernel(now=True)
            await self.nb_client.km.cleanup_resources()

            channels = [
                self.nb_client.kc.stdin_channel,  # 处理标准输入的通道
                self.nb_client.kc.hb_channel,  # 内核心跳通道
                self.nb_client.kc.control_channel,  # 控制内核的通道
            ]

            # 停止所有运行中的通道
            for channel in channels:
                if channel.is_alive():
                    channel.stop()

            self.nb_client.kc = None
            self.nb_client.km = None

    async def reset(self):
        """重置Notebook客户端，清理并重新构建。"""
        await self.terminate()

        # 等待1秒，确保内核完全清理
        await asyncio.sleep(1)
        await self.build()
        self.set_nb_client()

    def add_code_cell(self, code: str):
        """向Notebook中添加代码单元。"""
        self.nb.cells.append(new_code_cell(source=code))

    def add_markdown_cell(self, markdown: str):
        """向Notebook中添加Markdown单元。"""
        self.nb.cells.append(new_markdown_cell(source=markdown))

    def _display(self, code: str, language: Literal["python", "markdown"] = "python"):
        """在控制台中显示代码或Markdown内容。"""
        if language == "python":
            code = Syntax(code, "python", theme="paraiso-dark", line_numbers=True)
            self.console.print(code)
        elif language == "markdown":
            display_markdown(code)
        else:
            raise ValueError(f"只支持 python 或 markdown，但接收到 {language}")

    def add_output_to_cell(self, cell: NotebookNode, output: str):
        """将代码执行的输出添加到Notebook单元。"""
        if "outputs" not in cell:
            cell["outputs"] = []
        else:
            cell["outputs"].append(new_output(output_type="stream", name="stdout", text=str(output)))

    def parse_outputs(self, outputs: list[str], keep_len: int = 5000) -> Tuple[bool, str]:
        """解析Notebook执行返回的输出。"""
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        for i, output in enumerate(outputs):
            output_text = ""
            if output["output_type"] == "stream" and not any(
                tag in output["text"]
                for tag in ["| INFO     | metagpt", "| ERROR    | metagpt", "| WARNING  | metagpt", "DEBUG"]
            ):
                output_text = output["text"]
            elif output["output_type"] == "display_data":
                if "image/png" in output["data"]:
                    self.show_bytes_figure(output["data"]["image/png"], self.interaction)
                else:
                    logger.info(
                        f"{i}th output['data'] from nbclient outputs dont have image/png, continue next output ..."
                    )
            elif output["output_type"] == "execute_result":
                output_text = output["data"]["text/plain"]
            elif output["output_type"] == "error":
                output_text, is_success = "\n".join(output["traceback"]), False

            # 处理未异步执行的协程
            if output_text.strip().startswith("<coroutine object"):
                output_text = "执行代码失败，您需要使用关键字 'await' 来运行异步代码。"
                is_success = False

            output_text = remove_escape_and_color_codes(output_text)
            if is_success:
                output_text = remove_log_and_warning_lines(output_text)
            # 异常信息通常出现在输出的末尾，正常输出信息通常出现在开头。
            if "<!DOCTYPE html>" not in output_text:
                output_text = output_text[:keep_len] if is_success else output_text[-keep_len:]

            parsed_output.append(output_text)
        return is_success, ",".join(parsed_output)

    def show_bytes_figure(self, image_base64: str, interaction_type: Literal["ipython", None]):
        """显示图像（用于IPython交互式环境或本地环境）。"""
        image_bytes = base64.b64decode(image_base64)
        if interaction_type == "ipython":
            from IPython.display import Image, display

            display(Image(data=image_bytes))
        else:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            image.show()

    def is_ipython(self) -> bool:
        """判断是否在IPython环境中运行（如Jupyter Notebook）。"""
        try:
            # 在Jupyter Notebook中，__file__ 变量是不存在的
            from IPython import get_ipython

            if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
                return True
            else:
                return False
        except NameError:
            return False

    async def run_cell(self, cell: NotebookNode, cell_index: int) -> Tuple[bool, str]:
        """运行单元格并设置执行超时，返回执行结果及成功标识。"""
        await self.reporter.async_report(cell, "content")

        try:
            await self.nb_client.async_execute_cell(cell, cell_index)
            return self.parse_outputs(self.nb.cells[-1].outputs)
        except CellTimeoutError:
            assert self.nb_client.km is not None
            await self.nb_client.km.interrupt_kernel()
            await asyncio.sleep(1)
            error_msg = "单元格执行超时：执行超过了时间限制，已被停止；请考虑优化代码以提高性能。"
            return False, error_msg
        except DeadKernelError:
            await self.reset()
            return False, "内核崩溃"
        except Exception:
            return self.parse_outputs(self.nb.cells[-1].outputs)

    async def run(self, code: str, language: Literal["python", "markdown"] = "python") -> Tuple[str, bool]:
        """
        执行代码并返回执行结果以及成功标识。
        """
        self._display(code, language)

        async with self.reporter:
            if language == "python":
                # 向Notebook中添加代码
                self.add_code_cell(code=code)

                # 构建代码执行器
                await self.build()

                # 执行代码
                cell_index = len(self.nb.cells) - 1
                success, outputs = await self.run_cell(self.nb.cells[-1], cell_index)

                if "!pip" in code:
                    success = False
                    outputs = outputs[-INSTALL_KEEPLEN:]
                elif "git clone" in code:
                    outputs = outputs[:INSTALL_KEEPLEN] + "..." + outputs[-INSTALL_KEEPLEN:]

            elif language == "markdown":
                # 向Notebook中添加Markdown内容
                self.add_markdown_cell(code)
                # Markdown单元没有执行失败，因此直接返回True
                outputs, success = code, True
            else:
                raise ValueError(f"只支持语言类型：python, markdown，但接收到 {language}。")

            file_path = self.config.workspace.path / "code.ipynb"
            nbformat.write(self.nb, file_path)
            await self.reporter.async_report(file_path, "path")

            return outputs, success


def remove_log_and_warning_lines(input_str: str) -> str:
    # 定义要删除的行类型，如警告信息、日志信息等
    delete_lines = ["[warning]", "warning:", "[cv]", "[info]"]

    # 分割输入字符串为行，去掉包含指定关键字的行
    result = "\n".join(
        [line for line in input_str.split("\n") if not any(dl in line.lower() for dl in delete_lines)]
    ).strip()
    return result


def remove_escape_and_color_codes(input_str: str):
    # 使用正则表达式去除Jupyter Notebook输出中的转义字符和颜色代码
    pattern = re.compile(r"\x1b\[[0-9;]*[mK]")  # 匹配ANSI转义字符
    result = pattern.sub("", input_str)  # 替换掉匹配到的字符
    return result


def display_markdown(content: str):
    # 使用正则表达式匹配代码块，一次匹配一个
    matches = re.finditer(r"```(.+?)```", content, re.DOTALL)

    start_index = 0  # 初始化起始索引
    content_panels = []  # 存储Markdown和代码的内容面板

    # 设置文本背景颜色和字体颜色
    style = "black on white"

    # 遍历匹配到的代码块
    for match in matches:
        # 获取代码块前的文本内容
        text_content = content[start_index: match.start()].strip()

        # 获取代码块中的内容并去除多余的反引号
        code_content = match.group(0).strip()[3:-3]  # 去掉三重反引号

        # 如果文本内容非空，添加为一个Panel
        if text_content:
            content_panels.append(Panel(Markdown(text_content), style=style, box=MINIMAL))

        # 如果代码内容非空，添加为一个Panel
        if code_content:
            content_panels.append(Panel(Markdown(f"```{code_content}"), style=style, box=MINIMAL))

        # 更新起始索引
        start_index = match.end()

    # 打印剩余的文本（如果有的话）
    remaining_text = content[start_index:].strip()
    if remaining_text:
        content_panels.append(Panel(Markdown(remaining_text), style=style, box=MINIMAL))

    # 使用Live模式显示所有面板
    with Live(auto_refresh=False, console=Console(), vertical_overflow="visible") as live:
        live.update(Group(*content_panels))  # 更新面板
        live.refresh()  # 刷新显示内容
