"""
This file is borrowed from OpenDevin
You can find the original repository here:
https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/runtime/plugins/agent_skills/file_ops/file_ops.py
"""
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import tiktoken
from pydantic import BaseModel, ConfigDict

from metagpt.const import DEFAULT_MIN_TOKEN_COUNT, DEFAULT_WORKSPACE_ROOT
from metagpt.tools.libs.linter import Linter
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import awrite
from metagpt.utils.file import File
from metagpt.utils.report import EditorReporter

# 此常量用于单元测试！
LINTER_ERROR_MSG = "[您的编辑引入了新的语法错误。请理解错误并重试编辑命令。]\n"

# 错误信息，描述代码缩进问题
INDENTATION_INFO = """
上一行是：
"{pre_line}"
该行的缩进是 {pre_line_indent} 个空格。

错误行是：
"{insert_line}"
该行的缩进是 {insert_line_indent} 个空格。

请检查代码缩进，确保它不会导致任何错误。
尝试使用 {sub_4_space} 或 {add_4_space} 个空格进行缩进。
"""

# 错误指导信息，提供具体的错误信息和修改建议
ERROR_GUIDANCE = """
{linter_error_msg}

[如果您的编辑被应用，代码将如下所示]
-------------------------------------------------
{window_after_applied}
-------------------------------------------------

[您的编辑前的原始代码]
-------------------------------------------------
{window_before_applied}
-------------------------------------------------

您的更改未应用。请修正您的编辑命令并再次尝试。
{guidance_message}
"""

# 行号和内容不匹配的错误信息
LINE_NUMBER_AND_CONTENT_MISMATCH = """错误：`{position}_replaced_line_number` 与 `{position}_replaced_line_content` 不匹配。请修正参数。
`{position}_replaced_line_number` 是 {line_number}，对应的内容是 "{true_content}"。
但 `{position}_replaced_line_content` 是 "{fake_content}"。
指定行周围的内容是：
{context}
请注意新内容，确保它与新参数一致。
"""

# 成功编辑后的信息
SUCCESS_EDIT_INFO = """
[文件：{file_name}（编辑后共 {n_total_lines} 行）]
{window_after_applied}
[文件已更新（编辑行：{line_number}）]。
"""
# 请检查更改并确保它们是正确的（正确的缩进、没有重复行等）。如有必要，请再次编辑文件。

# 文件块类，表示文件中的一块内容
class FileBlock(BaseModel):
    """文件中的一块内容"""

    file_path: str  # 文件路径
    block_content: str  # 文件内容


# 行号错误异常类
class LineNumberError(Exception):
    pass


# 编辑器工具类，提供读取、理解、写入和编辑文件的功能
@register_tool(
    include_functions=[
        "write",
        "read",
        "open_file",
        "goto_line",
        "scroll_down",
        "scroll_up",
        "create_file",
        "edit_file_by_replace",
        "insert_content_at_line",
        "append_file",
        "search_dir",
        "search_file",
        "find_file",
        "similarity_search",
    ]
)
class Editor(BaseModel):
    """
    一个用于读取、理解、写入和编辑文件的工具类。
    支持本地文件，包括文本文件（txt、md、json、py、html、js、css等），pdf，docx，排除图片、csv、excel文件，或在线链接
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 配置字典
    resource: EditorReporter = EditorReporter()  # 编辑器报告工具
    current_file: Optional[Path] = None  # 当前打开的文件
    current_line: int = 1  # 当前行号
    window: int = 200  # 显示窗口大小
    enable_auto_lint: bool = False  # 是否启用自动 lint（代码检查）
    working_dir: Path = DEFAULT_WORKSPACE_ROOT  # 工作目录

    # 写入文件内容
    def write(self, path: str, content: str):
        """将完整内容写入文件。使用时，确保 `content` 参数包含文件的完整内容。"""

        path = self._try_fix_path(path)  # 修复路径

        if "\n" not in content and "\\n" in content:
            # 一个简单的规则来修正内容：如果 'content' 中没有实际的换行符（\n），但包含 '\\n'，则考虑
            # 将其替换为 '\n'，以修正误表示的换行符。
            content = content.replace("\\n", "\n")
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # 如果目录不存在，则创建目录
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)  # 写入文件内容
        return f"文件 {os.path.basename(path)} 的写入/编码已完成，文件 '{os.path.basename(path)}' 已成功创建。"

    # 异步读取文件内容
    async def read(self, path: str) -> FileBlock:
        """读取文件的完整内容。使用绝对路径作为参数指定文件位置。"""

        path = self._try_fix_path(path)  # 修复路径

        error = FileBlock(
            file_path=str(path),
            block_content="文件过大无法读取。请使用 `Editor.similarity_search` 来读取文件。",
        )
        path = Path(path)
        if path.stat().st_size > 5 * DEFAULT_MIN_TOKEN_COUNT:
            return error  # 文件过大时返回错误信息
        content = await File.read_text_file(path)
        if not content:
            return FileBlock(file_path=str(path), block_content="")  # 如果文件为空，返回空内容
        if self.is_large_file(content=content):
            return error  # 如果文件被认为是大文件，返回错误信息
        self.resource.report(str(path), "path")  # 记录报告

        lines = content.splitlines(keepends=True)  # 保留行尾字符
        lines_with_num = [f"{i + 1:03}|{line}" for i, line in enumerate(lines)]
        result = FileBlock(
            file_path=str(path),
            block_content="".join(lines_with_num),
        )
        return result

    # 校验文件名是否合法
    @staticmethod
    def _is_valid_filename(file_name: str) -> bool:
        if not file_name or not file_name.strip():
            return False
        invalid_chars = '<>:"/\\|?*'  # 无效字符
        if os.name == "nt":  # Windows 系统
            invalid_chars = '<>:"/\\|?*'
        elif os.name == "posix":  # Unix-like 系统
            invalid_chars = "\0"

        for char in invalid_chars:
            if char in file_name:
                return False
        return True

    # 校验路径是否合法
    @staticmethod
    def _is_valid_path(path: Path) -> bool:
        try:
            return path.exists()
        except PermissionError:
            return False

    # 创建文件路径
    @staticmethod
    def _create_paths(file_path: Path) -> bool:
        try:
            if file_path.parent:
                file_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
            return True
        except PermissionError:
            return False

    # 校验当前文件是否有效
    def _check_current_file(self, file_path: Optional[Path] = None) -> bool:
        if file_path is None:
            file_path = self.current_file
        if not file_path or not file_path.is_file():
            raise ValueError("没有文件被打开。请先使用 open_file 函数。")
        return True

    # 限制数值范围
    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))

    # 文件代码检查（lint）
    def _lint_file(self, file_path: Path) -> tuple[Optional[str], Optional[int]]:
        """检查文件的 lint，并返回一个元组，包含错误信息（如果有的话），以及第一个错误的行号（如果有的话）。"""

        linter = Linter(root=self.working_dir)
        lint_error = linter.lint(str(file_path))
        if not lint_error:
            # lint 成功，没有发现问题
            return None, None
        return "ERRORS:\n" + lint_error.text, lint_error.lines[0]

    # 打印文件窗口内容（针对特定行号及窗口大小）
    def _print_window(self, file_path: Path, targeted_line: int, window: int):
        self._check_current_file(file_path)  # 校验当前文件是否有效
        with file_path.open() as file:
            content = file.read()

            # 确保内容以换行符结尾
            if not content.endswith("\n"):
                content += "\n"

            lines = content.splitlines(True)  # 保留所有行结束符
            total_lines = len(lines)

            # 处理边界情况
            self.current_line = self._clamp(targeted_line, 1, total_lines)
            half_window = max(1, window // 2)

            # 确保目标行上方和下方至少各有一行
            start = max(1, self.current_line - half_window)
            end = min(total_lines, self.current_line + half_window)

            # 调整起始和结束行，以确保至少上下各有一行
            if start == 1:
                end = min(total_lines, start + window - 1)
            if end == total_lines:
                start = max(1, end - window + 1)

            output = ""

            # 如果上方有行，显示更多行数
            if start > 1:
                output += f"({start - 1} 行以上内容)\n"
            else:
                output += "(这是文件的开头)\n"
            for i in range(start, end + 1):
                _new_line = f"{i:03d}|{lines[i - 1]}"
                if not _new_line.endswith("\n"):
                    _new_line += "\n"
                output += _new_line
            if end < total_lines:
                output += f"({total_lines - end} 行以下内容)\n"
            else:
                output += "(这是文件的结尾)\n"
            output = output.rstrip()

            return output


    @staticmethod
    def _cur_file_header(current_file: Path, total_lines: int) -> str:
        """返回当前文件的头部信息，包括文件路径和总行数。"""
        if not current_file:
            return ""
        return f"[File: {current_file.resolve()} ({total_lines} lines total)]\n"

    def _set_workdir(self, path: str) -> None:
        """
        设置当前工作目录为给定路径，例如：仓库目录。
        必须在打开文件之前设置工作目录。

        参数:
            path: str: 要设置为工作目录的路径。
        """
        self.working_dir = Path(path)

    def open_file(
        self, path: Union[Path, str], line_number: Optional[int] = 1, context_lines: Optional[int] = None
    ) -> str:
        """打开指定路径的文件，如果提供了行号，则将窗口移动到该行。
        默认只显示前100行。最大支持的上下文行数是2000，如果想查看更多，可以使用滚动条。

        参数:
            path: str: 要打开的文件路径，建议使用绝对路径。
            line_number: int | None = 1: 要移动到的行号，默认是1。
            context_lines: int | None = 100: 显示的上下文行数（通常是从第1行开始），默认是100。
        """
        if context_lines is None:
            context_lines = self.window

        path = self._try_fix_path(path)

        if not path.is_file():
            raise FileNotFoundError(f"文件 {path} 未找到")

        self.current_file = path
        with path.open() as file:
            total_lines = max(1, sum(1 for _ in file))

        if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
            raise ValueError(f"行号必须在1和{total_lines}之间")
        self.current_line = line_number

        # 覆盖上下文行数
        if context_lines is None or context_lines < 1:
            context_lines = self.window

        output = self._cur_file_header(path, total_lines)
        output += self._print_window(path, self.current_line, self._clamp(context_lines, 1, 2000))
        self.resource.report(path, "path")
        return output

    def goto_line(self, line_number: int) -> str:
        """将窗口移动到指定的行号。

        参数:
            line_number: int: 要跳转的行号。
        """
        self._check_current_file()

        with self.current_file.open() as file:
            total_lines = max(1, sum(1 for _ in file))
        if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
            raise ValueError(f"行号必须在1和{total_lines}之间")

        self.current_line = self._clamp(line_number, 1, total_lines)

        output = self._cur_file_header(self.current_file, total_lines)
        output += self._print_window(self.current_file, self.current_line, self.window)
        return output

    def scroll_down(self) -> str:
        """将窗口向下滚动100行。"""
        self._check_current_file()

        with self.current_file.open() as file:
            total_lines = max(1, sum(1 for _ in file))
        self.current_line = self._clamp(self.current_line + self.window, 1, total_lines)
        output = self._cur_file_header(self.current_file, total_lines)
        output += self._print_window(self.current_file, self.current_line, self.window)
        return output

    def scroll_up(self) -> str:
        """将窗口向上滚动100行。"""
        self._check_current_file()

        with self.current_file.open() as file:
            total_lines = max(1, sum(1 for _ in file))
        self.current_line = self._clamp(self.current_line - self.window, 1, total_lines)
        output = self._cur_file_header(self.current_file, total_lines)
        output += self._print_window(self.current_file, self.current_line, self.window)
        return output

    async def create_file(self, filename: str) -> str:
        """创建并打开一个新文件。

        参数:
            filename: str: 要创建的文件名。如果父目录不存在，将会创建该目录。
        """
        filename = self._try_fix_path(filename)

        if filename.exists():
            raise FileExistsError(f"文件 '{filename}' 已存在。")
        await awrite(filename, "\n")

        self.open_file(filename)
        return f"[文件 {filename} 已创建。]"

    @staticmethod
    def _append_impl(lines, content):
        """内部方法，处理文件追加内容。

        参数:
            lines: list[str]: 原文件的行列表。
            content: str: 要追加的内容。

        返回:
            content: str: 文件的新内容。
            n_added_lines: int: 添加的行数。
        """
        content_lines = content.splitlines(keepends=True)
        n_added_lines = len(content_lines)
        if lines and not (len(lines) == 1 and lines[0].strip() == ""):
            # 文件不为空
            if not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            new_lines = lines + content_lines
            content = "".join(new_lines)
        else:
            # 文件为空
            content = "".join(content_lines)

        return content, n_added_lines

    @staticmethod
    def _insert_impl(lines, start, content):
        """内部方法，处理文件插入内容。

        参数:
            lines: list[str]: 原文件的行列表。
            start: int: 插入的起始行号。
            content: str: 要插入的内容。

        返回:
            content: str: 文件的新内容。
            n_added_lines: int: 添加的行数。

        异常:
            LineNumberError: 如果起始行号无效。
        """
        inserted_lines = [content + "\n" if not content.endswith("\n") else content]
        if len(lines) == 0:
            new_lines = inserted_lines
        elif start is not None:
            if len(lines) == 1 and lines[0].strip() == "":
                # 如果文件只有1行且该行为空
                lines = []

            if len(lines) == 0:
                new_lines = inserted_lines
            else:
                new_lines = lines[: start - 1] + inserted_lines + lines[start - 1 :]

        else:
            raise LineNumberError(
                f"无效的行号: {start}。行号必须在1和{len(lines)}之间。"
            )

        content = "".join(new_lines)
        n_added_lines = len(inserted_lines)
        return content, n_added_lines

    @staticmethod
    def _edit_impl(lines, start, end, content):
        """内部方法，处理编辑文件内容。

        参数:
            lines: list[str]: 原文件的行列表。
            start: int: 编辑的起始行号。
            end: int: 编辑的结束行号。
            content: str: 用于替换的内容。

        返回:
            content: str: 文件的新内容。
            n_added_lines: int: 添加的行数。
        """
        # 处理 start 或 end 为 None 的情况
        if start is None:
            start = 1  # 默认从第一行开始
        if end is None:
            end = len(lines)  # 默认到最后一行
        # 校验参数
        if not (1 <= start <= len(lines)):
            raise LineNumberError(
                f"无效的起始行号: {start}。行号必须在1和{len(lines)}之间。"
            )
        if not (1 <= end <= len(lines)):
            raise LineNumberError(
                f"无效的结束行号: {end}。行号必须在1和{len(lines)}之间。"
            )
        if start > end:
            raise LineNumberError(f"无效的行号范围: {start}-{end}。起始行号必须小于或等于结束行号。")

        # 确保内容以换行符结束
        if not content.endswith("\n"):
            content += "\n"
        content_lines = content.splitlines(True)

        # 计算添加的行数
        n_added_lines = len(content_lines)

        # 移除指定范围的行并插入新内容
        new_lines = lines[: start - 1] + content_lines + lines[end:]

        # 处理原始行为空的情况
        if len(lines) == 0:
            new_lines = content_lines

        # 合并行并生成新的内容
        content = "".join(new_lines)
        return content, n_added_lines

    @staticmethod
    def _insert_impl(lines, start, content):
        """处理文件插入操作的内部方法。

        参数:
            lines: list[str]: 原始文件中的所有行。
            start: int: 插入操作的起始行号。
            content: str: 要插入的内容。

        返回:
            content: str: 更新后的文件内容。
            n_added_lines: int: 插入的行数。

        异常:
            LineNumberError: 如果起始行号无效。
        """
        inserted_lines = [content + "\n" if not content.endswith("\n") else content]
        if len(lines) == 0:
            new_lines = inserted_lines
        elif start is not None:
            if len(lines) == 1 and lines[0].strip() == "":
                # 如果文件只有一行且该行为空
                lines = []

            if len(lines) == 0:
                new_lines = inserted_lines
            else:
                new_lines = lines[: start - 1] + inserted_lines + lines[start - 1 :]
        else:
            raise LineNumberError(
                f"无效的行号: {start}. 行号必须介于 1 和 {len(lines)} 之间（包含1和{len(lines)}）。"
            )

        content = "".join(new_lines)
        n_added_lines = len(inserted_lines)
        return content, n_added_lines

    @staticmethod
    def _edit_impl(lines, start, end, content):
        """处理文件编辑操作的内部方法。

        要求（由调用者检查）:
            start <= end
            start 和 end 必须介于 1 和 len(lines) 之间（包含1和len(lines)）。
            content 必须以换行符结尾。

        参数:
            lines: list[str]: 原始文件中的所有行。
            start: int: 编辑操作的起始行号。
            end: int: 编辑操作的结束行号。
            content: str: 用于替换的内容。

        返回:
            content: str: 更新后的文件内容。
            n_added_lines: int: 替换后新增的行数。
        """
        # 处理起始和结束行号为 None 的情况
        if start is None:
            start = 1  # 默认为文件开头
        if end is None:
            end = len(lines)  # 默认为文件末尾
        # 检查参数的合法性
        if not (1 <= start <= len(lines)):
            raise LineNumberError(
                f"无效的起始行号: {start}. 行号必须介于 1 和 {len(lines)} 之间（包含1和{len(lines)}）。"
            )
        if not (1 <= end <= len(lines)):
            raise LineNumberError(
                f"无效的结束行号: {end}. 行号必须介于 1 和 {len(lines)} 之间（包含1和{len(lines)}）。"
            )
        if start > end:
            raise LineNumberError(f"无效的行号范围: {start}-{end}. 起始行号必须小于或等于结束行号。")

        # 将内容按行分割，并确保内容以换行符结尾
        if not content.endswith("\n"):
            content += "\n"
        content_lines = content.splitlines(True)

        # 计算新增的行数
        n_added_lines = len(content_lines)

        # 删除指定范围的行，并插入新的内容
        new_lines = lines[: start - 1] + content_lines + lines[end :]

        # 处理原始行为空的情况
        if len(lines) == 0:
            new_lines = content_lines

        # 合并行并生成更新后的内容
        content = "".join(new_lines)
        return content, n_added_lines

    def _get_indentation_info(self, content, first_line):
        """
        获取插入行和前一行的缩进信息，并为下次尝试提供指导。
        """
        content_lines = content.split("\n")
        pre_line = content_lines[first_line - 2] if first_line - 2 >= 0 else ""
        pre_line_indent = len(pre_line) - len(pre_line.lstrip())
        insert_line = content_lines[first_line - 1]
        insert_line_indent = len(insert_line) - len(insert_line.lstrip())
        ret_str = INDENTATION_INFO.format(
            pre_line=pre_line,
            pre_line_indent=pre_line_indent,
            insert_line=insert_line,
            insert_line_indent=insert_line_indent,
            sub_4_space=max(insert_line_indent - 4, 0),
            add_4_space=insert_line_indent + 4,
        )
        return ret_str

    def _edit_file_impl(
            self,
            file_name: Path,
            start: Optional[int] = None,
            end: Optional[int] = None,
            content: str = "",
            is_insert: bool = False,
            is_append: bool = False,
    ) -> str:
        """处理编辑、插入和追加文件内容的通用逻辑。

        参数:
            file_name: Path: 要编辑或追加的文件名。
            start: int | None = None: 编辑的起始行号。如果是追加操作，则忽略此参数。
            end: int | None = None: 编辑的结束行号。如果是追加操作，则忽略此参数。
            content: str: 要替换的内容或要追加的内容。
            is_insert: bool = False: 是否在指定行号插入内容，而不是编辑。
            is_append: bool = False: 是否追加内容到文件末尾，而不是编辑。

        返回:
            str: 操作结果信息。
        """

        ERROR_MSG = f"[编辑文件 {file_name} 时出错，请确认文件是否正确。]"
        ERROR_MSG_SUFFIX = (
            "您的更改未被应用。请修复您的编辑命令并重新尝试。\n"
            "您需要 1) 打开正确的文件并重新尝试，或 2) 指定正确的行号参数。\n"
            "请勿重新运行相同的失败编辑命令。再次运行将导致相同的错误。"
        )

        if not self._is_valid_filename(file_name.name):
            raise FileNotFoundError("无效的文件名。")

        if not self._is_valid_path(file_name):
            raise FileNotFoundError("无效的路径或文件名。")

        if not self._create_paths(file_name):
            raise PermissionError("无法访问或创建目录。")

        if not file_name.is_file():
            raise FileNotFoundError(f"文件 {file_name} 未找到。")

        if is_insert and is_append:
            raise ValueError("不能同时进行插入和追加操作。")

        # 使用临时文件来写入更改
        content = str(content or "")
        temp_file_path = ""
        src_abs_path = file_name.resolve()
        first_error_line = None
        # 用于存储原始内容的备份文件，并将自动删除
        temp_backup_file = tempfile.NamedTemporaryFile("w", delete=True)

        try:
            # 如果启用了自动 lint，进行 lint 检查
            if self.enable_auto_lint:
                original_lint_error, _ = self._lint_file(file_name)

            # 创建临时文件
            with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
                temp_file_path = temp_file.name

                # 读取原文件，检查是否为空并处理末尾换行
                with file_name.open() as original_file:
                    lines = original_file.readlines()

                if is_append:
                    content, n_added_lines = self._append_impl(lines, content)
                elif is_insert:
                    try:
                        content, n_added_lines = self._insert_impl(lines, start, content)
                    except LineNumberError as e:
                        return (f"{ERROR_MSG}\n" f"{e}\n" f"{ERROR_MSG_SUFFIX}") + "\n"
                else:
                    try:
                        content, n_added_lines = self._edit_impl(lines, start, end, content)
                    except LineNumberError as e:
                        return (f"{ERROR_MSG}\n" f"{e}\n" f"{ERROR_MSG_SUFFIX}") + "\n"

                if not content.endswith("\n"):
                    content += "\n"

                # 将新内容写入临时文件
                temp_file.write(content)

            # 使用原子操作将临时文件替换为原文件
            shutil.move(temp_file_path, src_abs_path)

            # 进行 lint 检查
            if self.enable_auto_lint:
                # 备份原始文件
                temp_backup_file.writelines(lines)
                temp_backup_file.flush()
                lint_error, first_error_line = self._lint_file(file_name)

                # 提取修改导致的 lint 错误
                def extract_last_part(line):
                    parts = line.split(":")
                    if len(parts) > 1:
                        return parts[-1].strip()
                    return line.strip()

                def subtract_strings(str1, str2) -> str:
                    lines1 = str1.splitlines()
                    lines2 = str2.splitlines()

                    last_parts1 = [extract_last_part(line) for line in lines1]

                    remaining_lines = [line for line in lines2 if extract_last_part(line) not in last_parts1]

                    result = "\n".join(remaining_lines)
                    return result

                if original_lint_error and lint_error:
                    lint_error = subtract_strings(original_lint_error, lint_error)
                    if lint_error == "":
                        lint_error = None
                        first_error_line = None

                if lint_error is not None:
                    # 如果存在 lint 错误，提供修复建议
                    if is_append:
                        show_line = len(lines)
                    elif start is not None and end is not None:
                        show_line = int((start + end) / 2)
                    else:
                        raise ValueError("无效的状态。此情况不应发生。")

                    guidance_message = self._get_indentation_info(content, start or len(lines))
                    guidance_message += (
                        "您需要 1) 指定正确的起始/结束行参数，或 2) 修改您的编辑代码。\n"
                        "请勿重新运行相同的失败编辑命令。再次运行将导致相同的错误。"
                    )
                    lint_error_info = ERROR_GUIDANCE.format(
                        linter_error_msg=LINTER_ERROR_MSG + lint_error,
                        window_after_applied=self._print_window(file_name, show_line, n_added_lines + 20),
                        window_before_applied=self._print_window(
                            Path(temp_backup_file.name), show_line, n_added_lines + 20
                        ),
                        guidance_message=guidance_message,
                    ).strip()

                    # 恢复原始文件
                    shutil.move(temp_backup_file.name, src_abs_path)
                    return lint_error_info

        except FileNotFoundError as e:
            return f"文件未找到: {e}\n"
        except IOError as e:
            return f"处理文件时发生错误: {e}\n"
        except ValueError as e:
            return f"无效输入: {e}\n"
        except Exception as e:
            guidance_message = self._get_indentation_info(content, start or len(lines))
            guidance_message += (
                "您需要 1) 指定正确的起始/结束行参数，或 2) 增加原始代码范围。\n"
                "请勿重新运行相同的失败编辑命令。再次运行将导致相同的错误。"
            )
            error_info = ERROR_GUIDANCE.format(
                linter_error_msg=LINTER_ERROR_MSG + str(e),
                window_after_applied=self._print_window(file_name, start or len(lines), 100),
                window_before_applied=self._print_window(Path(temp_backup_file.name), start or len(lines), 100),
                guidance_message=guidance_message,
            ).strip()
            # 出现错误时清理临时文件
            shutil.move(temp_backup_file.name, src_abs_path)
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink()

            # 记录日志并抛出异常
            raise Exception(f"{error_info}") from e

        # 更新文件信息并打印更新后的内容
        with file_name.open("r", encoding="utf-8") as file:
            n_total_lines = max(1, len(file.readlines()))
        if first_error_line is not None and int(first_error_line) > 0:
            self.current_line = first_error_line
        else:
            if is_append:
                self.current_line = max(1, len(lines))  # 原文件的末尾
            else:
                self.current_line = start or n_total_lines or 1
        success_edit_info = SUCCESS_EDIT_INFO.format(
            file_name=file_name.resolve(),
            n_total_lines=n_total_lines,
            window_after_applied=self._print_window(file_name, self.current_line, self.window),
            line_number=self.current_line,
        ).strip()
        return success_edit_info

    def edit_file_by_replace(
            self,
            file_name: str,
            first_replaced_line_number: int,
            first_replaced_line_content: str,
            last_replaced_line_number: int,
            last_replaced_line_content: str,
            new_content: str,
    ) -> str:
        """
        行号从 1 开始。将文件中从 start_line 到 end_line（包含）之间的行替换为 new_content。
        所有的新内容都会被输入，因此请确保你的缩进格式正确。
        new_content 必须是完整的代码块。

        示例 1:
        给定一个文件 "/workspace/example.txt"，其内容如下：
        ```
        001|contain f
        002|contain g
        003|contain h
        004|contain i
        ```

        编辑：如果你想替换第 2 行和第 3 行

        edit_file_by_replace(
            "/workspace/example.txt",
            first_replaced_line_number=2,
            first_replaced_line_content="contain g",
            last_replaced_line_number=3,
            last_replaced_line_content="contain h",
            new_content="new content",
        )
        这将把第 2 行和第 3 行替换为 "new content"。

        结果文件将是：
        ```
        001|contain f
        002|new content
        003|contain i
        ```

        示例 2:
        给定一个文件 "/workspace/example.txt"，其内容如下：
        ```
        001|contain f
        002|contain g
        003|contain h
        004|contain i
        ```

        编辑：如果你想删除第 2 行和第 3 行。

        edit_file_by_replace(
            "/workspace/example.txt",
            first_replaced_line_number=2,
            first_replaced_line_content="contain g",
            last_replaced_line_number=3,
            last_replaced_line_content="contain h",
            new_content="",
        )
        这将删除第 2 行和第 3 行。

        结果文件将是：
        ```
        001|contain f
        002|
        003|contain i
        ```

        参数:
            file_name (str): 要编辑的文件名。
            first_replaced_line_number (int): 开始编辑的行号，从 1 开始。
            first_replaced_line_content (str): 开始替换的行的内容，依据 first_replaced_line_number。
            last_replaced_line_number (int): 结束编辑的行号（包含），从 1 开始。
            last_replaced_line_content (str): 结束替换的行的内容，依据 last_replaced_line_number。
            new_content (str): 用来替换当前选择的文本，必须符合 PEP8 标准。开始行和结束行的内容也会被替换。

        """

        # 尝试修复文件路径
        file_name = self._try_fix_path(file_name)

        # 检查 first_replaced_line_number 和 last_replaced_line_number 是否对应正确的内容
        mismatch_error = ""
        with file_name.open() as file:
            content = file.read()
            # 确保内容以换行符结尾
            if not content.endswith("\n"):
                content += "\n"
            lines = content.splitlines(True)
            total_lines = len(lines)
            check_list = [
                ("first", first_replaced_line_number, first_replaced_line_content),
                ("last", last_replaced_line_number, last_replaced_line_content),
            ]
            # 检查传入的行号和内容是否匹配
            for position, line_number, line_content in check_list:
                if line_number > len(lines) or lines[line_number - 1].rstrip() != line_content:
                    start = max(1, line_number - 3)
                    end = min(total_lines, line_number + 3)
                    context = "\n".join(
                        [
                            f'The {cur_line_number:03d} line is "{lines[cur_line_number - 1].rstrip()}"'
                            for cur_line_number in range(start, end + 1)
                        ]
                    )
                    mismatch_error += LINE_NUMBER_AND_CONTENT_MISMATCH.format(
                        position=position,
                        line_number=line_number,
                        true_content=lines[line_number - 1].rstrip()
                        if line_number - 1 < len(lines)
                        else "OUT OF FILE RANGE!",
                        fake_content=line_content.replace("\n", "\\n"),
                        context=context.strip(),
                    )
        if mismatch_error:
            raise ValueError(mismatch_error)

        # 调用内部方法进行文件编辑
        ret_str = self._edit_file_impl(
            file_name,
            start=first_replaced_line_number,
            end=last_replaced_line_number,
            content=new_content,
        )
        # TODO: 自动尝试修复 linter 错误（可能需要使用一些静态分析工具来处理编辑位置附近的缩进）
        self.resource.report(file_name, "path")
        return ret_str

    def _edit_file_by_replace(self, file_name: str, to_replace: str, new_content: str) -> str:
        """编辑文件。此函数将在给定文件中查找`to_replace`并将其替换为`new_content`。

        每个 *to_replace* 必须 *完全匹配* 现有源代码，逐字符匹配，包括所有注释、文档字符串等。

        请确保 `to_replace` 足够唯一，可以准确定位要替换的部分。`to_replace` 不得为空。

        示例：
        假设文件"/workspace/example.txt"的内容如下：
        ```
        line 1
        line 2
        line 2
        line 3
        ```

        编辑：如果你想替换第二次出现的 "line 2"，你可以使 `to_replace` 唯一：

        edit_file_by_replace(
            '/workspace/example.txt',
            to_replace='line 2\nline 3',
            new_content='new line\nline 3',
        )

        这将只替换第二次出现的 "line 2" 为 "new line"。第一次出现的 "line 2" 将保持不变。

        更新后的文件内容为：
        ```
        line 1
        line 2
        new line
        line 3
        ```

        删除：如果你想删除 "line 2" 和 "line 3"，可以将 `new_content` 设置为空字符串：

        edit_file_by_replace(
            '/workspace/example.txt',
            to_replace='line 2\nline 3',
            new_content='',
        )

        参数：
            file_name: (str): 要编辑的文件名。
            to_replace: (str): 要查找并替换的内容。
            new_content: (str): 用于替换旧内容的新内容。

        注意：
            此工具是独占的。如果使用此工具，则不能在当前响应中使用其他命令。
            如果需要多次使用它，请等待下一个回合。
        """
        # FIXME: 支持替换所有出现的内容

        if to_replace == new_content:
            raise ValueError("`to_replace` 和 `new_content` 必须不同。")

        # 在文件中查找 `to_replace`
        # 如果找到，将其替换为 `new_content`
        # 如果未找到，则执行模糊搜索以查找最接近的匹配并替换为 `new_content`
        file_name = self._try_fix_path(file_name)
        with file_name.open("r") as file:
            file_content = file.read()

        if to_replace.strip() == "":
            if file_content.strip() == "":
                raise ValueError(f"文件 '{file_name}' 为空。请使用追加方法添加内容。")
            raise ValueError("`to_replace` 不能为空。")

        if file_content.count(to_replace) > 1:
            raise ValueError(
                "`to_replace` 出现了多次，请包含足够的行以使 `to_replace` 唯一。"
            )
        start = file_content.find(to_replace)
        if start != -1:
            # 将起始位置从索引转换为行号
            start_line_number = file_content[:start].count("\n") + 1
            end_line_number = start_line_number + len(to_replace.splitlines()) - 1
        else:
            def _fuzzy_transform(s: str) -> str:
                # 移除所有空格，保留换行符
                return re.sub(r"[^\S\n]+", "", s)

            # 执行模糊搜索（移除所有空格，保留换行符）
            to_replace_fuzzy = _fuzzy_transform(to_replace)
            file_content_fuzzy = _fuzzy_transform(file_content)
            # 查找最接近的匹配
            start = file_content_fuzzy.find(to_replace_fuzzy)
            if start == -1:
                return f"[在 {file_name} 中未找到完全匹配的内容\n```\n{to_replace}\n```\n]"
            # 将起始位置从索引转换为行号（模糊匹配）
            start_line_number = file_content_fuzzy[:start].count("\n") + 1
            end_line_number = start_line_number + len(to_replace.splitlines()) - 1

        ret_str = self._edit_file_impl(
            file_name,
            start=start_line_number,
            end=end_line_number,
            content=new_content,
            is_insert=False,
        )
        # lint_error = bool(LINTER_ERROR_MSG in ret_str)
        # TODO: 自动尝试修复linter错误（可能需要使用静态分析工具来检查编辑附近的代码以确定缩进）
        self.resource.report(file_name, "path")
        return ret_str

    def insert_content_at_line(self, file_name: str, line_number: int, insert_content: str) -> str:
        """在文件中的指定行号之前插入一块完整的代码块。即，新内容将从指定行的开头开始，现有的该行内容将被向下移动。
        此操作不会修改指定行号之前或之后的行内容。
        该函数不能在文件末尾插入内容，请使用 `append_file` 方法。

        示例：如果文件具有以下内容：
        ```
        001|contain g
        002|contain h
        003|contain i
        004|contain j
        ```
        并且你调用：
        insert_content_at_line(
            file_name='file.txt',
            line_number=2,
            insert_content='new line'
        )
        文件将更新为：
        ```
        001|contain g
        002|new line
        003|contain h
        004|contain i
        005|contain j
        ```

        参数：
            file_name: (str): 要编辑的文件名。
            line_number (int): 要插入内容的行号（从1开始）。插入内容将在 `line_number-1` 和 `line_number` 之间添加。
            insert_content (str): 要插入的内容，必须是完整的代码块。

        注意：
            此工具是独占的。如果使用此工具，则不能在当前响应中使用其他命令。
            如果需要多次使用它，请等待下一个回合。
        """
        file_name = self._try_fix_path(file_name)
        ret_str = self._edit_file_impl(
            file_name,
            start=line_number,
            end=line_number,
            content=insert_content,
            is_insert=True,
            is_append=False,
        )
        self.resource.report(file_name, "path")
        return ret_str

    def append_file(self, file_name: str, content: str) -> str:
        """将内容追加到指定文件的末尾。
        将文本`content`追加到指定文件的末尾。

        参数:
            file_name: str: 要编辑的文件名。
            content: str: 要插入的内容。
        注意:
            该工具是专用的。如果您使用此工具，当前响应中不能使用其他命令。
            如果需要多次使用它，请等待下一轮。
        """
        file_name = self._try_fix_path(file_name)
        ret_str = self._edit_file_impl(
            file_name,
            start=None,
            end=None,
            content=content,
            is_insert=False,
            is_append=True,
        )
        self.resource.report(file_name, "path")
        return ret_str

    def search_dir(self, search_term: str, dir_path: str = "./") -> str:
        """在目录中的所有文件中搜索搜索词。如果没有提供目录，默认在当前目录中搜索。

        参数:
            search_term: str: 要搜索的词语。
            dir_path: str: 要搜索的目录路径。
        """
        dir_path = self._try_fix_path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"未找到目录 {dir_path}")
        matches = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.startswith("."):
                    continue
                file_path = Path(root) / file
                with file_path.open("r", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if search_term in line:
                            matches.append((file_path, line_num, line.strip()))

        if not matches:
            return f'在目录 {dir_path} 中未找到与 "{search_term}" 匹配的内容'

        num_matches = len(matches)
        num_files = len(set(match[0] for match in matches))

        if num_files > 100:
            return f'在目录 {dir_path} 中有超过 {num_files} 个文件匹配 "{search_term}"，请缩小搜索范围。'

        res_list = [f'[在 {dir_path} 中找到 {num_matches} 个匹配 "{search_term}"]']
        for file_path, line_num, line in matches:
            res_list.append(f"{file_path} (第 {line_num} 行): {line}")
        res_list.append(f'[结束 "{search_term}" 在 {dir_path} 中的匹配]')
        return "\n".join(res_list)

    def search_file(self, search_term: str, file_path: Optional[str] = None) -> str:
        """在文件中搜索搜索词。如果没有提供文件，则在当前打开的文件中搜索。

        参数:
            search_term: str: 要搜索的词语。
            file_path: str | None: 要搜索的文件路径。
        """
        if file_path is None:
            file_path = self.current_file
        else:
            file_path = self._try_fix_path(file_path)
        if file_path is None:
            raise FileNotFoundError("没有指定文件或打开文件。请先使用 open_file 函数。")
        if not file_path.is_file():
            raise FileNotFoundError(f"未找到文件 {file_path}")

        matches = []
        with file_path.open() as file:
            for i, line in enumerate(file, 1):
                if search_term in line:
                    matches.append((i, line.strip()))
        res_list = []
        if matches:
            res_list.append(f'[在 {file_path} 中找到 {len(matches)} 个匹配 "{search_term}"]')
            for match in matches:
                res_list.append(f"第 {match[0]} 行: {match[1]}")
            res_list.append(f'[结束 "{search_term}" 在 {file_path} 中的匹配]')
        else:
            res_list.append(f'[未在 {file_path} 中找到 "{search_term}" 的匹配]')

        extra = {"type": "search", "symbol": search_term, "lines": [i[0] - 1 for i in matches]} if matches else None
        self.resource.report(file_path, "path", extra=extra)
        return "\n".join(res_list)

    def find_file(self, file_name: str, dir_path: str = "./") -> str:
        """在指定目录中查找给定名称的文件。

        参数:
            file_name: str: 要查找的文件名。
            dir_path: str: 要搜索的目录路径。
        """
        file_name = self._try_fix_path(file_name)
        dir_path = self._try_fix_path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"未找到目录 {dir_path}")

        matches = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if str(file_name) in file:
                    matches.append(Path(root) / file)

        res_list = []
        if matches:
            res_list.append(f'[在 {dir_path} 中找到 {len(matches)} 个匹配 "{file_name}"]')
            for match in matches:
                res_list.append(f"{match}")
            res_list.append(f'[结束 "{file_name}" 在 {dir_path} 中的匹配]')
        else:
            res_list.append(f'[在 {dir_path} 中未找到 "{file_name}"]')
        return "\n".join(res_list)

    def _try_fix_path(self, path: Union[Path, str]) -> Path:
        """尝试修正路径，如果路径不是绝对路径。

        参数:
            path: Union[Path, str]: 要修正的路径。

        返回:
            Path: 修正后的路径。
        """
        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_absolute():
            path = self.working_dir / path
        return path

    @staticmethod
    async def similarity_search(query: str, path: Union[str, Path]) -> List[str]:
        """在指定文件或路径中进行相似度搜索。

        该方法会在索引库中对给定的查询进行搜索，并根据文件或路径对其进行分类。
        它在每个文件的聚类中执行搜索，并单独处理未索引的文件，合并来自结构化索引的结果以及直接来自非索引文件的结果。
        此功能调用不依赖其他函数。

        参数:
            query (str): 要搜索的查询字符串。
            path (Union[str, Path]): 要搜索的文件或路径。

        返回:
            List[str]: 结果列表，包含来自合并结果和任何非索引文件的直接结果的文本。

        示例:
            >>> query = "文档中需要分析的问题"
            >>> file_or_path = "要搜索的文件或路径"
            >>> texts: List[str] = await Editor.similarity_search(query=query, path=file_or_path)
            >>> print(texts)
        """
        try:
            from metagpt.tools.libs.index_repo import IndexRepo

            return await IndexRepo.cross_repo_search(query=query, file_or_path=path)
        except ImportError:
            raise ImportError("要使用相似度搜索，您需要安装 RAG 模块。")

    @staticmethod
    def is_large_file(content: str, mix_token_count: int = 0) -> bool:
        """判断内容是否为大文件。

        参数:
            content: str: 要检查的内容。
            mix_token_count: int: 要使用的最小标记数（可选，默认为 0）。

        返回:
            bool: 如果文件超过最小标记数，则返回 True，否则返回 False。
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(content))
        mix_token_count = mix_token_count or DEFAULT_MIN_TOKEN_COUNT
        return token_count >= mix_token_count
