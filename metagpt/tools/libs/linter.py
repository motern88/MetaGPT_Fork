"""
This file is borrowed from OpenDevin
You can find the original repository here:
https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/runtime/plugins/agent_skills/utils/aider/linter.py
"""
import os
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from grep_ast import TreeContext, filename_to_lang
from tree_sitter_languages import get_parser  # noqa: E402

# 忽略 FutureWarning 警告
warnings.simplefilter("ignore", category=FutureWarning)

@dataclass
class LintResult:
    """
    存储 Lint 结果的类
    text: Lint 结果文本
    lines: 出错的行号列表
    """
    text: str
    lines: list

class Linter:
    """
    Linter 类用于执行代码检查
    """
    def __init__(self, encoding="utf-8", root=None):
        self.encoding = encoding  # 设定文件编码
        self.root = root  # 项目根目录

        # 定义不同语言的 Lint 处理方式
        self.languages = dict(
            python=self.py_lint,  # Python 使用 py_lint 处理
            sql=self.fake_lint,   # SQL 语法不完全支持，使用 fake_lint 忽略检查
            css=self.fake_lint,   # CSS 语法不完全支持，使用 fake_lint 忽略检查
            js=self.fake_lint,    # JavaScript 语法不完全支持，使用 fake_lint 忽略检查
            javascript=self.fake_lint,
        )
        self.all_lint_cmd = None  # 通用的 lint 命令

    def set_linter(self, lang, cmd):
        """
        设置特定语言的 Linter 处理方式
        """
        if lang:
            self.languages[lang] = cmd
            return
        self.all_lint_cmd = cmd

    def get_rel_fname(self, fname):
        """
        获取文件的相对路径
        """
        if self.root:
            return os.path.relpath(fname, self.root)
        else:
            return fname

    def run_cmd(self, cmd, rel_fname, code):
        """
        运行 lint 命令
        """
        cmd += " " + rel_fname
        cmd = cmd.split()
        process = subprocess.Popen(cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate()
        errors = stdout.decode().strip()
        self.returncode = process.returncode
        if self.returncode == 0:
            return  # 无错误

        res = errors
        line_num = extract_error_line_from(res)
        return LintResult(text=res, lines=[line_num])

    def get_abs_fname(self, fname):
        """
        获取文件的绝对路径
        """
        if os.path.isabs(fname):
            return fname
        elif os.path.isfile(fname):
            rel_fname = self.get_rel_fname(fname)
            return os.path.abspath(rel_fname)
        else:
            return self.get_rel_fname(fname)

    def lint(self, fname, cmd=None) -> Optional[LintResult]:
        """
        进行 lint 代码检查
        """
        code = Path(fname).read_text(self.encoding)
        absolute_fname = self.get_abs_fname(fname)
        if cmd:
            cmd = cmd.strip()
        if not cmd:
            lang = filename_to_lang(fname)
            if not lang:
                return None
            if self.all_lint_cmd:
                cmd = self.all_lint_cmd
            else:
                cmd = self.languages.get(lang)
        if callable(cmd):
            linkres = cmd(fname, absolute_fname, code)
        elif cmd:
            linkres = self.run_cmd(cmd, absolute_fname, code)
        else:
            linkres = basic_lint(absolute_fname, code)
        return linkres

    def flake_lint(self, rel_fname, code):
        """
        使用 flake8 进行 Python 代码检查
        """
        fatal = "F821,F822,F831,E112,E113,E999,E902"
        flake8 = f"flake8 --select={fatal} --isolated"
        try:
            flake_res = self.run_cmd(flake8, rel_fname, code)
        except FileNotFoundError:
            flake_res = None
        return flake_res

    def py_lint(self, fname, rel_fname, code):
        """
        执行 Python 代码的 Lint 过程
        """
        error = self.flake_lint(rel_fname, code)
        if not error:
            error = lint_python_compile(fname, code)
        if not error:
            error = basic_lint(rel_fname, code)
        return error

    def fake_lint(self, fname, rel_fname, code):
        """
        用于不支持 lint 的语言，直接返回 None
        """
        return None


def lint_python_compile(fname, code):
    """
    使用 Python 内置编译器检查语法错误
    """
    try:
        compile(code, fname, "exec")
        return
    except IndentationError as err:
        end_lineno = getattr(err, "end_lineno", err.lineno)
        line_numbers = [end_lineno - 1] if isinstance(end_lineno, int) else []
        tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
    res = "".join(tb_lines)
    return LintResult(text=res, lines=line_numbers)


def basic_lint(fname, code):
    """
    使用 tree-sitter 解析代码并检查错误
    """
    lang = filename_to_lang(fname)
    if not lang:
        return
    parser = get_parser(lang)
    tree = parser.parse(bytes(code, "utf-8"))
    errors = traverse_tree(tree.root_node)
    if not errors:
        return
    return LintResult(text=f"{fname}:{errors[0]}", lines=errors)


def extract_error_line_from(lint_error):
    """
    从 lint 结果中提取错误行号
    """
    for line in lint_error.splitlines():
        parts = line.split(":")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                continue


def traverse_tree(node):
    """
    遍历 tree-sitter 语法树查找错误
    """
    errors = []
    if node.type == "ERROR" or node.is_missing:
        errors.append(node.start_point[0] + 1)
    for child in node.children:
        errors += traverse_tree(child)
    return errors


def main():
    """
    解析命令行参数并执行 Lint
    """
    if len(sys.argv) < 2:
        print("Usage: python linter.py <file1> <file2> ...")
        sys.exit(1)
    linter = Linter(root=os.getcwd())
    for file_path in sys.argv[1:]:
        errors = linter.lint(file_path)
        if errors:
            print(errors)

if __name__ == "__main__":
    main()
