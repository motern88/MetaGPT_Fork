#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/3/11
@Author  : mashenquan
@File    : tree.py
@Desc    : Implement the same functionality as the `tree` command.
        Example:
            >>> print_tree(".")
            utils
            +-- serialize.py
            +-- project_repo.py
            +-- tree.py
            +-- mmdc_playwright.py
            +-- cost_manager.py
            +-- __pycache__
            |   +-- __init__.cpython-39.pyc
            |   +-- redis.cpython-39.pyc
            |   +-- singleton.cpython-39.pyc
            |   +-- embedding.cpython-39.pyc
            |   +-- make_sk_kernel.cpython-39.pyc
            |   +-- file_repository.cpython-39.pyc
            +-- file.py
            +-- save_code.py
            +-- common.py
            +-- redis.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from gitignore_parser import parse_gitignore

from metagpt.tools.libs.shell import shell_execute


async def tree(root: str | Path, gitignore: str | Path = None, run_command: bool = False) -> str:
    """
    递归遍历目录结构，并以树形格式输出。

    参数:
        root (str 或 Path): 从该目录开始遍历的根目录。
        gitignore (str 或 Path): gitignore 文件的路径。
        run_command (bool): 是否执行 `tree` 命令。如果为 True，则执行 `tree` 命令并返回结果，
            否则使用 Python 代码执行。

    返回:
        str: 目录树的字符串表示。

    示例:
            >>> tree(".")
            utils
            +-- serialize.py
            +-- project_repo.py
            +-- tree.py
            +-- mmdc_playwright.py
            +-- __pycache__
            |   +-- __init__.cpython-39.pyc
            |   +-- redis.cpython-39.pyc
            |   +-- singleton.cpython-39.pyc
            +-- parse_docstring.py

            >>> tree(".", gitignore="../../.gitignore")
            utils
            +-- serialize.py
            +-- project_repo.py
            +-- tree.py
            +-- mmdc_playwright.py
            +-- parse_docstring.py

            >>> tree(".", gitignore="../../.gitignore", run_command=True)
            utils
            ├── serialize.py
            ├── project_repo.py
            ├── tree.py
            ├── mmdc_playwright.py
            └── parse_docstring.py
    """
    root = Path(root).resolve()  # 获取绝对路径
    if run_command:
        # 如果需要执行tree命令
        return await _execute_tree(root, gitignore)

    # 如果提供了gitignore文件，解析其中的规则
    git_ignore_rules = parse_gitignore(gitignore) if gitignore else None
    # 获取根目录下的子文件信息
    dir_ = {root.name: _list_children(root=root, git_ignore_rules=git_ignore_rules)}
    v = _print_tree(dir_)
    return "\n".join(v)


def _list_children(root: Path, git_ignore_rules: Callable) -> Dict[str, Dict]:
    """
    列出指定目录下的所有子文件和子目录，并根据gitignore规则忽略某些文件。

    参数:
        root (Path): 根目录
        git_ignore_rules (Callable): 用于检查是否需要忽略文件的规则

    返回:
        dict: 子目录和文件的字典
    """
    dir_ = {}
    for i in root.iterdir():
        # 如果该文件被gitignore规则排除，则跳过
        if git_ignore_rules and git_ignore_rules(str(i)):
            continue
        try:
            # 如果是文件，直接加入
            if i.is_file():
                dir_[i.name] = {}
            else:
                # 如果是目录，递归列出子文件
                dir_[i.name] = _list_children(root=i, git_ignore_rules=git_ignore_rules)
        except (FileNotFoundError, PermissionError, OSError):
            # 如果遇到文件权限问题等错误，跳过
            dir_[i.name] = {}
    return dir_


def _print_tree(dir_: Dict[str, Dict]) -> List[str]:
    """
    打印目录树。

    参数:
        dir_ (dict): 目录结构的字典

    返回:
        list: 目录树的字符串列表
    """
    ret = []
    for name, children in dir_.items():
        ret.append(name)  # 添加当前文件/目录名
        if not children:
            continue
        lines = _print_tree(children)  # 递归打印子目录
        for j, v in enumerate(lines):
            # 根据行的前缀符号决定如何打印
            if v[0] not in ["+", " ", "|"]:
                ret = _add_line(ret)
                row = f"+-- {v}"
            else:
                row = f"    {v}"
            ret.append(row)
    return ret


def _add_line(rows: List[str]) -> List[str]:
    """
    为目录树添加分支线条，确保正确的树形结构。

    参数:
        rows (list): 目录树的行列表

    返回:
        list: 添加了分支线条的目录树行列表
    """
    for i in range(len(rows) - 1, -1, -1):
        v = rows[i]
        if v[0] != " ":
            return rows
        rows[i] = "|" + v[1:]  # 替换空格为竖线
    return rows


async def _execute_tree(root: Path, gitignore: str | Path) -> str:
    """
    使用 `tree` 命令执行目录树生成。

    参数:
        root (Path): 根目录
        gitignore (str 或 Path): gitignore 文件路径

    返回:
        str: 执行 `tree` 命令后的输出结果
    """
    args = ["--gitfile", str(gitignore)] if gitignore else []  # 如果有gitignore文件，则传递给命令
    stdout, _, _ = await shell_execute(["tree"] + args + [str(root)])  # 执行tree命令
    return stdout
