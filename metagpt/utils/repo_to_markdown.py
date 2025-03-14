#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file provides functionality to convert a local repository into a markdown representation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Union

from gitignore_parser import parse_gitignore

from metagpt.logs import logger
from metagpt.utils.common import (
    aread,
    awrite,
    get_markdown_codeblock_type,
    get_mime_type,
    list_files,
)
from metagpt.utils.tree import tree


async def repo_to_markdown(repo_path: str | Path, output: str | Path = None) -> str:
    """
    将本地仓库转换为Markdown格式。

    该函数接受本地仓库的路径，并生成仓库结构的Markdown表示，包括目录树和文件列表。

    参数：
        repo_path (str | Path): 本地仓库的路径。
        output (str | Path, optional): 生成的Markdown文件保存路径。默认为None。

    返回：
        str: 仓库的Markdown表示。
    """
    repo_path = Path(repo_path).resolve()
    gitignore_file = repo_path / ".gitignore"

    # 写入目录树
    markdown = await _write_dir_tree(repo_path=repo_path, gitignore=gitignore_file)

    # 解析.gitignore文件中的规则
    gitignore_rules = parse_gitignore(full_path=str(gitignore_file)) if gitignore_file.exists() else None
    # 写入仓库中的文件列表
    markdown += await _write_files(repo_path=repo_path, gitignore_rules=gitignore_rules)

    if output:
        output_file = Path(output).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        await awrite(filename=str(output_file), data=markdown, encoding="utf-8")
        logger.info(f"保存到: {output_file}")
    return markdown



async def _write_dir_tree(repo_path: Path, gitignore: Path) -> str:
    try:
        # 获取目录树内容
        content = await tree(repo_path, gitignore, run_command=True)
    except Exception as e:
        logger.info(f"{e}, 使用安全模式.")
        content = await tree(repo_path, gitignore, run_command=False)

    doc = f"## 目录树\n```text\n{content}\n```\n---\n\n"
    return doc


async def _write_files(repo_path, gitignore_rules=None) -> str:
    filenames = list_files(repo_path)
    markdown = ""
    pattern = r"^\..*"  # 隐藏的文件夹/文件
    for filename in filenames:
        # 如果文件在.gitignore中，则跳过
        if gitignore_rules and gitignore_rules(str(filename)):
            continue
        ignore = False
        for i in filename.parts:
            if re.match(pattern, i):
                ignore = True
                break
        if ignore:
            continue
        markdown += await _write_file(filename=filename, repo_path=repo_path)
    return markdown


async def _write_file(filename: Path, repo_path: Path) -> str:
    # 检查文件是否为文本文件
    is_text, mime_type = await is_text_file(filename)
    if not is_text:
        logger.info(f"忽略内容: {filename}")
        return ""

    try:
        # 获取文件的相对路径，并生成Markdown
        relative_path = filename.relative_to(repo_path)
        markdown = f"## {relative_path}\n"
        content = await aread(filename, encoding="utf-8")
        # 转义Markdown中的特殊字符
        content = content.replace("```", "\\`\\`\\`").replace("---", "\\-\\-\\-")
        code_block_type = get_markdown_codeblock_type(filename.name)
        markdown += f"```{code_block_type}\n{content}\n```\n---\n\n"
        return markdown
    except Exception as e:
        logger.error(e)
        return ""


async def is_text_file(filename: Union[str, Path]) -> Tuple[bool, str]:
    """
    判断指定文件是否为文本文件，依据其MIME类型。

    参数：
        filename (Union[str, Path]): 文件的路径。

    返回：
        Tuple[bool, str]: 返回一个元组，第一个元素表示文件是否为文本文件（True表示是文本文件，False表示不是），
                          第二个元素为文件的MIME类型。
    """
    pass_set = {
        "application/json",
        "application/vnd.chipnuts.karaoke-mmd",
        "application/javascript",
        "application/xml",
        "application/x-sh",
        "application/sql",
    }
    denied_set = {
        "application/zlib",
        "application/octet-stream",
        "image/svg+xml",
        "application/pdf",
        "application/msword",
        "application/vnd.ms-excel",
        "audio/x-wav",
        "application/x-git",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "image/jpeg",
        "audio/mpeg",
        "video/mp2t",
        "inode/x-empty",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "image/png",
        "image/vnd.microsoft.icon",
        "video/mp4",
    }
    mime_type = await get_mime_type(Path(filename), force_read=True)
    v = "text/" in mime_type or mime_type in pass_set
    if v:
        return True, mime_type

    if mime_type not in denied_set:
        logger.info(mime_type)
    return False, mime_type
