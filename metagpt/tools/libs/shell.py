#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union


async def shell_execute(
    command: Union[List[str], str], cwd: str | Path = None, env: Dict = None, timeout: int = 600
) -> Tuple[str, str, int]:
    """
    异步执行命令，并返回标准输出、标准错误和返回码。

    参数：
        command (Union[List[str], str]): 要执行的命令及其参数，可以是字符串列表或单个字符串。
        cwd (str | Path, 可选): 命令的当前工作目录，默认为 None。
        env (Dict, 可选): 设置命令执行的环境变量，默认为 None。
        timeout (int, 可选): 命令执行的超时时间（秒），默认为 600。

    返回：
        Tuple[str, str, int]: 包含标准输出、标准错误和返回码的元组。

    异常：
        ValueError: 如果命令超时，将抛出此错误，错误消息包含超时进程的标准输出和标准错误。

    示例：
        >>> # 使用列表作为命令
        >>> stdout, stderr, returncode = await shell_execute(command=["ls", "-l"], cwd="/home/user", env={"PATH": "/usr/bin"})
        >>> print(stdout)
        total 8
        -rw-r--r-- 1 user user    0 Mar 22 10:00 file1.txt
        -rw-r--r-- 1 user user    0 Mar 22 10:00 file2.txt
        ...

        >>> # 使用字符串作为命令
        >>> stdout, stderr, returncode = await shell_execute(command="ls -l", cwd="/home/user", env={"PATH": "/usr/bin"})
        >>> print(stdout)
        total 8
        -rw-r--r-- 1 user user    0 Mar 22 10:00 file1.txt
        -rw-r--r-- 1 user user    0 Mar 22 10:00 file2.txt
        ...

    参考：
        该函数使用 `subprocess.run` 以异步方式执行 shell 命令。
    """
    cwd = str(cwd) if cwd else None
    shell = True if isinstance(command, str) else False
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout, shell=shell)
    return result.stdout, result.stderr, result.returncode

