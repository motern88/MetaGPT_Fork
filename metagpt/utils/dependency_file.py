#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/22
@Author  : mashenquan
@File    : dependency_file.py
@Desc: Implementation of the dependency file described in Section 2.2.3.2 of RFC 135.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set

from metagpt.utils.common import aread, awrite
from metagpt.utils.exceptions import handle_exception


class DependencyFile:
    """表示一个依赖文件类，用于管理依赖关系。

    :param workdir: 依赖文件的工作目录路径。
    """

    def __init__(self, workdir: Path | str):
        """初始化一个 DependencyFile 实例。

        :param workdir: 依赖文件的工作目录路径。
        """
        self._dependencies = {}  # 用于存储依赖关系的字典
        self._filename = Path(workdir) / ".dependencies.json"  # 依赖文件的路径

    async def load(self):
        """异步加载依赖关系数据从文件中。"""
        if not self._filename.exists():  # 如果依赖文件不存在，直接返回
            return
        json_data = await aread(self._filename)  # 异步读取文件内容
        json_data = re.sub(r"\\+", "/", json_data)  # 将Windows路径中的反斜杠替换为正斜杠，以兼容Windows
        self._dependencies = json.loads(json_data)  # 解析JSON数据

    @handle_exception
    async def save(self):
        """异步将当前依赖关系保存到文件。"""
        data = json.dumps(self._dependencies)  # 将依赖字典转为JSON字符串
        await awrite(filename=self._filename, data=data)  # 异步写入文件

    async def update(self, filename: Path | str, dependencies: Set[Path | str], persist=True):
        """异步更新某个文件的依赖关系。

        :param filename: 文件名或路径。
        :param dependencies: 该文件的依赖集合。
        :param persist: 是否立即持久化更改到文件。
        """
        if persist:
            await self.load()  # 如果需要持久化，先加载依赖数据

        root = self._filename.parent  # 获取依赖文件所在的根目录
        try:
            key = Path(filename).relative_to(root).as_posix()  # 将文件路径转换为相对路径
        except ValueError:
            key = filename  # 如果无法转换为相对路径，则保持原路径
        key = str(key)  # 将路径转换为字符串
        if dependencies:
            relative_paths = []
            for i in dependencies:
                try:
                    s = str(Path(i).relative_to(root).as_posix())  # 处理每个依赖的相对路径
                except ValueError:
                    s = str(i)  # 如果无法转换为相对路径，则保持原路径
                relative_paths.append(s)

            self._dependencies[key] = relative_paths  # 更新依赖关系字典
        elif key in self._dependencies:
            del self._dependencies[key]  # 如果没有依赖项，删除该文件的依赖记录

        if persist:
            await self.save()  # 如果需要持久化，保存依赖文件

    async def get(self, filename: Path | str, persist=True):
        """异步获取某个文件的依赖关系。

        :param filename: 文件名或路径。
        :param persist: 是否立即加载依赖关系。
        :return: 返回依赖集合。
        """
        if persist:
            await self.load()  # 如果需要持久化，先加载依赖数据

        root = self._filename.parent  # 获取依赖文件所在的根目录
        try:
            key = Path(filename).relative_to(root).as_posix()  # 将文件路径转换为相对路径
        except ValueError:
            key = Path(filename).as_posix()  # 如果无法转换为相对路径，则直接使用绝对路径
        return set(self._dependencies.get(str(key), {}))  # 返回文件的依赖集合

    def delete_file(self):
        """删除依赖文件。"""
        self._filename.unlink(missing_ok=True)  # 删除文件，如果文件不存在不抛出异常

    @property
    def exists(self):
        """检查依赖文件是否存在。"""
        return self._filename.exists()  # 返回依赖文件是否存在的布尔值
