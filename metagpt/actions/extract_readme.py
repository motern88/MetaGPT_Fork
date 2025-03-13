#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Description: 该脚本定义了 LearnReadMe 类，这是一个从 README.md 文件内容中学习的动作。
Author: mashenquan
Date: 2024-3-20
"""
from pathlib import Path
from typing import Optional

from pydantic import Field

from metagpt.actions import Action
from metagpt.const import GRAPH_REPO_FILE_REPO
from metagpt.schema import Message
from metagpt.utils.common import aread
from metagpt.utils.di_graph_repository import DiGraphRepository
from metagpt.utils.graph_repository import GraphKeyword, GraphRepository


class ExtractReadMe(Action):
    """
    一个从 README.md 文件内容中提取摘要、安装、配置、用法的动作类。

    属性：
        graph_db (Optional[GraphRepository]): 图数据库仓库。
        install_to_path (Optional[str]): 安装路径。
    """

    graph_db: Optional[GraphRepository] = None  # 图数据库
    install_to_path: Optional[str] = Field(default="/TO/PATH")  # 安装路径，默认是"/TO/PATH"
    _readme: Optional[str] = None  # README.md 文件的内容
    _filename: Optional[str] = None  # README.md 文件的文件名

    async def run(self, with_messages=None, **kwargs):
        """
        实现 `Action` 的 `run` 方法。

        参数：
            with_messages (Optional[Type]): 一个可选参数，指定要响应的消息。
        """
        # 获取图数据库仓库的路径
        graph_repo_pathname = self.context.git_repo.workdir / GRAPH_REPO_FILE_REPO / self.context.git_repo.workdir.name
        # 从 JSON 文件加载图数据库
        self.graph_db = await DiGraphRepository.load_from(str(graph_repo_pathname.with_suffix(".json")))

        # 提取并保存 README.md 的摘要
        summary = await self._summarize()
        await self.graph_db.insert(subject=self._filename, predicate=GraphKeyword.HAS_SUMMARY, object_=summary)

        # 提取并保存安装信息
        install = await self._extract_install()
        await self.graph_db.insert(subject=self._filename, predicate=GraphKeyword.HAS_INSTALL, object_=install)

        # 提取并保存配置
        conf = await self._extract_configuration()
        await self.graph_db.insert(subject=self._filename, predicate=GraphKeyword.HAS_CONFIG, object_=conf)

        # 提取并保存用法
        usage = await self._extract_usage()
        await self.graph_db.insert(subject=self._filename, predicate=GraphKeyword.HAS_USAGE, object_=usage)

        # 保存图数据库
        await self.graph_db.save()

        return Message(content="", cause_by=self)

    async def _summarize(self) -> str:
        """
        从 README.md 文件中提取摘要。

        返回：
            str: README.md 文件的摘要。
        """
        readme = await self._get()
        summary = await self.llm.aask(
            readme,
            system_msgs=[
                "你是一个工具，可以总结 git 仓库的 README.md 文件。",
                "返回关于仓库的摘要信息。",
            ],
            stream=False,
        )
        return summary

    async def _extract_install(self) -> str:
        """
        从 README.md 文件中提取安装信息。

        返回：
            str: 安装的 bash 代码块。
        """
        await self._get()
        install = await self.llm.aask(
            self._readme,
            system_msgs=[
                "你是一个工具，可以根据 README.md 文件安装 git 仓库。",
                "返回一个包含以下内容的 markdown bash 代码块：\n"
                f"1. 将仓库克隆到 `{self.install_to_path}` 目录；\n"
                f"2. 切换到 `{self.install_to_path}` 目录；\n"
                f"3. 安装该仓库。",
            ],
            stream=False,
        )
        return install

    async def _extract_configuration(self) -> str:
        """
        从 README.md 文件中提取配置相关的信息。

        返回：
            str: 配置的 bash 代码块。
        """
        await self._get()
        configuration = await self.llm.aask(
            self._readme,
            system_msgs=[
                "你是一个工具，可以根据 README.md 文件配置 git 仓库。",
                "返回一个配置仓库的 markdown bash 代码块，如果没有配置则返回空的 bash 代码块。",
            ],
            stream=False,
        )
        return configuration

    async def _extract_usage(self) -> str:
        """
        从 README.md 文件中提取用法信息。

        返回：
            str: 用法的 markdown 代码块列表。
        """
        await self._get()
        usage = await self.llm.aask(
            self._readme,
            system_msgs=[
                "你是一个工具，可以根据 README.md 文件总结 git 仓库的所有用法。",
                "返回一个包含仓库用法的 markdown 代码块的列表。",
            ],
            stream=False,
        )
        return usage

    async def _get(self) -> str:
        """
        获取 README.md 文件的内容。

        返回：
            str: README.md 文件的内容。
        """
        if self._readme is not None:
            return self._readme

        root = Path(self.i_context).resolve()
        filename = None
        for file_path in root.iterdir():
            if file_path.is_file() and file_path.stem == "README":
                filename = file_path
                break

        if not filename:
            return ""

        self._readme = await aread(filename=filename, encoding="utf-8")
        self._filename = str(filename)
        return self._readme
