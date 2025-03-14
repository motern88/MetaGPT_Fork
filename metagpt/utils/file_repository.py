#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : git_repository.py
@Desc: File repository management. RFC 135 2.2.3.2, 2.2.3.4 and 2.2.3.13.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from metagpt.logs import logger
from metagpt.schema import Document
from metagpt.utils.common import aread, awrite
from metagpt.utils.json_to_markdown import json_to_markdown


class FileRepository:
    """代表与 Git 仓库关联的文件仓库类。

    :param git_repo: 关联的 GitRepository 实例。
    :param relative_path: Git 仓库中的相对路径。

    属性：
        _relative_path (Path): Git 仓库中的相对路径。
        _git_repo (GitRepository): 关联的 GitRepository 实例。
    """

    def __init__(self, git_repo, relative_path: Path = Path(".")):
        """初始化 FileRepository 实例。

        :param git_repo: 关联的 GitRepository 实例。
        :param relative_path: Git 仓库中的相对路径。
        """
        self._relative_path = relative_path
        self._git_repo = git_repo

        # 初始化
        self.workdir.mkdir(parents=True, exist_ok=True)

    async def save(self, filename: Path | str, content, dependencies: List[str] = None) -> Document:
        """将内容保存到文件并更新其依赖关系。

        :param filename: 文件名或仓库中的路径。
        :param content: 要保存的内容。
        :param dependencies: 依赖文件名或路径列表。
        """
        pathname = self.workdir / filename
        pathname.parent.mkdir(parents=True, exist_ok=True)
        content = content if content else ""  # 避免`argument must be str, not None`，使其继续执行
        await awrite(filename=str(pathname), data=content)
        logger.info(f"保存到: {str(pathname)}")

        if dependencies is not None:
            dependency_file = await self._git_repo.get_dependency()
            await dependency_file.update(pathname, set(dependencies))
            logger.info(f"更新依赖: {str(pathname)}:{dependencies}")

        return Document(root_path=str(self._relative_path), filename=str(filename), content=content)

    async def get_dependency(self, filename: Path | str) -> Set[str]:
        """获取文件的依赖关系。

        :param filename: 文件名或仓库中的路径。
        :return: 依赖文件名或路径的集合。
        """
        pathname = self.workdir / filename
        dependency_file = await self._git_repo.get_dependency()
        return await dependency_file.get(pathname)

    async def get_changed_dependency(self, filename: Path | str) -> Set[str]:
        """获取已更改的文件的依赖关系。

        :param filename: 文件名或仓库中的路径。
        :return: 已更改的依赖文件名或路径的集合。
        """
        dependencies = await self.get_dependency(filename=filename)
        changed_files = set(self.changed_files.keys())
        changed_dependent_files = set()
        for df in dependencies:
            rdf = Path(df).relative_to(self._relative_path)
            if str(rdf) in changed_files:
                changed_dependent_files.add(df)
        return changed_dependent_files

    async def get(self, filename: Path | str) -> Document | None:
        """读取文件内容。

        :param filename: 文件名或仓库中的路径。
        :return: 文件内容。
        """
        doc = Document(root_path=str(self.root_path), filename=str(filename))
        path_name = self.workdir / filename
        if not path_name.exists():
            return None
        if not path_name.is_file():
            return None
        doc.content = await aread(path_name)
        return doc

    async def get_all(self, filter_ignored=True) -> List[Document]:
        """获取仓库中所有文件的内容。

        :return: 表示文件的 Document 实例列表。
        """
        docs = []
        if filter_ignored:
            for f in self.all_files:
                doc = await self.get(f)
                docs.append(doc)
        else:
            for root, dirs, files in os.walk(str(self.workdir)):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.workdir)
                    doc = await self.get(relative_path)
                    docs.append(doc)
        return docs

    @property
    def workdir(self):
        """返回文件仓库的工作目录的绝对路径。

        :return: 工作目录的绝对路径。
        """
        return self._git_repo.workdir / self._relative_path

    @property
    def root_path(self):
        """返回从 Git 仓库根目录开始的相对路径"""
        return self._relative_path

    @property
    def changed_files(self) -> Dict[str, str]:
        """返回已更改文件的字典及其更改类型。

        :return: 一个字典，键为文件路径，值为更改类型。
        """
        files = self._git_repo.changed_files
        relative_files = {}
        for p, ct in files.items():
            if ct.value == "D":  # 删除
                continue
            try:
                rf = Path(p).relative_to(self._relative_path)
            except ValueError:
                continue
            relative_files[str(rf)] = ct
        return relative_files

    @property
    def all_files(self) -> List:
        """获取仓库中的所有文件。

        字典包括相对于当前 FileRepository 的文件路径。

        :return: 一个字典，键为文件路径，值为文件信息。
        :rtype: List
        """
        return self._git_repo.get_files(relative_path=self._relative_path)

    def get_change_dir_files(self, dir: Path | str) -> List:
        """获取指定目录中已更改的文件。

        :param dir: 仓库中的目录路径。
        :return: 指定目录下已更改的文件名或路径列表。
        """
        changed_files = self.changed_files
        children = []
        for f in changed_files:
            try:
                Path(f).relative_to(Path(dir))
            except ValueError:
                continue
            children.append(str(f))
        return children

    @staticmethod
    def new_filename():
        """基于当前时间戳和 UUID 后缀生成一个新的文件名。

        :return: 新的文件名字符串。
        """
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        return current_time

    async def save_doc(self, doc: Document, dependencies: List[str] = None):
        """保存内容到文件并更新其依赖关系。

        :param doc: 要保存的 Document 实例。
        :type doc: Document
        :param dependencies: 要保存文件的依赖关系列表。
        :type dependencies: List[str], 可选
        """

        doc = await self.save(filename=doc.filename, content=doc.content, dependencies=dependencies)
        logger.debug(f"文件已保存: {str(doc.filename)}")
        return doc

    async def save_pdf(self, doc: Document, with_suffix: str = ".md", dependencies: List[str] = None):
        """将 Document 实例保存为 PDF 文件。

        该方法将 Document 实例的内容转换为 Markdown，保存到具有可选后缀的文件，并记录保存的文件。

        :param doc: 要保存的 Document 实例。
        :type doc: Document
        :param with_suffix: 可选的后缀，附加到保存文件名上。
        :type with_suffix: str, 可选
        :param dependencies: 要保存文件的依赖关系列表。
        :type dependencies: List[str], 可选
        """
        m = json.loads(doc.content)
        filename = Path(doc.filename).with_suffix(with_suffix) if with_suffix is not None else Path(doc.filename)
        doc = await self.save(filename=str(filename), content=json_to_markdown(m), dependencies=dependencies)
        logger.debug(f"文件已保存: {str(filename)}")
        return doc

    async def delete(self, filename: Path | str):
        """从文件仓库中删除文件。

        该方法根据提供的文件名删除文件。

        :param filename: 要删除的文件名或路径。
        :type filename: Path 或 str
        """
        pathname = self.workdir / filename
        if not pathname.exists():
            return
        pathname.unlink(missing_ok=True)

        dependency_file = await self._git_repo.get_dependency()
        await dependency_file.update(filename=pathname, dependencies=None)
        logger.info(f"删除依赖键: {str(pathname)}")
