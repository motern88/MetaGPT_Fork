#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : di_graph_repository.py
@Desc    : Graph repository based on DiGraph.
    This script defines a graph repository class based on a directed graph (DiGraph), providing functionalities
    specific to handling directed relationships between entities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import networkx

from metagpt.utils.common import aread, awrite
from metagpt.utils.graph_repository import SPO, GraphRepository


class DiGraphRepository(GraphRepository):
    """基于有向图 (DiGraph) 的图存储库。"""

    def __init__(self, name: str | Path, **kwargs):
        super().__init__(name=str(name), **kwargs)
        self._repo = networkx.DiGraph()

    async def insert(self, subject: str, predicate: str, object_: str):
        """向有向图存储库中插入一个新的三元组。

        参数:
            subject (str): 三元组的主语。
            predicate (str): 描述关系的谓词。
            object_ (str): 三元组的宾语。

        示例:
            await my_di_graph_repo.insert(subject="Node1", predicate="connects_to", object_="Node2")
            # 插入一个有向关系：Node1 connects_to Node2
        """
        self._repo.add_edge(subject, object_, predicate=predicate)

    async def delete(self, subject: str = None, predicate: str = None, object_: str = None) -> int:
        """根据指定的条件删除有向图存储库中的三元组。

        参数:
            subject (str, 可选): 要过滤的三元组的主语。
            predicate (str, 可选): 要过滤的关系谓词。
            object_ (str, 可选): 要过滤的三元组的宾语。

        返回:
            int: 从存储库中删除的三元组数量。

        示例:
            deleted_count = await my_di_graph_repo.delete(subject="Node1", predicate="connects_to")
            # 删除所有主语为 Node1，谓词为 'connects_to' 的有向关系。
        """
        rows = await self.select(subject=subject, predicate=predicate, object_=object_)
        if not rows:
            return 0
        for r in rows:
            self._repo.remove_edge(r.subject, r.object_)
        return len(rows)

    async def select(self, subject: str = None, predicate: str = None, object_: str = None) -> List[SPO]:
        """根据指定的条件从有向图存储库中检索三元组。

        参数:
            subject (str, 可选): 要过滤的三元组的主语。
            predicate (str, 可选): 要过滤的关系谓词。
            object_ (str, 可选): 要过滤的三元组的宾语。

        返回:
            List[SPO]: 返回满足条件的三元组列表。

        示例:
            selected_triples = await my_di_graph_repo.select(subject="Node1", predicate="connects_to")
            # 检索所有主语为 Node1，谓词为 'connects_to' 的有向关系。
        """
        result = []
        for s, o, p in self._repo.edges(data="predicate"):
            if subject and subject != s:
                continue
            if predicate and predicate != p:
                continue
            if object_ and object_ != o:
                continue
            result.append(SPO(subject=s, predicate=p, object_=o))
        return result


    def json(self) -> str:
        """将有向图存储库转换为 JSON 格式的字符串。"""
        m = networkx.node_link_data(self._repo)
        data = json.dumps(m)
        return data

    async def save(self, path: str | Path = None):
        """将有向图存储库保存为 JSON 文件。

        参数:
            path (Union[str, Path], 可选): 保存 JSON 文件的目录路径。
                如果未提供，将使用关键字参数中的 'root' 键的默认路径。
        """
        data = self.json()
        path = path or self._kwargs.get("root")
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        pathname = Path(path) / self.name
        await awrite(filename=pathname.with_suffix(".json"), data=data, encoding="utf-8")

    async def load(self, pathname: str | Path):
        """从 JSON 文件加载有向图存储库。"""
        data = await aread(filename=pathname, encoding="utf-8")
        self.load_json(data)

    def load_json(self, val: str):
        """
        加载表示图结构的 JSON 编码字符串，并使用解析后的图更新内部存储库 (_repo)。

        参数:
            val (str): 表示图结构的 JSON 编码字符串。

        返回:
            self: 返回更新后的类实例，内部存储库 (_repo) 已更新。

        异常:
            TypeError: 如果 val 不是有效的 JSON 字符串或无法解析为有效的图结构。
        """
        if not val:
            return self
        m = json.loads(val)
        self._repo = networkx.node_link_graph(m)
        return self

    @staticmethod
    async def load_from(pathname: str | Path) -> GraphRepository:
        """从 JSON 文件创建并加载有向图存储库。

        参数:
            pathname (Union[str, Path]): 要加载的 JSON 文件的路径。

        返回:
            GraphRepository: 从指定 JSON 文件加载的图存储库的新实例。
        """
        pathname = Path(pathname)
        graph = DiGraphRepository(name=pathname.stem, root=pathname.parent)
        if pathname.exists():
            await graph.load(pathname=pathname)
        return graph

    @property
    def root(self) -> str:
        """返回图存储库文件的根目录路径。"""
        return self._kwargs.get("root")

    @property
    def pathname(self) -> Path:
        """返回图存储库文件的路径和文件名。"""
        p = Path(self.root) / self.name
        return p.with_suffix(".json")

    @property
    def repo(self):
        """获取底层的有向图存储库。"""
        return self._repo
