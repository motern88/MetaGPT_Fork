#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : visualize_graph.py
@Desc    : Visualization tool to visualize the class diagrams or sequence diagrams of the graph repository.
"""
from __future__ import annotations

import re
from abc import ABC
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from metagpt.const import AGGREGATION, COMPOSITION, GENERALIZATION
from metagpt.schema import UMLClassView
from metagpt.utils.common import split_namespace
from metagpt.utils.di_graph_repository import DiGraphRepository
from metagpt.utils.graph_repository import GraphKeyword, GraphRepository


class _VisualClassView(BaseModel):
    """内部使用的保护类，用于VisualGraphRepo。

    属性:
        package (str): 与类相关联的包名。
        uml (Optional[UMLClassView]): 与类关联的可选 UMLClassView。
        generalizations (List[str]): 类的泛化列表。
        compositions (List[str]): 类的组合列表。
        aggregations (List[str]): 类的聚合列表。
    """

    package: str
    uml: Optional[UMLClassView] = None  # UML类视图，可选
    generalizations: List[str] = Field(default_factory=list)  # 泛化列表
    compositions: List[str] = Field(default_factory=list)  # 组合列表
    aggregations: List[str] = Field(default_factory=list)  # 聚合列表

    def get_mermaid(self, align: int = 1) -> str:
        """生成Markdown格式的Mermaid类图。

        参数:
            align (int): 用于对齐的缩进级别。

        返回:
            str: 生成的Mermaid类图Markdown文本。
        """
        if not self.uml:
            return ""
        prefix = "\t" * align  # 根据对齐级别生成前缀

        mermaid_txt = self.uml.get_mermaid(align=align)  # 获取UML类视图的Mermaid文本
        for i in self.generalizations:
            mermaid_txt += f"{prefix}{i} <|-- {self.name}\n"  # 添加泛化关系
        for i in self.compositions:
            mermaid_txt += f"{prefix}{i} *-- {self.name}\n"  # 添加组合关系
        for i in self.aggregations:
            mermaid_txt += f"{prefix}{i} o-- {self.name}\n"  # 添加聚合关系
        return mermaid_txt

    @property
    def name(self) -> str:
        """返回类名，不包含命名空间前缀。"""
        return split_namespace(self.package)[-1]  # 获取类名（去掉命名空间部分）


class VisualGraphRepo(ABC):
    """VisualGraphRepo的抽象基类。"""

    graph_db: GraphRepository  # 图数据库实例

    def __init__(self, graph_db):
        self.graph_db = graph_db  # 初始化图数据库


class VisualDiGraphRepo(VisualGraphRepo):
    """VisualGraphRepo的DiGraph图仓库实现。

    该类扩展了VisualGraphRepo，提供了特定的功能，用于DiGraph图仓库。
    """

    @classmethod
    async def load_from(cls, filename: str | Path):
        """从文件加载一个VisualDiGraphRepo实例。"""
        graph_db = await DiGraphRepository.load_from(str(filename))  # 从文件加载图数据库
        return cls(graph_db=graph_db)  # 返回实例

    async def get_mermaid_class_view(self) -> str:
        """返回Markdown格式的Mermaid类图代码块对象。"""
        rows = await self.graph_db.select(predicate=GraphKeyword.IS, object_=GraphKeyword.CLASS)
        mermaid_txt = "classDiagram\n"  # Mermaid类图的起始部分
        for r in rows:
            v = await self._get_class_view(ns_class_name=r.subject)
            mermaid_txt += v.get_mermaid()  # 获取每个类的Mermaid类图文本
        return mermaid_txt

    async def _get_class_view(self, ns_class_name: str) -> _VisualClassView:
        """返回指定类的Mermaid类图代码块对象。"""
        rows = await self.graph_db.select(subject=ns_class_name)
        class_view = _VisualClassView(package=ns_class_name)  # 创建类视图
        for r in rows:
            if r.predicate == GraphKeyword.HAS_CLASS_VIEW:
                class_view.uml = UMLClassView.model_validate_json(r.object_)  # 获取UML类视图
            elif r.predicate == GraphKeyword.IS + GENERALIZATION + GraphKeyword.OF:
                name = split_namespace(r.object_)[-1]
                name = self._refine_name(name)
                if name:
                    class_view.generalizations.append(name)  # 添加泛化关系
            elif r.predicate == GraphKeyword.IS + COMPOSITION + GraphKeyword.OF:
                name = split_namespace(r.object_)[-1]
                name = self._refine_name(name)
                if name:
                    class_view.compositions.append(name)  # 添加组合关系
            elif r.predicate == GraphKeyword.IS + AGGREGATION + GraphKeyword.OF:
                name = split_namespace(r.object_)[-1]
                name = self._refine_name(name)
                if name:
                    class_view.aggregations.append(name)  # 添加聚合关系
        return class_view

    async def get_mermaid_sequence_views(self) -> List[(str, str)]:
        """返回所有Markdown格式的序列图及其对应的图仓库键。"""
        sequence_views = []
        rows = await self.graph_db.select(predicate=GraphKeyword.HAS_SEQUENCE_VIEW)
        for r in rows:
            sequence_views.append((r.subject, r.object_))  # 收集序列图及其键
        return sequence_views

    @staticmethod
    def _refine_name(name: str) -> str:
        """去除名称中的杂质内容。

        示例:
            >>> _refine_name("int")
            ""

            >>> _refine_name('"Class1"')
            'Class1'

            >>> _refine_name("pkg.Class1")
            "Class1"
        """
        name = re.sub(r'^[\'"\\\(\)]+|[\'"\\\(\)]+$', "", name)  # 去掉名称两边的特殊字符
        if name in ["int", "float", "bool", "str", "list", "tuple", "set", "dict", "None"]:
            return ""  # 排除Python内建类型
        if "." in name:
            name = name.split(".")[-1]  # 去掉包名部分
        return name

    async def get_mermaid_sequence_view_versions(self) -> List[(str, str)]:
        """返回所有版本化的Markdown格式序列图及其对应的图仓库键。"""
        sequence_views = []
        rows = await self.graph_db.select(predicate=GraphKeyword.HAS_SEQUENCE_VIEW_VER)
        for r in rows:
            sequence_views.append((r.subject, r.object_))  # 收集版本化的序列图及其键
        return sequence_views
