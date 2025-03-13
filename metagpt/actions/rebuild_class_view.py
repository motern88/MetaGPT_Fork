#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : rebuild_class_view.py
@Desc    : Reconstructs class diagram from a source code project.
    Implement RFC197, https://deepwisdom.feishu.cn/wiki/VyK0wfq56ivuvjklMKJcmHQknGt
"""

from pathlib import Path
from typing import Optional, Set, Tuple

import aiofiles

from metagpt.actions import Action
from metagpt.const import (
    AGGREGATION,
    COMPOSITION,
    DATA_API_DESIGN_FILE_REPO,
    GENERALIZATION,
    GRAPH_REPO_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.repo_parser import DotClassInfo, RepoParser
from metagpt.schema import UMLClassView
from metagpt.utils.common import concat_namespace, split_namespace
from metagpt.utils.di_graph_repository import DiGraphRepository
from metagpt.utils.graph_repository import GraphKeyword, GraphRepository


class RebuildClassView(Action):
    """
    重建类图的图谱存储库，基于源代码项目。

    属性：
        graph_db (Optional[GraphRepository]): 可选的图谱存储库。
    """

    graph_db: Optional[GraphRepository] = None

    async def run(self, with_messages=None, format=None):
        """
        实现 `Action` 的 `run` 方法。

        参数：
            with_messages (Optional[Type]): 可选参数，指定要响应的消息。
            format (str): 提示模式的格式。
        """
        # 如果没有指定格式，则使用默认的配置格式
        format = format if format else self.config.prompt_schema

        # 设置图谱存储库路径
        graph_repo_pathname = self.context.git_repo.workdir / GRAPH_REPO_FILE_REPO / self.context.git_repo.workdir.name
        # 从 JSON 文件加载图谱存储库
        self.graph_db = await DiGraphRepository.load_from(str(graph_repo_pathname.with_suffix(".json")))

        # 创建源代码库解析器
        repo_parser = RepoParser(base_directory=Path(self.i_context))

        # 使用 pylint 重建类视图、关系视图和包根目录
        class_views, relationship_views, package_root = await repo_parser.rebuild_class_views(path=Path(self.i_context))

        # 更新图谱存储库的类视图
        await GraphRepository.update_graph_db_with_class_views(self.graph_db, class_views)

        # 更新图谱存储库的类关系视图
        await GraphRepository.update_graph_db_with_class_relationship_views(self.graph_db, relationship_views)

        # 重建组合关系
        await GraphRepository.rebuild_composition_relationship(self.graph_db)

        # 使用 ast 来分析源代码
        direction, diff_path = self._diff_path(path_root=Path(self.i_context).resolve(), package_root=package_root)

        # 生成符号信息
        symbols = repo_parser.generate_symbols()

        # 对每个文件信息进行路径对齐并更新图谱
        for file_info in symbols:
            file_info.file = self._align_root(file_info.file, direction, diff_path)
            await GraphRepository.update_graph_db_with_file_info(self.graph_db, file_info)

        # 创建 mermaid 类图
        await self._create_mermaid_class_views()

        # 保存图谱数据库
        await self.graph_db.save()

    async def _create_mermaid_class_views(self) -> str:
        """使用图谱数据库中的数据创建 Mermaid 类图。

        此方法使用存储在图谱数据库中的信息来生成 Mermaid 类图。
        返回：
            str: mermaid 类图的文件名。
        """
        # 设置 mermaid 类图存储路径
        path = self.context.git_repo.workdir / DATA_API_DESIGN_FILE_REPO
        path.mkdir(parents=True, exist_ok=True)
        pathname = path / self.context.git_repo.workdir.name
        filename = str(pathname.with_suffix(".class_diagram.mmd"))

        # 异步写入 Mermaid 类图
        async with aiofiles.open(filename, mode="w", encoding="utf-8") as writer:
            content = "classDiagram\n"
            logger.debug(content)
            await writer.write(content)

            # 获取所有类的行并生成类的 Mermaid 图
            rows = await self.graph_db.select(predicate=GraphKeyword.IS, object_=GraphKeyword.CLASS)
            class_distinct = set()
            relationship_distinct = set()

            for r in rows:
                content = await self._create_mermaid_class(r.subject)
                if content:
                    await writer.write(content)
                    class_distinct.add(r.subject)

            # 获取所有类的关系并生成 Mermaid 关系图
            for r in rows:
                content, distinct = await self._create_mermaid_relationship(r.subject)
                if content:
                    logger.debug(content)
                    await writer.write(content)
                    relationship_distinct.update(distinct)

        logger.info(f"classes: {len(class_distinct)}, relationship: {len(relationship_distinct)}")

        # 如果有 i_context，更新图谱并记录 Mermaid 文件
        if self.i_context:
            r_filename = Path(filename).relative_to(self.context.git_repo.workdir)
            await self.graph_db.insert(
                subject=self.i_context, predicate="hasMermaidClassDiagramFile", object_=str(r_filename)
            )
            logger.info(f"{self.i_context} hasMermaidClassDiagramFile {filename}")
        return filename

    async def _create_mermaid_class(self, ns_class_name) -> str:
        """为特定类生成 Mermaid 类图，基于图谱数据库中的数据。

        参数：
            ns_class_name (str): 要为其创建 Mermaid 类图的命名空间前缀类名。

        返回：
            str: 一个 Mermaid 代码块，表示类图的 markdown 格式。
        """
        # 拆分类名的命名空间
        fields = split_namespace(ns_class_name)
        if len(fields) > 2:
            # 忽略子类
            return ""

        # 获取类的详细信息
        rows = await self.graph_db.select(subject=ns_class_name, predicate=GraphKeyword.HAS_DETAIL)
        if not rows:
            return ""

        # 使用 DotClassInfo 生成类视图
        dot_class_info = DotClassInfo.model_validate_json(rows[0].object_)
        class_view = UMLClassView.load_dot_class_info(dot_class_info)

        # 更新 UML 类视图
        await self.graph_db.insert(ns_class_name, GraphKeyword.HAS_CLASS_VIEW, class_view.model_dump_json())

        # 更新 UML 的组合关系
        for c in dot_class_info.compositions:
            await self.graph_db.insert(
                subject=ns_class_name,
                predicate=GraphKeyword.IS + COMPOSITION + GraphKeyword.OF,
                object_=concat_namespace("?", c),
            )

        # 更新 UML 的聚合关系
        for a in dot_class_info.aggregations:
            await self.graph_db.insert(
                subject=ns_class_name,
                predicate=GraphKeyword.IS + AGGREGATION + GraphKeyword.OF,
                object_=concat_namespace("?", a),
            )

        # 获取并返回 Mermaid 类图的代码
        content = class_view.get_mermaid(align=1)
        logger.debug(content)
        return content

    async def _create_mermaid_relationship(self, ns_class_name: str) -> Tuple[Optional[str], Optional[Set]]:
        """为特定类生成 Mermaid 关系图，基于图谱数据库中的数据。

        参数：
            ns_class_name (str): 要为其创建关系图的命名空间前缀类名。

        返回：
            Tuple[str, Set]: 包含关系图的字符串和去重的关系集合。
        """
        # 拆分类名的命名空间
        s_fields = split_namespace(ns_class_name)
        if len(s_fields) > 2:
            # 忽略子类
            return None, None

        # 定义关系类型和对应的符号
        predicates = {GraphKeyword.IS + v + GraphKeyword.OF: v for v in [GENERALIZATION, COMPOSITION, AGGREGATION]}
        mappings = {
            GENERALIZATION: " <|-- ",
            COMPOSITION: " *-- ",
            AGGREGATION: " o-- ",
        }

        content = ""
        distinct = set()

        # 遍历关系类型并生成对应的关系图
        for p, v in predicates.items():
            rows = await self.graph_db.select(subject=ns_class_name, predicate=p)
            for r in rows:
                o_fields = split_namespace(r.object_)
                if len(o_fields) > 2:
                    # 忽略子类
                    continue
                relationship = mappings.get(v, " .. ")
                link = f"{o_fields[1]}{relationship}{s_fields[1]}"
                distinct.add(link)
                content += f"\t{link}\n"

        return content, distinct

    @staticmethod
    def _diff_path(path_root: Path, package_root: Path) -> (str, str):
        """返回根路径与包名所表示路径信息之间的差异。

        参数：
            path_root (Path): 根路径。
            package_root (Path): 包根路径。

        返回：
            Tuple[str, str]: 返回差异表示符号（"+", "-", "="）以及差异部分的路径。
        """
        if len(str(path_root)) > len(str(package_root)):
            return "+", str(path_root.relative_to(package_root))
        if len(str(path_root)) < len(str(package_root)):
            return "-", str(package_root.relative_to(path_root))
        return "=", "."

    @staticmethod
    def _align_root(path: str, direction: str, diff_path: str) -> str:
        """将路径对齐到与 `diff_path` 相同的根路径。

        参数：
            path (str): 要对齐的路径。
            direction (str): 对齐方向 ('+', '-', '=')。
            diff_path (str): 表示差异部分的路径。

        返回：
            str: 对齐后的路径。
        """
        if direction == "=":
            return path
        if direction == "+":
            return diff_path + "/" + path
        else:
            return path[len(diff_path) + 1:]
