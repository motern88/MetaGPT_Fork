#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19
@Author  : mashenquan
@File    : graph_repository.py
@Desc    : Superclass for graph repository. This script defines a superclass for a graph repository, providing a
    foundation for specific implementations.

"""

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List

from pydantic import BaseModel

from metagpt.repo_parser import DotClassInfo, DotClassRelationship, RepoFileInfo
from metagpt.utils.common import concat_namespace, split_namespace


class GraphKeyword:
    """图数据库的基本词汇。

    该类定义了一组常用于图数据库上下文中的基本词汇。
    """

    IS = "is"  # 表示“是”
    OF = "Of"  # 表示“的”
    ON = "On"  # 表示“在”
    CLASS = "class"  # 表示“类”
    FUNCTION = "function"  # 表示“函数”
    HAS_FUNCTION = "has_function"  # 表示“拥有函数”
    SOURCE_CODE = "source_code"  # 表示“源代码”
    NULL = "<null>"  # 表示“空”或“无”
    GLOBAL_VARIABLE = "global_variable"  # 表示“全局变量”
    CLASS_METHOD = "class_method"  # 表示“类方法”
    CLASS_PROPERTY = "class_property"  # 表示“类属性”
    HAS_CLASS_METHOD = "has_class_method"  # 表示“拥有类方法”
    HAS_CLASS_PROPERTY = "has_class_property"  # 表示“拥有类属性”
    HAS_CLASS = "has_class"  # 表示“拥有类”
    HAS_DETAIL = "has_detail"  # 表示“拥有详细信息”
    HAS_PAGE_INFO = "has_page_info"  # 表示“拥有页面信息”
    HAS_CLASS_VIEW = "has_class_view"  # 表示“拥有类视图”
    HAS_SEQUENCE_VIEW = "has_sequence_view"  # 表示“拥有序列视图”
    HAS_SEQUENCE_VIEW_VER = "has_sequence_view_ver"  # 表示“拥有序列视图版本”
    HAS_CLASS_USE_CASE = "has_class_use_case"  # 表示“拥有类用例”
    IS_COMPOSITE_OF = "is_composite_of"  # 表示“是...的组合”
    IS_AGGREGATE_OF = "is_aggregate_of"  # 表示“是...的聚合”
    HAS_PARTICIPANT = "has_participant"  # 表示“拥有参与者”
    HAS_SUMMARY = "has_summary"  # 表示“拥有摘要”
    HAS_INSTALL = "has_install"  # 表示“拥有安装”
    HAS_CONFIG = "has_config"  # 表示“拥有配置”
    HAS_USAGE = "has_usage"  # 表示“拥有使用情况”


class SPO(BaseModel):
    """图数据库记录类型。

    该类表示图数据库中的一条记录，包含三个组成部分：
    - 主体 (Subject): 三元组中的主体。
    - 谓词 (Predicate): 描述主体和客体之间关系的谓词。
    - 客体 (Object): 三元组中的客体。

    属性：
        subject (str): 三元组中的主体。
        predicate (str): 描述主体和客体关系的谓词。
        object_ (str): 三元组中的客体。

    示例：
        spo_record = SPO(subject="Node1", predicate="connects_to", object_="Node2")
        # 表示一个三元组：Node1 连接到 Node2
    """

    subject: str  # 主体，表示三元组中的主体部分
    predicate: str  # 谓词，表示主体和客体之间的关系
    object_: str  # 客体，表示三元组中的客体部分


class GraphRepository(ABC):
    """图数据库仓库的抽象基类。

    该类定义了图数据库仓库的接口，提供了插入、选择、删除和保存图数据的方法。
    具体的实现类必须提供这些操作的功能。
    """

    def __init__(self, name: str, **kwargs):
        self._repo_name = name  # 仓库的名称
        self._kwargs = kwargs  # 其他可能的配置参数

    @abstractmethod
    async def insert(self, subject: str, predicate: str, object_: str):
        """向图数据库仓库插入一条新的三元组。

        参数：
            subject (str): 三元组的主体。
            predicate (str): 描述主体和客体关系的谓词。
            object_ (str): 三元组的客体。

        示例：
            await my_repository.insert(subject="Node1", predicate="connects_to", object_="Node2")
            # 插入一条三元组：Node1 连接到 Node2 到图数据库仓库中。
        """
        pass

    @abstractmethod
    async def select(self, subject: str = None, predicate: str = None, object_: str = None) -> List[SPO]:
        """根据指定的条件从图数据库仓库中检索三元组。

        参数：
            subject (str, optional): 用于过滤的三元组主体。
            predicate (str, optional): 用于过滤的描述关系的谓词。
            object_ (str, optional): 用于过滤的三元组客体。

        返回：
            List[SPO]: 代表选定三元组的 SPO 对象列表。

        示例：
            selected_triples = await my_repository.select(subject="Node1", predicate="connects_to")
            # 检索出主体为 Node1，谓词为 'connects_to' 的三元组。
        """
        pass

    @abstractmethod
    async def delete(self, subject: str = None, predicate: str = None, object_: str = None) -> int:
        """根据指定的条件从图数据库仓库中删除三元组。

        参数：
            subject (str, optional): 用于过滤的三元组主体。
            predicate (str, optional): 用于过滤的描述关系的谓词。
            object_ (str, optional): 用于过滤的三元组客体。

        返回：
            int: 从仓库中删除的三元组数量。

        示例：
            deleted_count = await my_repository.delete(subject="Node1", predicate="connects_to")
            # 删除主体为 Node1，谓词为 'connects_to' 的三元组。
        """
        pass

    @abstractmethod
    async def save(self):
        """保存对图数据库仓库所做的所有更改。

        示例：
            await my_repository.save()
            # 持久化所有对图数据库仓库所做的更改。
        """
        pass

    @property
    def name(self) -> str:
        """获取图数据库仓库的名称。"""
        return self._repo_name

    @staticmethod
    async def update_graph_db_with_file_info(graph_db: "GraphRepository", file_info: RepoFileInfo):
        """将 RepoFileInfo 中的信息插入到指定的图数据库仓库中。

        此方法将给定的 RepoFileInfo 对象中的信息更新到提供的图数据库仓库中。
        它插入与文件类型、类、类方法、函数、全局变量和页面信息等相关的三元组。

        三元组模式：
        - (?, is, [文件类型])
        - (?, has class, ?)
        - (?, is, [类])
        - (?, has class method, ?)
        - (?, has function, ?)
        - (?, is, 全局变量)
        - (?, has page info, ?)

        参数：
            graph_db (GraphRepository): 要更新的图数据库对象。
            file_info (RepoFileInfo): 包含插入信息的 RepoFileInfo 对象。

        示例：
            await update_graph_db_with_file_info(my_graph_repo, my_file_info)
            # 使用 'my_file_info' 中的信息更新 'my_graph_repo'。
        """
        # 插入文件类型信息
        await graph_db.insert(subject=file_info.file, predicate=GraphKeyword.IS, object_=GraphKeyword.SOURCE_CODE)
        file_types = {".py": "python", ".js": "javascript"}
        file_type = file_types.get(Path(file_info.file).suffix, GraphKeyword.NULL)
        await graph_db.insert(subject=file_info.file, predicate=GraphKeyword.IS, object_=file_type)

        # 插入类信息
        for c in file_info.classes:
            class_name = c.get("name", "")
            # 文件 -> 类
            await graph_db.insert(
                subject=file_info.file,
                predicate=GraphKeyword.HAS_CLASS,
                object_=concat_namespace(file_info.file, class_name),
            )
            # 类详细信息
            await graph_db.insert(
                subject=concat_namespace(file_info.file, class_name),
                predicate=GraphKeyword.IS,
                object_=GraphKeyword.CLASS,
            )
            methods = c.get("methods", [])
            for fn in methods:
                # 类 -> 类方法
                await graph_db.insert(
                    subject=concat_namespace(file_info.file, class_name),
                    predicate=GraphKeyword.HAS_CLASS_METHOD,
                    object_=concat_namespace(file_info.file, class_name, fn),
                )
                # 类方法详细信息
                await graph_db.insert(
                    subject=concat_namespace(file_info.file, class_name, fn),
                    predicate=GraphKeyword.IS,
                    object_=GraphKeyword.CLASS_METHOD,
                )
        # 插入函数信息
        for f in file_info.functions:
            # 文件 -> 函数
            await graph_db.insert(
                subject=file_info.file, predicate=GraphKeyword.HAS_FUNCTION, object_=concat_namespace(file_info.file, f)
            )
            # 函数详细信息
            await graph_db.insert(
                subject=concat_namespace(file_info.file, f), predicate=GraphKeyword.IS, object_=GraphKeyword.FUNCTION
            )

        # 插入全局变量信息
        for g in file_info.globals:
            await graph_db.insert(
                subject=concat_namespace(file_info.file, g),
                predicate=GraphKeyword.IS,
                object_=GraphKeyword.GLOBAL_VARIABLE,
            )

        # 插入页面信息
        for code_block in file_info.page_info:
            if code_block.tokens:
                await graph_db.insert(
                    subject=concat_namespace(file_info.file, *code_block.tokens),
                    predicate=GraphKeyword.HAS_PAGE_INFO,
                    object_=code_block.model_dump_json(),
                )
            for k, v in code_block.properties.items():
                await graph_db.insert(
                    subject=concat_namespace(file_info.file, k, v),
                    predicate=GraphKeyword.HAS_PAGE_INFO,
                    object_=code_block.model_dump_json(),
                )

    @staticmethod
    async def update_graph_db_with_class_views(graph_db: "GraphRepository", class_views: List[DotClassInfo]):
        """将 dot 格式的类信息插入到指定的图数据库仓库中。

        此方法将给定的 DotClassInfo 对象列表中的类信息更新到图数据库仓库中。
        它插入与类视图的各种方面相关的三元组，包括源代码、文件类型、类、类属性、类详细信息、方法、组合和聚合。

        三元组模式：
        - (?, is, source code)
        - (?, is, file type)
        - (?, has class, ?)
        - (?, is, class)
        - (?, has class property, ?)
        - (?, is, class property)
        - (?, has detail, ?)
        - (?, has method, ?)
        - (?, is composite of, ?)
        - (?, is aggregate of, ?)

        参数：
            graph_db (GraphRepository): 要更新的图数据库对象。
            class_views (List[DotClassInfo]): 包含要插入的类信息的 DotClassInfo 对象列表。

        示例：
            await update_graph_db_with_class_views(my_graph_repo, [class_info1, class_info2])
            # 使用提供的 DotClassInfo 对象列表中的类信息更新 'my_graph_repo'。
        """
        for c in class_views:
            filename, _ = c.package.split(":", 1)
            # 插入源代码信息
            await graph_db.insert(subject=filename, predicate=GraphKeyword.IS, object_=GraphKeyword.SOURCE_CODE)
            file_types = {".py": "python", ".js": "javascript"}
            file_type = file_types.get(Path(filename).suffix, GraphKeyword.NULL)
            await graph_db.insert(subject=filename, predicate=GraphKeyword.IS, object_=file_type)
            # 文件 -> 类
            await graph_db.insert(subject=filename, predicate=GraphKeyword.HAS_CLASS, object_=c.package)
            # 类详细信息
            await graph_db.insert(
                subject=c.package,
                predicate=GraphKeyword.IS,
                object_=GraphKeyword.CLASS,
            )
            await graph_db.insert(subject=c.package, predicate=GraphKeyword.HAS_DETAIL, object_=c.model_dump_json())

            # 插入类属性信息
            for vn, vt in c.attributes.items():
                # 类 -> 属性
                await graph_db.insert(
                    subject=c.package,
                    predicate=GraphKeyword.HAS_CLASS_PROPERTY,
                    object_=concat_namespace(c.package, vn),
                )
                # 属性详细信息
                await graph_db.insert(
                    subject=concat_namespace(c.package, vn),
                    predicate=GraphKeyword.IS,
                    object_=GraphKeyword.CLASS_PROPERTY,
                )
                await graph_db.insert(
                    subject=concat_namespace(c.package, vn),
                    predicate=GraphKeyword.HAS_DETAIL,
                    object_=vt.model_dump_json(),
                )

            # 插入类方法信息
            for fn, ft in c.methods.items():
                # 类 -> 方法
                await graph_db.insert(
                    subject=c.package,
                    predicate=GraphKeyword.HAS_CLASS_METHOD,
                    object_=concat_namespace(c.package, fn),
                )
                # 方法详细信息
                await graph_db.insert(
                    subject=concat_namespace(c.package, fn),
                    predicate=GraphKeyword.IS,
                    object_=GraphKeyword.CLASS_METHOD,
                )
                await graph_db.insert(
                    subject=concat_namespace(c.package, fn),
                    predicate=GraphKeyword.HAS_DETAIL,
                    object_=ft.model_dump_json(),
                )

            # 插入组合信息
            for i in c.compositions:
                await graph_db.insert(
                    subject=c.package, predicate=GraphKeyword.IS_COMPOSITE_OF, object_=concat_namespace("?", i)
                )

            # 插入聚合信息
            for i in c.aggregations:
                await graph_db.insert(
                    subject=c.package, predicate=GraphKeyword.IS_AGGREGATE_OF, object_=concat_namespace("?", i)
                )

    @staticmethod
    async def update_graph_db_with_class_relationship_views(
            graph_db: "GraphRepository", relationship_views: List[DotClassRelationship]
    ):
        """插入类之间的关系和标签到指定的图数据库中。

        这个函数将通过给定的 DotClassRelationship 对象列表更新图数据库中的类关系信息。函数会插入表示类之间关系和标签的三元组。

        三元组模式：
        - (?, 是关系的, ?)
        - (?, 是关系上的, ?)

        参数:
            graph_db (GraphRepository): 需要更新的图数据库对象。
            relationship_views (List[DotClassRelationship]): 包含类关系信息的 DotClassRelationship 对象列表。

        示例:
            await update_graph_db_with_class_relationship_views(my_graph_repo, [relationship1, relationship2])
            # 使用提供的 DotClassRelationship 对象列表更新 'my_graph_repo' 的类关系信息。
        """
        for r in relationship_views:
            # 插入关系三元组（源 -> 关系的 -> 目标）
            await graph_db.insert(
                subject=r.src, predicate=GraphKeyword.IS + r.relationship + GraphKeyword.OF, object_=r.dest
            )
            if not r.label:
                continue
            # 插入带标签的关系三元组（源 -> 关系上的 -> 目标标签）
            await graph_db.insert(
                subject=r.src,
                predicate=GraphKeyword.IS + r.relationship + GraphKeyword.ON,
                object_=concat_namespace(r.dest, r.label),
            )

    @staticmethod
    async def rebuild_composition_relationship(graph_db: "GraphRepository"):
        """向图数据库中的关系 SPO（主体-谓词-宾语）对象追加命名空间前缀信息。

        该函数通过向现有的关系 SPO 对象追加命名空间前缀信息来更新图数据库。

        参数:
            graph_db (GraphRepository): 需要更新的图数据库对象。
        """
        # 查询所有类的三元组（以获取类的命名空间）
        classes = await graph_db.select(predicate=GraphKeyword.IS, object_=GraphKeyword.CLASS)
        mapping = defaultdict(list)
        for c in classes:
            name = split_namespace(c.subject)[-1]
            mapping[name].append(c.subject)

        # 查询所有“是组合关系”的三元组
        rows = await graph_db.select(predicate=GraphKeyword.IS_COMPOSITE_OF)
        for r in rows:
            ns, class_ = split_namespace(r.object_)
            # 如果对象的命名空间不为“？”
            if ns != "?":
                continue
            val = mapping[class_]
            # 如果存在多个命名空间映射，跳过
            if len(val) != 1:
                continue
            ns_name = val[0]
            # 删除旧的关系三元组并插入带有命名空间的三元组
            await graph_db.delete(subject=r.subject, predicate=r.predicate, object_=r.object_)
            await graph_db.insert(subject=r.subject, predicate=r.predicate, object_=ns_name)