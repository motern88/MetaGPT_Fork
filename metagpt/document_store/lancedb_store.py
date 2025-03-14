#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/9 15:42
@Author  : unkn-wn (Leon Yee)
@File    : lancedb_store.py
"""
import os
import shutil

import lancedb


class LanceStore:
    """LanceStore 类，用于操作 LanceDB 数据库的向量存储和查询。

    该类提供了对数据库的连接、数据插入、查询、删除等基本操作，支持基于向量的相似度搜索。
    """

    def __init__(self, name):
        """初始化 LanceStore 对象

        Args:
            name (str): 数据库表名，用于标识存储的表
        """
        db = lancedb.connect("./data/lancedb")  # 连接到本地 LanceDB 数据库
        self.db = db
        self.name = name
        self.table = None  # 表格尚未创建

    def search(self, query, n_results=2, metric="L2", nprobes=20, **kwargs):
        """执行相似度搜索

        Args:
            query (list): 查询向量（嵌入表示）
            n_results (int, optional): 返回的结果数量，默认为 2
            metric (str, optional): 距离度量方法，默认为 L2 距离
            nprobes (int, optional): nprobes 值，值越大，召回率越高，默认为 20
            kwargs: 可选的过滤条件，如 `select`、`where` 等

        Returns:
            pd.DataFrame: 返回包含搜索结果的 DataFrame
        """
        # 假设 query 是一个向量嵌入
        # kwargs 可以用于可选的过滤条件
        # .select - 只搜索指定的列
        # .where - 使用 SQL 语法过滤元数据（例如 where("price > 100")）
        # .metric - 指定使用的距离度量
        # .nprobes - 通过提高该值来增加召回率，但会增加延迟

        if self.table is None:
            raise Exception("Table not created yet, please add data first.")  # 如果表格不存在，抛出异常

        # 执行查询并返回结果
        results = (
            self.table.search(query)
            .limit(n_results)  # 限制返回结果的数量
            .select(kwargs.get("select"))  # 选择特定的列
            .where(kwargs.get("where"))  # 添加过滤条件
            .metric(metric)  # 设置距离度量
            .nprobes(nprobes)  # 设置 nprobes 值
            .to_df()  # 转换为 DataFrame 格式
        )
        return results

    def persist(self):
        """持久化存储

        该方法暂时未实现，用于将数据保存到持久化存储中。
        """
        raise NotImplementedError  # 尚未实现

    def write(self, data, metadatas, ids):
        """写入数据

        该方法用于将数据插入到 LanceDB 表中。它支持批量插入，并将元数据展开为 DataFrame 格式。

        Args:
            data (list): 向量数据（嵌入表示）
            metadatas (list[dict]): 元数据列表
            ids (list): 文档的唯一标识符
        """
        documents = []
        for i in range(len(data)):
            row = {"vector": data[i], "id": ids[i]}  # 创建每行数据
            row.update(metadatas[i])  # 将元数据添加到行数据中
            documents.append(row)

        if self.table is not None:
            self.table.add(documents)  # 如果表格已创建，添加数据
        else:
            self.table = self.db.create_table(self.name, documents)  # 否则创建新表并添加数据

    def add(self, data, metadata, _id):
        """添加单个文档

        该方法用于添加单个文档的向量数据和元数据。

        Args:
            data (list): 向量数据（嵌入表示）
            metadata (dict): 文档的元数据
            _id (str): 文档的唯一标识符
        """
        row = {"vector": data, "id": _id}  # 创建数据行
        row.update(metadata)  # 将元数据添加到行数据中

        if self.table is not None:
            self.table.add([row])  # 如果表格已创建，添加数据
        else:
            self.table = self.db.create_table(self.name, [row])  # 否则创建新表并添加数据

    def delete(self, _id):
        """删除文档

        该方法根据文档的 id 删除相应的行。

        Args:
            _id (str or int): 要删除的文档的唯一标识符
        """
        if self.table is None:
            raise Exception("Table not created yet, please add data first")  # 如果表格不存在，抛出异常

        if isinstance(_id, str):
            return self.table.delete(f"id = '{_id}'")  # 如果 id 是字符串，删除对应的记录
        else:
            return self.table.delete(f"id = {_id}")  # 如果 id 是整数，删除对应的记录

    def drop(self, name):
        """删除表格

        该方法删除指定名称的表格（如果存在）。

        Args:
            name (str): 表格的名称
        """
        path = os.path.join(self.db.uri, name + ".lance")  # 表格文件路径
        if os.path.exists(path):
            shutil.rmtree(path)  # 删除表格文件夹
