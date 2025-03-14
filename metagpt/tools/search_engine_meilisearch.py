#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/22 21:33
@Author  : alexanderwu
@File    : search_engine_meilisearch.py
"""

from typing import List

import meilisearch
from meilisearch.index import Index

from metagpt.utils.exceptions import handle_exception


class DataSource:
    def __init__(self, name: str, url: str):
        """初始化数据源

        参数：
            name: 数据源的名称。
            url: 数据源的URL地址。
        """
        self.name = name  # 数据源名称
        self.url = url  # 数据源URL


class MeilisearchEngine:
    def __init__(self, url, token):
        """初始化Meilisearch引擎

        参数：
            url: Meilisearch实例的URL。
            token: 用于访问Meilisearch的API令牌。
        """
        self.client = meilisearch.Client(url, token)  # 创建Meilisearch客户端
        self._index: Index = None  # 初始化索引为None

    def set_index(self, index):
        """设置索引

        参数：
            index: 要设置的Meilisearch索引。
        """
        self._index = index  # 设置索引

    def add_documents(self, data_source: DataSource, documents: List[dict]):
        """将文档添加到Meilisearch索引中

        参数：
            data_source: 数据源对象，包含数据源的名称和URL。
            documents: 需要添加的文档列表，每个文档是一个字典。
        """
        # 使用数据源的名称创建索引名称
        index_name = f"{data_source.name}_index"
        # 如果索引不存在，创建新的索引
        if index_name not in self.client.get_indexes():
            self.client.create_index(uid=index_name, options={"primaryKey": "id"})
        # 获取或创建索引
        index = self.client.get_index(index_name)
        # 添加文档到索引中
        index.add_documents(documents)
        # 设置当前索引
        self.set_index(index)

    @handle_exception(exception_type=Exception, default_return=[])
    def search(self, query):
        """执行搜索操作

        参数：
            query: 搜索查询字符串。

        返回：
            返回搜索结果中的命中项列表。
        """
        search_results = self._index.search(query)  # 在当前索引中执行搜索
        return search_results["hits"]  # 返回搜索结果中的命中项
