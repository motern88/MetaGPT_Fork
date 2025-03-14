#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/29 14:46
@Author  : alexanderwu
@File    : chromadb_store.py
"""
import chromadb


class ChromaStore:
    """Chroma存储类

    该类提供了与Chroma数据库交互的功能，包括搜索、添加、删除和持久化数据。
    注意：如果继承自 BaseStore 或从 metagpt 导入其他模块，会导致 Python 异常，原因尚不明。
    """

    def __init__(self, name: str, get_or_create: bool = False):
        """初始化 Chroma 存储

        该方法创建一个新的 Chroma 客户端并获取或创建一个集合。

        Args:
            name (str): 集合的名称
            get_or_create (bool, optional): 是否创建新集合，默认为 False。如果集合已存在，则返回现有集合。
        """
        client = chromadb.Client()  # 创建 Chroma 客户端
        collection = client.create_collection(name, get_or_create=get_or_create)  # 获取或创建集合
        self.client = client
        self.collection = collection

    def search(self, query, n_results=2, metadata_filter=None, document_filter=None):
        """执行搜索操作

        该方法根据查询文本返回最相似的文档，支持元数据过滤和文档过滤。

        Args:
            query (str): 搜索查询文本
            n_results (int, optional): 返回的结果数量，默认为 2
            metadata_filter (dict, optional): 元数据过滤条件，默认为 None
            document_filter (dict, optional): 文档过滤条件，默认为 None

        Returns:
            dict: 搜索结果
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=metadata_filter,  # 可选的元数据过滤条件
            where_document=document_filter,  # 可选的文档过滤条件
        )
        return results

    def persist(self):
        """Chroma建议使用服务器模式，而非本地持久化"""
        raise NotImplementedError  # 该方法未实现

    def write(self, documents, metadatas, ids):
        """写入文档数据

        该方法用于更新或写入多个文档。它接受文档、元数据和文档ID的列表。

        Args:
            documents (list): 文档内容列表
            metadatas (list): 文档的元数据列表
            ids (list): 文档的ID列表

        Returns:
            AddResult: 添加结果
        """
        return self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def add(self, document, metadata, _id):
        """添加单个文档

        该方法用于向集合中添加单个文档。它接受一个文档、元数据和文档ID。

        Args:
            document (str): 文档内容
            metadata (dict): 文档的元数据
            _id (str): 文档ID

        Returns:
            AddResult: 添加结果
        """
        return self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[_id],
        )

    def delete(self, _id):
        """删除文档

        该方法根据提供的ID删除文档。

        Args:
            _id (str): 要删除的文档ID

        Returns:
            DeleteResult: 删除结果
        """
        return self.collection.delete([_id])
