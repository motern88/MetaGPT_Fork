#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/25 10:20
@Author  : alexanderwu
@File    : faiss_store.py
"""
import asyncio
from pathlib import Path
from typing import Any, Optional

import faiss
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import Document, QueryBundle, TextNode
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

from metagpt.document import IndexableDocument
from metagpt.document_store.base_store import LocalStore
from metagpt.logs import logger
from metagpt.utils.embedding import get_embedding


class FaissStore(LocalStore):
    """Faiss 存储类

    该类实现了使用 Faiss 进行向量存储和检索的功能，并继承自 LocalStore 类。它提供了文档的写入、检索、删除和持久化功能。
    """

    def __init__(
        self, raw_data: Path, cache_dir=None, meta_col="source", content_col="output", embedding: BaseEmbedding = None
    ):
        """初始化 Faiss 存储

        Args:
            raw_data (Path): 原始数据路径
            cache_dir (Path, optional): 缓存目录，默认为 None
            meta_col (str, optional): 元数据列名，默认为 "source"
            content_col (str, optional): 内容列名，默认为 "output"
            embedding (BaseEmbedding, optional): 嵌入模型，默认为 None
        """
        self.meta_col = meta_col
        self.content_col = content_col
        self.embedding = embedding or get_embedding()  # 如果没有提供嵌入模型，则使用默认嵌入模型
        self.store: VectorStoreIndex  # 存储向量索引
        super().__init__(raw_data, cache_dir)  # 调用父类初始化方法

    def _load(self) -> Optional["VectorStoreIndex"]:
        """加载索引

        加载 Faiss 向量存储和索引。

        Returns:
            VectorStoreIndex or None: 如果加载成功，返回 VectorStoreIndex；否则返回 None。
        """
        index_file, store_file = self._get_index_and_store_fname()

        if not (index_file.exists() and store_file.exists()):
            logger.info("Missing at least one of index_file/store_file, load failed and return None")
            return None
        vector_store = FaissVectorStore.from_persist_dir(persist_dir=self.cache_dir)
        storage_context = StorageContext.from_defaults(persist_dir=self.cache_dir, vector_store=vector_store)
        index = load_index_from_storage(storage_context, embed_model=self.embedding)

        return index

    def _write(self, docs: list[str], metadatas: list[dict[str, Any]]) -> VectorStoreIndex:
        """写入文档数据

        该方法将文档和元数据写入 Faiss 向量存储。

        Args:
            docs (list[str]): 文档内容列表
            metadatas (list[dict]): 文档元数据列表

        Returns:
            VectorStoreIndex: 返回向量存储的索引
        """
        assert len(docs) == len(metadatas)
        documents = [Document(text=doc, metadata=metadatas[idx]) for idx, doc in enumerate(docs)]

        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))  # 创建 Faiss 向量存储
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents, storage_context=storage_context, embed_model=self.embedding
        )

        return index

    def persist(self):
        """持久化存储

        将当前的存储上下文持久化到缓存目录。
        """
        self.store.storage_context.persist(self.cache_dir)

    def search(self, query: str, expand_cols=False, sep="\n", *args, k=5, **kwargs):
        """执行搜索操作

        该方法根据查询文本返回最相似的文档，并支持元数据过滤和文档过滤。

        Args:
            query (str): 搜索查询文本
            expand_cols (bool, optional): 是否展开列，默认为 False
            sep (str, optional): 用于连接结果的分隔符，默认为换行符
            k (int, optional): 返回的结果数量，默认为 5

        Returns:
            str: 搜索结果
        """
        retriever = self.store.as_retriever(similarity_top_k=k)  # 获取检索器
        rsp = retriever.retrieve(QueryBundle(query_str=query, embedding=self.embedding.get_text_embedding(query)))

        logger.debug(rsp)
        if expand_cols:
            return str(sep.join([f"{x.node.text}: {x.node.metadata}" for x in rsp]))  # 展开列并返回
        else:
            return str(sep.join([f"{x.node.text}" for x in rsp]))  # 仅返回文本

    async def asearch(self, *args, **kwargs):
        """异步搜索

        该方法异步执行搜索操作。

        Returns:
            str: 搜索结果
        """
        return await asyncio.to_thread(self.search, *args, **kwargs)

    def write(self):
        """初始化索引和库

        根据用户提供的文档（JSON、XLSX等）文件初始化索引和文档库。

        Returns:
            VectorStoreIndex: 返回创建的向量存储索引
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError
        doc = IndexableDocument.from_path(self.raw_data_path, self.content_col, self.meta_col)
        docs, metadatas = doc.get_docs_and_metadatas()

        self.store = self._write(docs, metadatas)
        self.persist()
        return self.store

    def add(self, texts: list[str], *args, **kwargs) -> list[str]:
        """添加文档

        该方法用于添加新文档，当前添加后不会更新存储。

        Args:
            texts (list[str]): 要添加的文本列表

        Returns:
            list[str]: 返回空列表（当前未实现更新）
        """
        texts_embeds = self.embedding.get_text_embedding_batch(texts)  # 获取文本的嵌入表示
        nodes = [TextNode(text=texts[idx], embedding=embed) for idx, embed in enumerate(texts_embeds)]
        self.store.insert_nodes(nodes)  # 向存储中插入新节点

        return []

    def delete(self, *args, **kwargs):
        """删除文档

        目前，Faiss 不提供删除接口。

        Raises:
            NotImplementedError: Faiss 不支持删除操作
        """
        raise NotImplementedError  # 暂不支持删除操作
