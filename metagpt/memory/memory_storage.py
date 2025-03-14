#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : the implement of memory storage
"""
import shutil
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding

from metagpt.const import DATA_PATH, MEM_TTL
from metagpt.logs import logger
from metagpt.rag.engines.simple import SimpleEngine
from metagpt.rag.schema import FAISSIndexConfig, FAISSRetrieverConfig
from metagpt.schema import Message
from metagpt.utils.embedding import get_embedding


class MemoryStorage(object):
    """
    使用 Faiss 作为 ANN 搜索引擎的记忆存储
    """

    def __init__(self, mem_ttl: int = MEM_TTL, embedding: BaseEmbedding = None):
        self.role_id: str = None  # 角色 ID
        self.role_mem_path: str = None  # 角色记忆存储路径
        self.mem_ttl: int = mem_ttl  # 内存 TTL（生命周期），稍后使用
        self.threshold: float = 0.1  # 阈值，用于过滤相似记忆，经验值。TODO: 待完善
        self._initialized: bool = False  # 是否已初始化
        self.embedding = embedding or get_embedding()  # 获取嵌入模型

        self.faiss_engine = None  # Faiss 引擎实例

    @property
    def is_initialized(self) -> bool:
        """返回存储是否已初始化"""
        return self._initialized

    def recover_memory(self, role_id: str) -> list[Message]:
        """
        恢复角色记忆
        根据角色 ID 恢复记忆并加载 Faiss 引擎
        """
        self.role_id = role_id  # 设置角色 ID
        self.role_mem_path = Path(DATA_PATH / f"role_mem/{self.role_id}/")  # 设置角色记忆路径
        self.role_mem_path.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
        self.cache_dir = self.role_mem_path  # 缓存目录

        # 如果存在向量存储文件，则从文件加载 Faiss 引擎
        if self.role_mem_path.joinpath("default__vector_store.json").exists():
            self.faiss_engine = SimpleEngine.from_index(
                index_config=FAISSIndexConfig(persist_path=self.cache_dir),
                retriever_configs=[FAISSRetrieverConfig()],
                embed_model=self.embedding,
            )
        else:
            # 如果没有向量存储文件，则初始化一个空的 Faiss 引擎
            self.faiss_engine = SimpleEngine.from_objs(
                objs=[], retriever_configs=[FAISSRetrieverConfig()], embed_model=self.embedding
            )
        self._initialized = True  # 标记为已初始化

    def add(self, message: Message) -> bool:
        """将消息添加到记忆存储中"""
        self.faiss_engine.add_objs([message])  # 使用 Faiss 引擎添加消息
        logger.info(f"角色 {self.role_id} 的 memory_storage 添加了一条消息")

    async def search_similar(self, message: Message, k=4) -> list[Message]:
        """
        搜索相似的消息
        根据给定的消息和相似度阈值，查找相似的消息
        """
        filtered_resp = []  # 存储过滤后的响应
        resp = await self.faiss_engine.aretrieve(message.content)  # 从 Faiss 引擎检索相似消息
        for item in resp:
            if item.score < self.threshold:
                filtered_resp.append(item.metadata.get("obj"))  # 过滤掉低于阈值的相似消息
        return filtered_resp

    def clean(self):
        """清除缓存目录并重置初始化状态"""
        shutil.rmtree(self.cache_dir, ignore_errors=True)  # 删除缓存目录
        self._initialized = False  # 重置初始化标志

    def persist(self):
        """将当前 Faiss 引擎的状态持久化到存储中"""
        if self.faiss_engine:
            self.faiss_engine.retriever._index.storage_context.persist(self.cache_dir)  # 持久化存储