"""RAG Index Factory."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore

from metagpt.rag.factories.base import ConfigBasedFactory
from metagpt.rag.schema import (
    BaseIndexConfig,
    BM25IndexConfig,
    ChromaIndexConfig,
    ElasticsearchIndexConfig,
    ElasticsearchKeywordIndexConfig,
    FAISSIndexConfig,
    MilvusIndexConfig,
)


class RAGIndexFactory(ConfigBasedFactory):
    """RAG 索引工厂类，根据不同的配置创建对应的索引实例。"""

    def __init__(self):
        creators = {
            FAISSIndexConfig: self._create_faiss,  # FAISS 索引配置
            ChromaIndexConfig: self._create_chroma,  # Chroma 索引配置
            BM25IndexConfig: self._create_bm25,  # BM25 索引配置
            ElasticsearchIndexConfig: self._create_es,  # Elasticsearch 索引配置
            ElasticsearchKeywordIndexConfig: self._create_es,  # Elasticsearch 关键词索引配置
            MilvusIndexConfig: self._create_milvus,  # Milvus 索引配置
        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig, **kwargs) -> BaseIndex:
        """根据索引配置返回对应的索引实例。

        参数 `config` 是索引配置，`kwargs` 是额外的参数。
        """
        return super().get_instance(config, **kwargs)

    def _create_faiss(self, config: FAISSIndexConfig, **kwargs) -> VectorStoreIndex:
        """创建 FAISS 索引实例"""
        vector_store = FaissVectorStore.from_persist_dir(str(config.persist_path))
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=config.persist_path)

        return self._index_from_storage(storage_context=storage_context, config=config, **kwargs)

    def _create_bm25(self, config: BM25IndexConfig, **kwargs) -> VectorStoreIndex:
        """创建 BM25 索引实例"""
        storage_context = StorageContext.from_defaults(persist_dir=config.persist_path)

        return self._index_from_storage(storage_context=storage_context, config=config, **kwargs)

    def _create_milvus(self, config: MilvusIndexConfig, **kwargs) -> VectorStoreIndex:
        """创建 Milvus 索引实例"""
        vector_store = MilvusVectorStore(collection_name=config.collection_name, uri=config.uri, token=config.token)

        return self._index_from_vector_store(vector_store=vector_store, config=config, **kwargs)

    def _create_chroma(self, config: ChromaIndexConfig, **kwargs) -> VectorStoreIndex:
        """创建 Chroma 索引实例"""
        db = chromadb.PersistentClient(str(config.persist_path))
        chroma_collection = db.get_or_create_collection(config.collection_name, metadata=config.metadata)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        return self._index_from_vector_store(vector_store=vector_store, config=config, **kwargs)

    def _create_es(self, config: ElasticsearchIndexConfig, **kwargs) -> VectorStoreIndex:
        """创建 Elasticsearch 索引实例"""
        vector_store = ElasticsearchStore(**config.store_config.model_dump())

        return self._index_from_vector_store(vector_store=vector_store, config=config, **kwargs)

    def _index_from_storage(
        self, storage_context: StorageContext, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        """从存储上下文创建索引实例"""
        embed_model = self._extract_embed_model(config, **kwargs)

        return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

    def _index_from_vector_store(
        self, vector_store: BasePydanticVectorStore, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        """从向量存储创建索引实例"""
        embed_model = self._extract_embed_model(config, **kwargs)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    def _extract_embed_model(self, config, **kwargs) -> BaseEmbedding:
        """从配置或传入参数中提取嵌入模型"""
        return self._val_from_config_or_kwargs("embed_model", config, **kwargs)


get_index = RAGIndexFactory().get_index
