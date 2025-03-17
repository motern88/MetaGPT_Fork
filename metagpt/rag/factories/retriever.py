"""RAG Retriever Factory."""


from functools import wraps

import chromadb
import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore

from metagpt.rag.factories.base import ConfigBasedFactory
from metagpt.rag.retrievers.base import RAGRetriever
from metagpt.rag.retrievers.bm25_retriever import DynamicBM25Retriever
from metagpt.rag.retrievers.chroma_retriever import ChromaRetriever
from metagpt.rag.retrievers.es_retriever import ElasticsearchRetriever
from metagpt.rag.retrievers.faiss_retriever import FAISSRetriever
from metagpt.rag.retrievers.hybrid_retriever import SimpleHybridRetriever
from metagpt.rag.retrievers.milvus_retriever import MilvusRetriever
from metagpt.rag.schema import (
    BaseRetrieverConfig,
    BM25RetrieverConfig,
    ChromaRetrieverConfig,
    ElasticsearchKeywordRetrieverConfig,
    ElasticsearchRetrieverConfig,
    FAISSRetrieverConfig,
    MilvusRetrieverConfig,
)


def get_or_build_index(build_index_func):
    """装饰器，用于获取或构建索引。

    使用 `_extract_index` 方法获取索引，如果未找到，则使用 build_index_func 构建索引。
    """

    @wraps(build_index_func)
    def wrapper(self, config, **kwargs):
        index = self._extract_index(config, **kwargs)
        if index is not None:
            return index
        return build_index_func(self, config, **kwargs)

    return wrapper


class RetrieverFactory(ConfigBasedFactory):
    """动态实例化实现的创建者工厂类。"""

    def __init__(self):
        creators = {
            FAISSRetrieverConfig: self._create_faiss_retriever,
            BM25RetrieverConfig: self._create_bm25_retriever,
            ChromaRetrieverConfig: self._create_chroma_retriever,
            ElasticsearchRetrieverConfig: self._create_es_retriever,
            ElasticsearchKeywordRetrieverConfig: self._create_es_retriever,
            MilvusRetrieverConfig: self._create_milvus_retriever,
        }
        super().__init__(creators)

    def get_retriever(self, configs: list[BaseRetrieverConfig] = None, **kwargs) -> RAGRetriever:
        """根据提供的配置创建并返回检索器实例。

        如果有多个检索器，则使用 SimpleHybridRetriever。
        """
        if not configs:
            return self._create_default(**kwargs)

        retrievers = super().get_instances(configs, **kwargs)

        return SimpleHybridRetriever(*retrievers) if len(retrievers) > 1 else retrievers[0]

    def _create_default(self, **kwargs) -> RAGRetriever:
        """创建默认的检索器实例"""
        index = self._extract_index(None, **kwargs) or self._build_default_index(**kwargs)

        return index.as_retriever()

    def _create_milvus_retriever(self, config: MilvusRetrieverConfig, **kwargs) -> MilvusRetriever:
        """创建 Milvus 检索器实例"""
        config.index = self._build_milvus_index(config, **kwargs)

        return MilvusRetriever(**config.model_dump())

    def _create_faiss_retriever(self, config: FAISSRetrieverConfig, **kwargs) -> FAISSRetriever:
        """创建 FAISS 检索器实例"""
        config.index = self._build_faiss_index(config, **kwargs)

        return FAISSRetriever(**config.model_dump())

    def _create_bm25_retriever(self, config: BM25RetrieverConfig, **kwargs) -> DynamicBM25Retriever:
        """创建 BM25 检索器实例"""
        index = self._extract_index(config, **kwargs)
        nodes = list(index.docstore.docs.values()) if index else self._extract_nodes(config, **kwargs)

        if index and not config.index:
            config.index = index

        if not config.index and config.create_index:
            config.index = VectorStoreIndex(nodes, embed_model=MockEmbedding(embed_dim=1))

        return DynamicBM25Retriever(nodes=nodes, **config.model_dump())

    def _create_chroma_retriever(self, config: ChromaRetrieverConfig, **kwargs) -> ChromaRetriever:
        """创建 Chroma 检索器实例"""
        config.index = self._build_chroma_index(config, **kwargs)

        return ChromaRetriever(**config.model_dump())

    def _create_es_retriever(self, config: ElasticsearchRetrieverConfig, **kwargs) -> ElasticsearchRetriever:
        """创建 Elasticsearch 检索器实例"""
        config.index = self._build_es_index(config, **kwargs)

        return ElasticsearchRetriever(**config.model_dump())

    def _extract_index(self, config: BaseRetrieverConfig = None, **kwargs) -> VectorStoreIndex:
        """从配置或关键字参数中提取索引"""
        return self._val_from_config_or_kwargs("index", config, **kwargs)

    def _extract_nodes(self, config: BaseRetrieverConfig = None, **kwargs) -> list[BaseNode]:
        """从配置或关键字参数中提取节点"""
        return self._val_from_config_or_kwargs("nodes", config, **kwargs)

    def _extract_embed_model(self, config: BaseRetrieverConfig = None, **kwargs) -> BaseEmbedding:
        """从配置或关键字参数中提取嵌入模型"""
        return self._val_from_config_or_kwargs("embed_model", config, **kwargs)

    def _build_default_index(self, **kwargs) -> VectorStoreIndex:
        """构建默认的索引"""
        index = VectorStoreIndex(
            nodes=self._extract_nodes(**kwargs),
            embed_model=self._extract_embed_model(**kwargs),
        )

        return index

    @get_or_build_index
    def _build_faiss_index(self, config: FAISSRetrieverConfig, **kwargs) -> VectorStoreIndex:
        """构建 FAISS 索引"""
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(config.dimensions))

        return self._build_index_from_vector_store(config, vector_store, **kwargs)

    @get_or_build_index
    def _build_chroma_index(self, config: ChromaRetrieverConfig, **kwargs) -> VectorStoreIndex:
        """构建 Chroma 索引"""
        db = chromadb.PersistentClient(path=str(config.persist_path))
        chroma_collection = db.get_or_create_collection(config.collection_name, metadata=config.metadata)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        return self._build_index_from_vector_store(config, vector_store, **kwargs)

    @get_or_build_index
    def _build_milvus_index(self, config: MilvusRetrieverConfig, **kwargs) -> VectorStoreIndex:
        """构建 Milvus 索引"""
        vector_store = MilvusVectorStore(
            uri=config.uri, collection_name=config.collection_name, token=config.token, dim=config.dimensions
        )

        return self._build_index_from_vector_store(config, vector_store, **kwargs)

    @get_or_build_index
    def _build_es_index(self, config: ElasticsearchRetrieverConfig, **kwargs) -> VectorStoreIndex:
        """构建 Elasticsearch 索引"""
        vector_store = ElasticsearchStore(**config.store_config.model_dump())

        return self._build_index_from_vector_store(config, vector_store, **kwargs)

    def _build_index_from_vector_store(
        self, config: BaseRetrieverConfig, vector_store: BasePydanticVectorStore, **kwargs
    ) -> VectorStoreIndex:
        """根据向量存储构建索引"""
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=self._extract_nodes(config, **kwargs),
            storage_context=storage_context,
            embed_model=self._extract_embed_model(config, **kwargs),
        )

        return index


get_retriever = RetrieverFactory().get_retriever
