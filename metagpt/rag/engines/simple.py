"""Simple Engine."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Set, Union

import fsspec
from llama_index.core import SimpleDirectoryReader
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers.base import BaseReader
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    get_response_synthesizer,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeWithScore,
    QueryBundle,
    QueryType,
    TransformComponent,
)

from metagpt.config2 import config
from metagpt.rag.factories import (
    get_index,
    get_rag_embedding,
    get_rag_llm,
    get_rankers,
    get_retriever,
)
from metagpt.rag.interface import NoEmbedding, RAGObject
from metagpt.rag.parsers import OmniParse
from metagpt.rag.retrievers.base import (
    DeletableRAGRetriever,
    ModifiableRAGRetriever,
    PersistableRAGRetriever,
    QueryableRAGRetriever,
)
from metagpt.rag.retrievers.hybrid_retriever import SimpleHybridRetriever
from metagpt.rag.schema import (
    BaseIndexConfig,
    BaseRankerConfig,
    BaseRetrieverConfig,
    BM25RetrieverConfig,
    ObjectNode,
    OmniParseOptions,
    OmniParseType,
    ParseResultType,
)
from metagpt.utils.common import import_class


class SimpleEngine(RetrieverQueryEngine):
    """SimpleEngine 设计为简单易用。

    它是一个轻量级且易于使用的搜索引擎，集成了文档读取、嵌入、索引、检索和排序功能，
    形成一个简单直接的工作流程。它旨在快速从文档集合中设置一个搜索引擎。
    """

    def __init__(
        self,
        retriever: BaseRetriever,  # 检索器
        response_synthesizer: Optional[BaseSynthesizer] = None,  # 响应合成器
        node_postprocessors: Optional[list[BaseNodePostprocessor]] = None,  # 节点后处理器
        callback_manager: Optional[CallbackManager] = None,  # 回调管理器
        transformations: Optional[list[TransformComponent]] = None,  # 转换组件
    ) -> None:
        super().__init__(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
        )
        self._transformations = transformations or self._default_transformations()  # 转换组件
        self._filenames = set()  # 文件名集合

    @classmethod
    def from_docs(
        cls,
        input_dir: str = None,  # 输入目录
        input_files: list[str] = None,  # 输入文件列表
        transformations: Optional[list[TransformComponent]] = None,  # 转换组件
        embed_model: BaseEmbedding = None,  # 嵌入模型
        llm: LLM = None,  # 大语言模型
        retriever_configs: list[BaseRetrieverConfig] = None,  # 检索器配置
        ranker_configs: list[BaseRankerConfig] = None,  # 排序器配置
        fs: Optional[fsspec.AbstractFileSystem] = None,  # 文件系统
    ) -> "SimpleEngine":
        """从文档创建 SimpleEngine。

        必须提供 `input_dir` 或 `input_files`。

        参数：
            input_dir: 目录路径。
            input_files: 要读取的文件路径列表（可选；覆盖 input_dir 和 exclude）。
            transformations: 将文档解析为节点的转换组件。默认为 [SentenceSplitter]。
            embed_model: 将节点解析为嵌入的模型。必须支持 llama index。默认为 OpenAIEmbedding。
            llm: 必须支持 llama index。默认为 OpenAI。
            retriever_configs: 检索器的配置。如果有多个配置，将使用 SimpleHybridRetriever。
            ranker_configs: 排序器的配置。
            fs: 使用的文件系统。
        """
        if not input_dir and not input_files:
            raise ValueError("必须提供 `input_dir` 或 `input_files`。")

        file_extractor = cls._get_file_extractor()  # 获取文件提取器
        documents = SimpleDirectoryReader(
            input_dir=input_dir, input_files=input_files, file_extractor=file_extractor, fs=fs
        ).load_data()  # 加载文档
        cls._fix_document_metadata(documents)  # 修复文档元数据

        transformations = transformations or cls._default_transformations()  # 获取转换组件
        nodes = run_transformations(documents, transformations=transformations)  # 运行转换

        return cls._from_nodes(
            nodes=nodes,
            transformations=transformations,
            embed_model=embed_model,
            llm=llm,
            retriever_configs=retriever_configs,
            ranker_configs=ranker_configs,
        )  # 从节点创建 SimpleEngine

    @classmethod
    def from_objs(
        cls,
        objs: Optional[list[RAGObject]] = None,  # RAG 对象列表
        transformations: Optional[list[TransformComponent]] = None,  # 转换组件
        embed_model: BaseEmbedding = None,  # 嵌入模型
        llm: LLM = None,  # 大语言模型
        retriever_configs: list[BaseRetrieverConfig] = None,  # 检索器配置
        ranker_configs: list[BaseRankerConfig] = None,  # 排序器配置
    ) -> "SimpleEngine":
        """从 RAG 对象创建 SimpleEngine。

        参数：
            objs: RAG 对象列表。
            transformations: 将文档解析为节点的转换组件。默认为 [SentenceSplitter]。
            embed_model: 将节点解析为嵌入的模型。必须支持 llama index。默认为 OpenAIEmbedding。
            llm: 必须支持 llama index。默认为 OpenAI。
            retriever_configs: 检索器的配置。如果有多个配置，将使用 SimpleHybridRetriever。
            ranker_configs: 排序器的配置。
        """
        objs = objs or []
        retriever_configs = retriever_configs or []

        if not objs and any(isinstance(config, BM25RetrieverConfig) for config in retriever_configs):
            raise ValueError("在 BM25RetrieverConfig 中，objs 不能为空。")

        nodes = cls.get_obj_nodes(objs)  # 获取 RAG 对象的节点

        return cls._from_nodes(
            nodes=nodes,
            transformations=transformations,
            embed_model=embed_model,
            llm=llm,
            retriever_configs=retriever_configs,
            ranker_configs=ranker_configs,
        )  # 从节点创建 SimpleEngine

    @classmethod
    def from_index(
        cls,
        index_config: BaseIndexConfig,  # 索引配置
        embed_model: BaseEmbedding = None,  # 嵌入模型
        llm: LLM = None,  # 大语言模型
        retriever_configs: list[BaseRetrieverConfig] = None,  # 检索器配置
        ranker_configs: list[BaseRankerConfig] = None,  # 排序器配置
    ) -> "SimpleEngine":
        """从之前维护的索引加载，index_config 包含持久化路径。"""
        index = get_index(index_config, embed_model=cls._resolve_embed_model(embed_model, [index_config]))  # 获取索引
        return cls._from_index(index, llm=llm, retriever_configs=retriever_configs, ranker_configs=ranker_configs)  # 从索引创建 SimpleEngine

    async def asearch(self, content: str, **kwargs) -> str:
        """实现 tools.SearchInterface 接口"""
        return await self.aquery(content)  # 异步查询

    def retrieve(self, query: QueryType) -> list[NodeWithScore]:
        """检索节点"""
        query_bundle = QueryBundle(query) if isinstance(query, str) else query  # 创建查询包

        nodes = super().retrieve(query_bundle)  # 检索节点
        self._try_reconstruct_obj(nodes)  # 尝试重建对象
        return nodes

    async def aretrieve(self, query: QueryType) -> list[NodeWithScore]:
        """允许查询为字符串"""
        query_bundle = QueryBundle(query) if isinstance(query, str) else query  # 创建查询包

        nodes = await super().aretrieve(query_bundle)  # 异步检索节点
        self._try_reconstruct_obj(nodes)  # 尝试重建对象
        return nodes

    def add_docs(self, input_files: List[Union[str, Path]]):
        """向检索器添加文档。检索器必须具有 add_nodes 函数。"""
        self._ensure_retriever_modifiable()  # 确保检索器可修改

        documents = SimpleDirectoryReader(input_files=[str(i) for i in input_files]).load_data()  # 加载文档
        self._fix_document_metadata(documents)  # 修复文档元数据

        nodes = run_transformations(documents, transformations=self._transformations)  # 运行转换
        self._save_nodes(nodes)  # 保存节点

    def add_objs(self, objs: list[RAGObject]):
        """向检索器添加对象，将每个对象的原始形式存储在元数据中以供将来参考。"""
        self._ensure_retriever_modifiable()  # 确保检索器可修改

        nodes = self.get_obj_nodes(objs)  # 获取对象的节点
        self._save_nodes(nodes)  # 保存节点

    def persist(self, persist_dir: Union[str, os.PathLike], **kwargs):
        """持久化。"""
        self._ensure_retriever_persistable()  # 确保检索器可持久化

        self._persist(str(persist_dir), **kwargs)  # 持久化到指定目录

    def count(self) -> int:
        """计数。"""
        self._ensure_retriever_queryable()  # 确保检索器可查询

        return self.retriever.query_total_count()  # 返回检索器的总计数

    def clear(self, **kwargs):
        """清除。"""
        self._ensure_retriever_deletable()  # 确保检索器可删除

        return self.retriever.clear(**kwargs)  # 清除检索器

    def delete_docs(self, input_files: List[Union[str, Path]]):
        """从索引和文档存储中删除文档。

        参数：
            input_files (List[Union[str, Path]]): 要删除的文件路径或文件名的列表。

        抛出：
            NotImplementedError: 如果方法未实现。
        """
        exists_filenames = set()  # 存在的文件名集合
        filenames = {str(i) for i in input_files}  # 输入文件名集合
        for doc_id, info in self.retriever._index.ref_doc_info.items():  # 遍历文档信息
            if info.metadata.get("file_path") in filenames:  # 如果文件路径在输入文件名集合中
                exists_filenames.add(doc_id)  # 添加到存在的文件名集合

        for doc_id in exists_filenames:  # 遍历存在的文件名集合
            self.retriever._index.delete_ref_doc(doc_id, delete_from_docstore=True)  # 删除文档

    @staticmethod
    def get_obj_nodes(objs: Optional[list[RAGObject]] = None) -> list[ObjectNode]:
        """将 RAGObjects 列表转换为 ObjectNodes 列表。"""

        return [ObjectNode(text=obj.rag_key(), metadata=ObjectNode.get_obj_metadata(obj)) for obj in objs]  # 返回对象节点列表

    @classmethod
    def _from_nodes(
            cls,
            nodes: list[BaseNode],  # 节点列表
            transformations: Optional[list[TransformComponent]] = None,  # 转换组件
            embed_model: BaseEmbedding = None,  # 嵌入模型
            llm: LLM = None,  # 大语言模型
            retriever_configs: list[BaseRetrieverConfig] = None,  # 检索器配置
            ranker_configs: list[BaseRankerConfig] = None,  # 排序器配置
    ) -> "SimpleEngine":
        embed_model = cls._resolve_embed_model(embed_model, retriever_configs)  # 解析嵌入模型
        llm = llm or get_rag_llm()  # 获取大语言模型

        retriever = get_retriever(configs=retriever_configs, nodes=nodes, embed_model=embed_model)  # 获取检索器
        rankers = get_rankers(configs=ranker_configs, llm=llm)  # 获取排序器，默认为空列表

        return cls(
            retriever=retriever,
            node_postprocessors=rankers,
            response_synthesizer=get_response_synthesizer(llm=llm),
            transformations=transformations,
        )  # 返回 SimpleEngine 实例

    @classmethod
    def _from_index(
            cls,
            index: BaseIndex,  # 索引
            llm: LLM = None,  # 大语言模型
            retriever_configs: list[BaseRetrieverConfig] = None,  # 检索器配置
            ranker_configs: list[BaseRankerConfig] = None,  # 排序器配置
    ) -> "SimpleEngine":
        llm = llm or get_rag_llm()  # 获取大语言模型

        retriever = get_retriever(configs=retriever_configs, index=index)  # 获取检索器，默认为 index.as_retriever
        rankers = get_rankers(configs=ranker_configs, llm=llm)  # 获取排序器，默认为空列表

        return cls(
            retriever=retriever,
            node_postprocessors=rankers,
            response_synthesizer=get_response_synthesizer(llm=llm),
        )  # 返回 SimpleEngine 实例

    def _ensure_retriever_modifiable(self):
        self._ensure_retriever_of_type(ModifiableRAGRetriever)  # 确保检索器可修改

    def _ensure_retriever_persistable(self):
        self._ensure_retriever_of_type(PersistableRAGRetriever)  # 确保检索器可持久化

    def _ensure_retriever_queryable(self):
        self._ensure_retriever_of_type(QueryableRAGRetriever)  # 确保检索器可查询

    def _ensure_retriever_deletable(self):
        self._ensure_retriever_of_type(DeletableRAGRetriever)  # 确保检索器可删除

    def _ensure_retriever_of_type(self, required_type: BaseRetriever):
        """确保 self.retriever 是 required_type 类型，或者至少是 SimpleHybridRetriever 中的一个组件。

        参数：
            required_type: 检索器期望的类类型。
        """
        if isinstance(self.retriever, SimpleHybridRetriever):  # 如果是 SimpleHybridRetriever
            if not any(isinstance(r, required_type) for r in self.retriever.retrievers):  # 如果没有一个检索器是 required_type 类型
                raise TypeError(
                    f"在 SimpleHybridRetriever 中必须至少有一个 {required_type.__name__} 类型的检索器"
                )

        if not isinstance(self.retriever, required_type):  # 如果检索器不是 required_type 类型
            raise TypeError(f"检索器不是 {required_type.__name__} 类型: {type(self.retriever)}")

    def _save_nodes(self, nodes: list[BaseNode]):
        self.retriever.add_nodes(nodes)  # 保存节点

    def _persist(self, persist_dir: str, **kwargs):
        self.retriever.persist(persist_dir, **kwargs)  # 持久化到指定目录

    @staticmethod
    def _try_reconstruct_obj(nodes: list[NodeWithScore]):
        """如果节点是对象，则动态重建对象，并将对象保存到 node.metadata["obj"] 中。"""
        for node in nodes:  # 遍历节点
            if node.metadata.get("is_obj", False):  # 如果节点是对象
                obj_cls = import_class(node.metadata["obj_cls_name"], node.metadata["obj_mod_name"])  # 导入对象类
                obj_dict = json.loads(node.metadata["obj_json"])  # 加载对象字典
                node.metadata["obj"] = obj_cls(**obj_dict)  # 重建对象并保存到元数据中

    @staticmethod
    def _fix_document_metadata(documents: list[Document]):
        """LlamaIndex 保留 metadata['file_path']，这是不必要的，可能在不久的将来删除。"""
        for doc in documents:  # 遍历文档
            doc.excluded_embed_metadata_keys.append("file_path")  # 将 file_path 添加到排除的嵌入元数据键中

    @staticmethod
    @staticmethod
    def _resolve_embed_model(embed_model: BaseEmbedding = None, configs: list[Any] = None) -> BaseEmbedding:
        """解析嵌入模型。

        如果所有配置都是 NoEmbedding 类型，则返回 MockEmbedding。
        否则返回传入的嵌入模型或默认的 RAG 嵌入模型。

        参数：
            embed_model: 嵌入模型。
            configs: 配置列表。

        返回：
            BaseEmbedding: 解析后的嵌入模型。
        """
        if configs and all(isinstance(c, NoEmbedding) for c in configs):  # 如果所有配置都是 NoEmbedding 类型
            return MockEmbedding(embed_dim=1)  # 返回 MockEmbedding

        return embed_model or get_rag_embedding()  # 返回传入的嵌入模型或默认的 RAG 嵌入模型

    @staticmethod
    def _default_transformations():
        """获取默认的转换组件。"""
        return [SentenceSplitter()]  # 返回句子分割器

    @property
    def filenames(self) -> Set[str]:
        """获取文件名集合。"""
        return self._filenames  # 返回文件名集合

    @staticmethod
    def _get_file_extractor() -> dict[str:BaseReader]:
        """获取文件提取器。

        目前，只有 PDF 使用 OmniParse。其他文档类型使用 llama_index 的内置读取器。

        返回：
            dict[file_type: BaseReader]: 文件类型到读取器的映射。
        """
        file_extractor: dict[str:BaseReader] = {}  # 文件提取器字典
        if config.omniparse.base_url:  # 如果 OmniParse 的 base_url 存在
            pdf_parser = OmniParse(
                api_key=config.omniparse.api_key,  # API Key
                base_url=config.omniparse.base_url,  # 基础 URL
                parse_options=OmniParseOptions(parse_type=OmniParseType.PDF, result_type=ParseResultType.MD),  # 解析选项
            )
            file_extractor[".pdf"] = pdf_parser  # 将 PDF 解析器添加到文件提取器

        return file_extractor  # 返回文件提取器
