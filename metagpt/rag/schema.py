"""RAG schemas."""
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Union

from chromadb.api.types import CollectionMetadata
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from metagpt.config2 import config
from metagpt.configs.embedding_config import EmbeddingType
from metagpt.logs import logger
from metagpt.rag.interface import RAGObject
from metagpt.rag.prompts.default_prompts import DEFAULT_CHOICE_SELECT_PROMPT


class BaseRetrieverConfig(BaseModel):
    """检索器的通用配置。

    如果添加新的子配置，需要在 rag.factories.retriever 中实现相应的实例。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    similarity_top_k: int = Field(default=5, description="检索时返回的最相似的 top-k 结果数量。")


class IndexRetrieverConfig(BaseRetrieverConfig):
    """基于索引的检索器配置。"""

    index: BaseIndex = Field(default=None, description="检索器的索引。")


class FAISSRetrieverConfig(IndexRetrieverConfig):
    """基于 FAISS 的检索器配置。"""

    dimensions: int = Field(default=0, description="FAISS 索引构建的向量维度。")

    _embedding_type_to_dimensions: ClassVar[dict[EmbeddingType, int]] = {
        EmbeddingType.GEMINI: 768,
        EmbeddingType.OLLAMA: 4096,
    }

    @model_validator(mode="after")
    def check_dimensions(self):
        if self.dimensions == 0:
            self.dimensions = config.embedding.dimensions or self._embedding_type_to_dimensions.get(
                config.embedding.api_type, 1536
            )
            if not config.embedding.dimensions and config.embedding.api_type not in self._embedding_type_to_dimensions:
                logger.warning(
                    f"未设置 {config.embedding.api_type} 的维度，默认使用 1536。"
                )

        return self


class BM25RetrieverConfig(IndexRetrieverConfig):
    """基于 BM25 的检索器配置。"""

    create_index: bool = Field(
        default=False,
        description="指示是否为节点创建索引。对于仅使用 BM25 但需要持久化数据的情况很有用。",
        exclude=True,
    )
    _no_embedding: bool = PrivateAttr(default=True)


class MilvusRetrieverConfig(IndexRetrieverConfig):
    """基于 Milvus 的检索器配置。"""

    uri: str = Field(default="./milvus_local.db", description="数据保存目录。")
    collection_name: str = Field(default="metagpt", description="集合名称。")
    token: str = Field(default=None, description="Milvus 的 token")
    metadata: Optional[CollectionMetadata] = Field(
        default=None, description="与集合关联的可选元数据"
    )
    dimensions: int = Field(default=0, description="Milvus 索引构建的向量维度。")

    _embedding_type_to_dimensions: ClassVar[dict[EmbeddingType, int]] = {
        EmbeddingType.GEMINI: 768,
        EmbeddingType.OLLAMA: 4096,
    }

    @model_validator(mode="after")
    def check_dimensions(self):
        if self.dimensions == 0:
            self.dimensions = config.embedding.dimensions or self._embedding_type_to_dimensions.get(
                config.embedding.api_type, 1536
            )
            if not config.embedding.dimensions and config.embedding.api_type not in self._embedding_type_to_dimensions:
                logger.warning(
                    f"未设置 {config.embedding.api_type} 的维度，默认使用 1536。"
                )

        return self


class ChromaRetrieverConfig(IndexRetrieverConfig):
    """基于 Chroma 的检索器配置。"""

    persist_path: Union[str, Path] = Field(default="./chroma_db", description="数据保存目录。")
    collection_name: str = Field(default="metagpt", description="集合名称。")
    metadata: Optional[CollectionMetadata] = Field(
        default=None, description="与集合关联的可选元数据"
    )


class ElasticsearchStoreConfig(BaseModel):
    index_name: str = Field(default="metagpt", description="Elasticsearch 索引名称。")
    es_url: str = Field(default=None, description="Elasticsearch URL。")
    es_cloud_id: str = Field(default=None, description="Elasticsearch 云 ID。")
    es_api_key: str = Field(default=None, description="Elasticsearch API 密钥。")
    es_user: str = Field(default=None, description="Elasticsearch 用户名。")
    es_password: str = Field(default=None, description="Elasticsearch 密码。")
    batch_size: int = Field(default=200, description="批量索引时的批次大小。")
    distance_strategy: str = Field(default="COSINE", description="用于相似性搜索的距离策略。")


class ElasticsearchRetrieverConfig(IndexRetrieverConfig):
    """基于 Elasticsearch 的检索器配置，支持向量和文本。"""

    store_config: ElasticsearchStoreConfig = Field(..., description="Elasticsearch 存储配置。")
    vector_store_query_mode: VectorStoreQueryMode = Field(
        default=VectorStoreQueryMode.DEFAULT, description="默认是向量查询。"
    )


class ElasticsearchKeywordRetrieverConfig(ElasticsearchRetrieverConfig):
    """基于 Elasticsearch 的检索器配置，仅支持文本。"""

    _no_embedding: bool = PrivateAttr(default=True)
    vector_store_query_mode: Literal[VectorStoreQueryMode.TEXT_SEARCH] = Field(
        default=VectorStoreQueryMode.TEXT_SEARCH, description="仅支持文本查询。"
    )


class BaseRankerConfig(BaseModel):
    """Ranker 的通用配置。

    如果添加新的子配置，需要在 rag.factories.ranker 中实现相应的实例。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    top_n: int = Field(default=5, description="返回的 top 结果数量。")


class LLMRankerConfig(BaseRankerConfig):
    """基于 LLM 的 Ranker 配置。"""

    llm: Any = Field(
        default=None,
        description="用于重排序的 LLM，使用 Any 而不是 LLM，因为 llama_index.core.llms.LLM 是 pydantic.v1。",
    )
    choice_select_prompt: Optional[BasePromptTemplate] = Field(
        default=DEFAULT_CHOICE_SELECT_PROMPT, description="选择提示模板。"
    )


class ColbertRerankConfig(BaseRankerConfig):
    model: str = Field(default="colbert-ir/colbertv2.0", description="Colbert 模型名称。")
    device: str = Field(default="cpu", description="用于句子转换的设备。")
    keep_retrieval_score: bool = Field(default=False, description="是否保留检索分数在元数据中。")


class CohereRerankConfig(BaseRankerConfig):
    model: str = Field(default="rerank-english-v3.0")
    api_key: str = Field(default="YOUR_COHERE_API")


class BGERerankConfig(BaseRankerConfig):
    model: str = Field(default="BAAI/bge-reranker-large", description="BAAI Reranker模型名称。")
    use_fp16: bool = Field(default=True, description="是否使用fp16进行推理。")


class ObjectRankerConfig(BaseRankerConfig):
    field_name: str = Field(..., description="对象的字段名，字段的值必须可比较。")
    order: Literal["desc", "asc"] = Field(default="desc", description="排序方向。")


class BaseIndexConfig(BaseModel):
    """索引的通用配置。

    如果添加新的子配置，需要在rag.factories.index中添加相应的实例实现。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    persist_path: Union[str, Path] = Field(description="保存数据的目录。")


class VectorIndexConfig(BaseIndexConfig):
    """基于向量的索引配置。"""

    embed_model: BaseEmbedding = Field(default=None, description="嵌入模型。")


class FAISSIndexConfig(VectorIndexConfig):
    """基于FAISS的索引配置。"""


class ChromaIndexConfig(VectorIndexConfig):
    """基于Chroma的索引配置。"""

    collection_name: str = Field(default="metagpt", description="集合的名称。")
    metadata: Optional[CollectionMetadata] = Field(
        default=None, description="可选的元数据，关联到集合。"
    )


class MilvusIndexConfig(VectorIndexConfig):
    """基于Milvus的索引配置。"""

    collection_name: str = Field(default="metagpt", description="集合的名称。")
    uri: str = Field(default="./milvus_local.db", description="索引的uri。")
    token: Optional[str] = Field(default=None, description="索引的token。")
    metadata: Optional[CollectionMetadata] = Field(
        default=None, description="可选的元数据，关联到集合。"
    )


class BM25IndexConfig(BaseIndexConfig):
    """基于BM25的索引配置。"""

    _no_embedding: bool = PrivateAttr(default=True)


class ElasticsearchIndexConfig(VectorIndexConfig):
    """基于Elasticsearch的索引配置。"""

    store_config: ElasticsearchStoreConfig = Field(..., description="ElasticsearchStore配置。")
    persist_path: Union[str, Path] = ""


class ElasticsearchKeywordIndexConfig(ElasticsearchIndexConfig):
    """基于Elasticsearch的索引配置，且不使用嵌入。"""

    _no_embedding: bool = PrivateAttr(default=True)


class ObjectNodeMetadata(BaseModel):
    """ObjectNode的元数据。"""

    is_obj: bool = Field(default=True)
    obj: Any = Field(default=None, description="当RAG检索时，会根据obj_json重建对象。")
    obj_json: str = Field(..., description="对象的json表示，例如obj.model_dump_json()")
    obj_cls_name: str = Field(..., description="对象的类名，例如obj.__class__.__name__")
    obj_mod_name: str = Field(..., description="类的模块名，例如obj.__class__.__module__")


class ObjectNode(TextNode):
    """RAG添加对象。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.excluded_llm_metadata_keys = list(ObjectNodeMetadata.model_fields.keys())
        self.excluded_embed_metadata_keys = self.excluded_llm_metadata_keys

    @staticmethod
    def get_obj_metadata(obj: RAGObject) -> dict:
        metadata = ObjectNodeMetadata(
            obj_json=obj.model_dump_json(), obj_cls_name=obj.__class__.__name__, obj_mod_name=obj.__class__.__module__
        )

        return metadata.model_dump()


class OmniParseType(str, Enum):
    """OmniParse类型枚举"""

    PDF = "PDF"
    DOCUMENT = "DOCUMENT"


class ParseResultType(str, Enum):
    """解析结果类型枚举"""

    TXT = "text"
    MD = "markdown"
    JSON = "json"


class OmniParseOptions(BaseModel):
    """OmniParse配置选项"""

    result_type: ParseResultType = Field(default=ParseResultType.MD, description="OmniParse的结果类型")
    parse_type: OmniParseType = Field(default=OmniParseType.DOCUMENT, description="OmniParse的解析类型")
    max_timeout: Optional[int] = Field(default=120, description="OmniParse服务请求的最大超时限制")
    num_workers: int = Field(
        default=5,
        gt=0,
        lt=10,
        description="多个文件并发请求的数量",
    )


class OminParseImage(BaseModel):
    image: str = Field(default="", description="图像的字节字符串")
    image_name: str = Field(default="", description="图像名称")
    image_info: Optional[dict] = Field(default={}, description="图像的元信息")


class OmniParsedResult(BaseModel):
    markdown: str = Field(default="", description="markdown格式文本")
    text: str = Field(default="", description="纯文本")
    images: Optional[List[OminParseImage]] = Field(default=[], description="图像列表")
    metadata: Optional[dict] = Field(default={}, description="元数据")

    @model_validator(mode="before")
    def set_markdown(cls, values):
        if not values.get("markdown"):
            values["markdown"] = values.get("text")
        return values
