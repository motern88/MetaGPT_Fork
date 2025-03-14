"""Experience Manager."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from metagpt.config2 import Config
from metagpt.configs.exp_pool_config import ExperiencePoolRetrievalType
from metagpt.exp_pool.schema import DEFAULT_SIMILARITY_TOP_K, Experience, QueryType
from metagpt.logs import logger
from metagpt.utils.exceptions import handle_exception

if TYPE_CHECKING:
    from metagpt.rag.engines import SimpleEngine


class ExperienceManager(BaseModel):
    """ExperienceManager 管理经验的生命周期，包括 CRUD 操作和优化。

    参数:
        config (Config): 用于管理经验的配置。
        _storage (SimpleEngine): 处理经验存储和检索的引擎。
        _vector_store (ChromaVectorStore): 实际存储向量的地方。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config = Field(default_factory=Config.default)

    _storage: Any = None

    @property
    def storage(self) -> "SimpleEngine":
        """存储引擎属性，如果 _storage 为 None，则解析存储并返回。

        返回:
            SimpleEngine: 用于存储的引擎实例。
        """
        if self._storage is None:
            logger.info(f"exp_pool config: {self.config.exp_pool}")

            self._storage = self._resolve_storage()

        return self._storage

    @storage.setter
    def storage(self, value):
        """设置存储引擎。

        参数:
            value: 需要设置的存储引擎。
        """
        self._storage = value

    @property
    def is_readable(self) -> bool:
        """是否启用读取操作的属性。

        返回:
            bool: 是否启用读取操作。
        """
        return self.config.exp_pool.enabled and self.config.exp_pool.enable_read

    @is_readable.setter
    def is_readable(self, value: bool):
        """设置是否启用读取操作。

        参数:
            value: 是否启用读取操作。
        """
        self.config.exp_pool.enable_read = value

        # 如果启用读取操作，确保启用 exp_pool。
        if value:
            self.config.exp_pool.enabled = True

    @property
    def is_writable(self) -> bool:
        """是否启用写入操作的属性。

        返回:
            bool: 是否启用写入操作。
        """
        return self.config.exp_pool.enabled and self.config.exp_pool.enable_write

    @is_writable.setter
    def is_writable(self, value: bool):
        """设置是否启用写入操作。

        参数:
            value: 是否启用写入操作。
        """
        self.config.exp_pool.enable_write = value

        # 如果启用写入操作，确保启用 exp_pool。
        if value:
            self.config.exp_pool.enabled = True

    @handle_exception
    def create_exp(self, exp: Experience):
        """如果启用了写入操作，添加一个经验到存储。

        参数:
            exp (Experience): 要添加的经验。
        """
        self.create_exps([exp])

    @handle_exception
    def create_exps(self, exps: list[Experience]):
        """如果启用了写入操作，添加多个经验到存储。

        参数:
            exps (list[Experience]): 要添加的经验列表。
        """
        if not self.is_writable:
            return

        self.storage.add_objs(exps)
        self.storage.persist(self.config.exp_pool.persist_path)

    @handle_exception(default_return=[])
    async def query_exps(self, req: str, tag: str = "", query_type: QueryType = QueryType.SEMANTIC) -> list[Experience]:
        """检索和过滤经验。

        参数:
            req (str): 查询字符串，用于检索经验。
            tag (str): 可选的标签，用于根据标签过滤经验。
            query_type (QueryType): 默认语义匹配，exact 用于精确匹配。

        返回:
            list[Experience]: 匹配的经验列表。
        """
        if not self.is_readable:
            return []

        nodes = await self.storage.aretrieve(req)
        exps: list[Experience] = [node.metadata["obj"] for node in nodes]

        # TODO: 根据元数据过滤
        if tag:
            exps = [exp for exp in exps if exp.tag == tag]

        if query_type == QueryType.EXACT:
            exps = [exp for exp in exps if exp.req == req]

        return exps

    @handle_exception
    def delete_all_exps(self):
        """删除所有经验。"""
        if not self.is_writable:
            return

        self.storage.clear(persist_dir=self.config.exp_pool.persist_path)

    def get_exps_count(self) -> int:
        """获取经验的总数。

        返回:
            int: 经验总数。
        """
        return self.storage.count()

    def _resolve_storage(self) -> "SimpleEngine":
        """根据配置的检索类型选择合适的存储创建方法。

        返回:
            SimpleEngine: 创建的存储引擎。
        """
        storage_creators = {
            ExperiencePoolRetrievalType.BM25: self._create_bm25_storage,
            ExperiencePoolRetrievalType.CHROMA: self._create_chroma_storage,
        }

        return storage_creators[self.config.exp_pool.retrieval_type]()

    def _create_bm25_storage(self) -> "SimpleEngine":
        """创建或加载 BM25 存储。

        尝试创建一个新的 BM25 存储，如果指定的文档存储路径不存在，则创建。如果路径存在，则加载现有的 BM25 存储。

        返回:
            SimpleEngine: 配置了 BM25 存储的 SimpleEngine 实例。

        异常:
            ImportError: 如果需要的模块未安装。
        """
        try:
            from metagpt.rag.engines import SimpleEngine
            from metagpt.rag.schema import BM25IndexConfig, BM25RetrieverConfig
        except ImportError:
            raise ImportError("要使用经验池，您需要安装 rag 模块。")

        persist_path = Path(self.config.exp_pool.persist_path)
        docstore_path = persist_path / "docstore.json"

        ranker_configs = self._get_ranker_configs()

        if not docstore_path.exists():
            logger.debug(f"Path `{docstore_path}` 不存在，尝试创建新的 bm25 存储。")
            exps = [Experience(req="req", resp="resp")]

            retriever_configs = [BM25RetrieverConfig(create_index=True, similarity_top_k=DEFAULT_SIMILARITY_TOP_K)]

            storage = SimpleEngine.from_objs(
                objs=exps, retriever_configs=retriever_configs, ranker_configs=ranker_configs
            )
            return storage

        logger.debug(f"Path `{docstore_path}` 已存在，尝试加载 bm25 存储。")
        retriever_configs = [BM25RetrieverConfig(similarity_top_k=DEFAULT_SIMILARITY_TOP_K)]
        storage = SimpleEngine.from_index(
            BM25IndexConfig(persist_path=persist_path),
            retriever_configs=retriever_configs,
            ranker_configs=ranker_configs,
        )

        return storage

    def _create_chroma_storage(self) -> "SimpleEngine":
        """创建 Chroma 存储。

        返回:
            SimpleEngine: 配置了 Chroma 存储的 SimpleEngine 实例。

        异常:
            ImportError: 如果需要的模块未安装。
        """
        try:
            from metagpt.rag.engines import SimpleEngine
            from metagpt.rag.schema import ChromaRetrieverConfig
        except ImportError:
            raise ImportError("要使用经验池，您需要安装 rag 模块。")

        retriever_configs = [
            ChromaRetrieverConfig(
                persist_path=self.config.exp_pool.persist_path,
                collection_name=self.config.exp_pool.collection_name,
                similarity_top_k=DEFAULT_SIMILARITY_TOP_K,
            )
        ]
        ranker_configs = self._get_ranker_configs()

        storage = SimpleEngine.from_objs(retriever_configs=retriever_configs, ranker_configs=ranker_configs)

        return storage

    def _get_ranker_configs(self):
        """根据配置返回排名器配置。

        如果 `use_llm_ranker` 为 True，返回一个包含 LLMRankerConfig 实例的列表；否则返回一个空列表。

        返回:
            list: 包含 LLMRankerConfig 实例或空列表。
        """
        from metagpt.rag.schema import LLMRankerConfig

        return [LLMRankerConfig(top_n=DEFAULT_SIMILARITY_TOP_K)] if self.config.exp_pool.use_llm_ranker else []

_exp_manager = None


def get_exp_manager() -> ExperienceManager:
    global _exp_manager
    if _exp_manager is None:
        _exp_manager = ExperienceManager()
    return _exp_manager
