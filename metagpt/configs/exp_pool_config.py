from enum import Enum

from pydantic import Field

from metagpt.utils.yaml_model import YamlModel


class ExperiencePoolRetrievalType(Enum):
    """
    经验池检索类型枚举。
    - BM25：经典的 BM25 检索算法。
    - CHROMA：基于 Chroma 的检索算法。
    """

    BM25 = "bm25"  # 使用 BM25 检索算法
    CHROMA = "chroma"  # 使用 Chroma 检索算法


class ExperiencePoolConfig(YamlModel):
    """
    经验池配置类。

    启用和配置经验池的相关设置。经验池用于存储和检索数据，根据配置可以启用/禁用读取和写入操作。

    配置项:
    - enabled: 启用/禁用经验池功能，禁用时读取和写入都无效。
    - enable_read: 是否启用从经验池读取数据。
    - enable_write: 是否启用向经验池写入数据。
    - persist_path: 经验池的持久化存储路径。
    - retrieval_type: 经验池的检索类型（BM25 或 Chroma）。
    - use_llm_ranker: 是否使用 LLM 排序器来获取更好的结果。
    - collection_name: 在 ChromaDB 中使用的集合名称。
    """

    enabled: bool = Field(
        default=False,
        description="启用或禁用经验池。当禁用时，读取和写入都无效。"
    )
    enable_read: bool = Field(default=False, description="启用从经验池读取数据。")
    enable_write: bool = Field(default=False, description="启用向经验池写入数据。")
    persist_path: str = Field(default=".chroma_exp_data", description="经验池的持久化存储路径。")
    retrieval_type: ExperiencePoolRetrievalType = Field(
        default=ExperiencePoolRetrievalType.BM25, description="经验池的检索类型。"
    )
    use_llm_ranker: bool = Field(default=True, description="是否使用 LLM 排序器来获取更好的结果。")
    collection_name: str = Field(default="experience_pool", description="在 ChromaDB 中使用的集合名称。")

