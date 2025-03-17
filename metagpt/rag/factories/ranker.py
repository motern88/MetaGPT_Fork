"""RAG Ranker Factory."""

from llama_index.core.llms import LLM
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from metagpt.rag.factories.base import ConfigBasedFactory
from metagpt.rag.rankers.object_ranker import ObjectSortPostprocessor
from metagpt.rag.schema import (
    BaseRankerConfig,
    BGERerankConfig,
    CohereRerankConfig,
    ColbertRerankConfig,
    LLMRankerConfig,
    ObjectRankerConfig,
)


class RankerFactory(ConfigBasedFactory):
    """修改创建者方法，以便动态实例化不同的排序器实现。"""

    def __init__(self):
        """初始化 RankerFactory，定义了不同配置对应的排序器创建方法。"""
        creators = {
            LLMRankerConfig: self._create_llm_ranker,  # LLM排序器的创建方法
            ColbertRerankConfig: self._create_colbert_ranker,  # Colbert排序器的创建方法
            ObjectRankerConfig: self._create_object_ranker,  # Object排序器的创建方法
            CohereRerankConfig: self._create_cohere_rerank,  # Cohere排序器的创建方法
            BGERerankConfig: self._create_bge_rerank,  # BGE排序器的创建方法
        }
        super().__init__(creators)  # 调用父类的初始化方法，传入创建方法映射

    def get_rankers(self, configs: list[BaseRankerConfig] = None, **kwargs) -> list[BaseNodePostprocessor]:
        """根据提供的配置创建并返回排序器实例。

        参数:
        - configs: 排序器的配置列表
        - kwargs: 其他附加参数

        返回:
        - 返回一个排序器实例的列表
        """
        if not configs:
            return []  # 如果没有配置，返回空列表

        return super().get_instances(configs, **kwargs)  # 调用父类方法创建实例并返回

    def _create_llm_ranker(self, config: LLMRankerConfig, **kwargs) -> LLMRerank:
        """根据LLMRankerConfig配置创建LLM排序器。

        参数:
        - config: LLM排序器的配置
        - kwargs: 其他附加参数

        返回:
        - 返回创建的LLM排序器实例
        """
        config.llm = self._extract_llm(config, **kwargs)  # 提取LLM模型
        return LLMRerank(**config.model_dump())  # 返回LLM排序器实例

    def _create_colbert_ranker(self, config: ColbertRerankConfig, **kwargs) -> LLMRerank:
        """根据ColbertRerankConfig配置创建Colbert排序器。

        参数:
        - config: Colbert排序器的配置
        - kwargs: 其他附加参数

        返回:
        - 返回创建的Colbert排序器实例
        """
        try:
            from llama_index.postprocessor.colbert_rerank import ColbertRerank
        except ImportError:
            raise ImportError(
                "`llama-index-postprocessor-colbert-rerank`包未找到，请运行 `pip install llama-index-postprocessor-colbert-rerank`"
            )
        return ColbertRerank(**config.model_dump())  # 返回Colbert排序器实例

    def _create_cohere_rerank(self, config: CohereRerankConfig, **kwargs) -> LLMRerank:
        """根据CohereRerankConfig配置创建Cohere排序器。

        参数:
        - config: Cohere排序器的配置
        - kwargs: 其他附加参数

        返回:
        - 返回创建的Cohere排序器实例
        """
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
        except ImportError:
            raise ImportError(
                "`llama-index-postprocessor-cohere-rerank`包未找到，请运行 `pip install llama-index-postprocessor-cohere-rerank`"
            )
        return CohereRerank(**config.model_dump())  # 返回Cohere排序器实例

    def _create_bge_rerank(self, config: BGERerankConfig, **kwargs) -> LLMRerank:
        """根据BGERerankConfig配置创建BGE排序器。

        参数:
        - config: BGE排序器的配置
        - kwargs: 其他附加参数

        返回:
        - 返回创建的BGE排序器实例
        """
        try:
            from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        except ImportError:
            raise ImportError(
                "`llama-index-postprocessor-flag-embedding-reranker`包未找到，请运行 `pip install llama-index-postprocessor-flag-embedding-reranker`"
            )
        return FlagEmbeddingReranker(**config.model_dump())  # 返回BGE排序器实例

    def _create_object_ranker(self, config: ObjectRankerConfig, **kwargs) -> LLMRerank:
        """根据ObjectRankerConfig配置创建Object排序器。

        参数:
        - config: Object排序器的配置
        - kwargs: 其他附加参数

        返回:
        - 返回创建的Object排序器实例
        """
        return ObjectSortPostprocessor(**config.model_dump())  # 返回Object排序器实例

    def _extract_llm(self, config: BaseRankerConfig = None, **kwargs) -> LLM:
        """从配置或附加参数中提取LLM实例。

        参数:
        - config: 排序器配置
        - kwargs: 其他附加参数

        返回:
        - 返回提取的LLM实例
        """
        return self._val_from_config_or_kwargs("llm", config, **kwargs)


# 获取排序器实例的快捷方法
get_rankers = RankerFactory().get_rankers
