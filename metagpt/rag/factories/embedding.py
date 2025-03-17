"""RAG Embedding Factory."""
from __future__ import annotations

from typing import Any, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from metagpt.config2 import Config
from metagpt.configs.embedding_config import EmbeddingType
from metagpt.configs.llm_config import LLMType
from metagpt.rag.factories.base import GenericFactory


class RAGEmbeddingFactory(GenericFactory):
    """创建具有 MetaGPT 的嵌入配置的 LlamaIndex 嵌入实例。"""

    def __init__(self, config: Optional[Config] = None):
        creators = {
            EmbeddingType.OPENAI: self._create_openai,
            EmbeddingType.AZURE: self._create_azure,
            EmbeddingType.GEMINI: self._create_gemini,
            EmbeddingType.OLLAMA: self._create_ollama,
            # 为了向后兼容
            LLMType.OPENAI: self._create_openai,
            LLMType.AZURE: self._create_azure,
        }
        super().__init__(creators)
        self.config = config if config else Config.default()

    def get_rag_embedding(self, key: EmbeddingType = None) -> BaseEmbedding:
        """获取 RAG 嵌入实例。

        参数 `key` 是嵌入类型（EmbeddingType）。如果没有提供 `key`，则根据配置自动选择。
        """
        return super().get_instance(key or self._resolve_embedding_type())

    def _resolve_embedding_type(self) -> EmbeddingType | LLMType:
        """解析嵌入类型。

        如果没有指定嵌入类型，则会检查 LLM API 类型是否为 OPENAI 或 AZURE，向后兼容。
        如果未找到匹配类型，则抛出 TypeError。
        """
        if self.config.embedding.api_type:
            return self.config.embedding.api_type

        if self.config.llm.api_type in [LLMType.OPENAI, LLMType.AZURE]:
            return self.config.llm.api_type

        raise TypeError("使用 RAG 时，请在 config2.yaml 中设置嵌入配置。")

    def _create_openai(self) -> "OpenAIEmbedding":
        """创建 OpenAI 嵌入实例"""
        from llama_index.embeddings.openai import OpenAIEmbedding

        params = dict(
            api_key=self.config.embedding.api_key or self.config.llm.api_key,
            api_base=self.config.embedding.base_url or self.config.llm.base_url,
        )

        self._try_set_model_and_batch_size(params)

        return OpenAIEmbedding(**params)

    def _create_azure(self) -> AzureOpenAIEmbedding:
        """创建 Azure OpenAI 嵌入实例"""
        params = dict(
            api_key=self.config.embedding.api_key or self.config.llm.api_key,
            azure_endpoint=self.config.embedding.base_url or self.config.llm.base_url,
            api_version=self.config.embedding.api_version or self.config.llm.api_version,
        )

        self._try_set_model_and_batch_size(params)

        return AzureOpenAIEmbedding(**params)

    def _create_gemini(self) -> "GeminiEmbedding":
        """创建 Gemini 嵌入实例"""
        from llama_index.embeddings.gemini import GeminiEmbedding

        params = dict(
            api_key=self.config.embedding.api_key,
            api_base=self.config.embedding.base_url,
        )

        self._try_set_model_and_batch_size(params)

        return GeminiEmbedding(**params)

    def _create_ollama(self) -> "OllamaEmbedding":
        """创建 Ollama 嵌入实例"""
        from llama_index.embeddings.ollama import OllamaEmbedding

        params = dict(
            base_url=self.config.embedding.base_url,
        )

        self._try_set_model_and_batch_size(params)

        return OllamaEmbedding(**params)

    def _try_set_model_and_batch_size(self, params: dict):
        """仅在配置中指定时，设置 model_name 和 embed_batch_size 参数。"""
        if self.config.embedding.model:
            params["model_name"] = self.config.embedding.model

        if self.config.embedding.embed_batch_size:
            params["embed_batch_size"] = self.config.embedding.embed_batch_size

    def _raise_for_key(self, key: Any):
        """抛出嵌入类型不支持的异常"""
        raise ValueError(f"当前不支持嵌入类型: `{type(key)}`, {key}")


def get_rag_embedding(key: EmbeddingType = None, config: Optional[Config] = None):
    """获取 RAG 嵌入实例的封装函数"""
    return RAGEmbeddingFactory(config=config).get_rag_embedding(key)