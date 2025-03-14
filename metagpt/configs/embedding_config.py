from enum import Enum
from typing import Optional

from pydantic import field_validator

from metagpt.utils.yaml_model import YamlModel


class EmbeddingType(Enum):
    """
    嵌入类型枚举。
    提供了不同的嵌入服务类型：
    - "openai"：OpenAI 嵌入服务。
    - "azure"：Azure 嵌入服务。
    - "gemini"：Gemini 嵌入服务。
    - "ollama"：Ollama 嵌入服务。
    """

    OPENAI = "openai"   # OpenAI 嵌入
    AZURE = "azure"     # Azure 嵌入
    GEMINI = "gemini"   # Gemini 嵌入
    OLLAMA = "ollama"   # Ollama 嵌入


class EmbeddingConfig(YamlModel):
    """嵌入配置类。

    示例:
    ---------
    api_type: "openai"       # 使用 OpenAI 嵌入
    api_key: "YOUR_API_KEY"  # OpenAI API 密钥
    dimensions: "YOUR_MODEL_DIMENSIONS"  # 模型输出维度

    api_type: "azure"        # 使用 Azure 嵌入
    api_key: "YOUR_API_KEY"  # Azure API 密钥
    base_url: "YOUR_BASE_URL" # Azure 基础 URL
    api_version: "YOUR_API_VERSION"  # Azure API 版本
    dimensions: "YOUR_MODEL_DIMENSIONS"  # 模型输出维度

    api_type: "gemini"       # 使用 Gemini 嵌入
    api_key: "YOUR_API_KEY"  # Gemini API 密钥

    api_type: "ollama"       # 使用 Ollama 嵌入
    base_url: "YOUR_BASE_URL" # Ollama 基础 URL
    model: "YOUR_MODEL"       # 使用的模型
    dimensions: "YOUR_MODEL_DIMENSIONS"  # 模型输出维度
    """

    api_type: Optional[EmbeddingType] = None  # 嵌入服务类型（例如 OpenAI、Azure）
    api_key: Optional[str] = None  # API 密钥
    base_url: Optional[str] = None  # 基础 URL（针对 Azure 或 Ollama）
    api_version: Optional[str] = None  # API 版本（针对 Azure）

    model: Optional[str] = None  # 使用的模型（针对 Ollama）
    embed_batch_size: Optional[int] = None  # 嵌入批处理大小
    dimensions: Optional[int] = None  # 嵌入模型的输出维度

    @field_validator("api_type", mode="before")
    @classmethod
    def check_api_type(cls, v):
        """检查 api_type 是否为空，若为空则返回 None"""
        if v == "":
            return None
        return v
