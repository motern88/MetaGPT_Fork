#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 16:33
@Author  : alexanderwu
@File    : llm_config.py
"""

from enum import Enum
from typing import Optional

from pydantic import field_validator

from metagpt.configs.compress_msg_config import CompressType
from metagpt.const import CONFIG_ROOT, LLM_API_TIMEOUT, METAGPT_ROOT
from metagpt.utils.yaml_model import YamlModel


class LLMType(Enum):
    OPENAI = "openai"  # OpenAI 提供的模型
    ANTHROPIC = "anthropic"  # Anthropic 提供的模型
    CLAUDE = "claude"  # Claude 模型的别名
    SPARK = "spark"  # Spark 提供的模型
    ZHIPUAI = "zhipuai"  # 智谱 AI 提供的模型
    FIREWORKS = "fireworks"  # Fireworks 提供的模型
    OPEN_LLM = "open_llm"  # 开放 LLM 模型
    GEMINI = "gemini"  # Google Gemini 模型
    METAGPT = "metagpt"  # Meta 提供的 GPT 模型
    AZURE = "azure"  # Microsoft Azure 提供的 LLM
    OLLAMA = "ollama"  # Ollama 提供的模型
    OLLAMA_GENERATE = "ollama.generate"  # Ollama 的 /generate 接口
    OLLAMA_EMBEDDINGS = "ollama.embeddings"  # Ollama 的 /embeddings 接口
    OLLAMA_EMBED = "ollama.embed"  # Ollama 的 /embed 接口
    QIANFAN = "qianfan"  # 百度提供的 Qianfan 模型
    DASHSCOPE = "dashscope"  # 阿里云 LingJi DashScope 提供的模型
    MOONSHOT = "moonshot"  # Moonshot 模型
    MISTRAL = "mistral"  # Mistral 提供的模型
    YI = "yi"  # Lingyiwanwu 提供的模型
    OPEN_ROUTER = "open_router"  # OpenRouter 路由模型
    DEEPSEEK = "deepseek"  # DeepSeek 模型
    SILICONFLOW = "siliconflow"  # SiliconFlow 模型
    OPENROUTER = "openrouter"  # OpenRouter 模型（别名）
    OPENROUTER_REASONING = "openrouter_reasoning"  # OpenRouter 推理模型
    BEDROCK = "bedrock"  # 亚马逊提供的 Bedrock 模型
    ARK = "ark"  # 火山引擎提供的 Ark 模型  # https://www.volcengine.com/docs/82379/1263482#python-sdk

    def __missing__(self, key):
        """如果未匹配到类型，默认返回 OpenAI。"""
        return self.OPENAI



class LLMConfig(YamlModel):
    """
    LLM 配置类。用于配置与语言模型相关的设置，包括 API 密钥、请求参数、云服务提供商信息等。

    示例:
    - api_key: "sk-"
    - api_type: LLMType.OPENAI
    - model: "gpt-3.5-turbo"
    - max_token: 4096
    - temperature: 0.7
    - timeout: 600
    """

    api_key: str = "sk-"  # API 密钥
    api_type: LLMType = LLMType.OPENAI  # 语言模型类型，默认使用 OpenAI
    base_url: str = "https://api.openai.com/v1"  # API 基础 URL
    api_version: Optional[str] = None  # API 版本（可选）

    model: Optional[str] = None  # 模型名称，通常也作为部署名称
    pricing_plan: Optional[str] = None  # 计费方案参数

    # 云服务提供商相关设置
    access_key: Optional[str] = None  # 访问密钥
    secret_key: Optional[str] = None  # 秘密密钥
    session_token: Optional[str] = None  # 会话令牌
    endpoint: Optional[str] = None  # 自部署模型的端点 URL

    # Spark (Xunfei) 特有参数（未来可能移除）
    app_id: Optional[str] = None  # 应用 ID
    api_secret: Optional[str] = None  # API 秘密
    domain: Optional[str] = None  # 域名

    # 对话生成相关参数
    max_token: int = 4096  # 最大 token 数
    temperature: float = 0.0  # 生成文本的温度
    top_p: float = 1.0  # 控制文本生成的多样性
    top_k: int = 0  # 控制多样性
    repetition_penalty: float = 1.0  # 重复惩罚
    stop: Optional[str] = None  # 停止词
    presence_penalty: float = 0.0  # 存在惩罚
    frequency_penalty: float = 0.0  # 频率惩罚
    best_of: Optional[int] = None  # 选择最好的生成结果
    n: Optional[int] = None  # 每次生成的结果数
    stream: bool = True  # 是否启用流式生成
    seed: Optional[int] = None  # 随机种子
    logprobs: Optional[bool] = None  # 是否返回 logprobs
    top_logprobs: Optional[int] = None  # 返回的 top logprobs 数量
    timeout: int = 600  # 超时时间（秒）
    context_length: Optional[int] = None  # 最大输入 token 数

    # 亚马逊 Bedrock 设置
    region_name: str = None  # 区域名称

    # 网络设置
    proxy: Optional[str] = None  # 代理

    # 成本控制
    calc_usage: bool = True  # 是否计算 API 使用量

    # 压缩请求消息，防止超过 token 限制
    compress_type: CompressType = CompressType.NO_COMPRESS  # 压缩类型

    # 控制消息的系统提示
    use_system_prompt: bool = True  # 是否使用系统提示

    # 推理/思维开关
    reasoning: bool = False  # 是否启用推理
    reasoning_max_token: int = 4000  # 推理预算 token 数，通常比最大 token 小

    @field_validator("api_key")
    @classmethod
    def check_llm_key(cls, v):
        """验证 API 密钥是否有效。"""
        if v in ["", None, "YOUR_API_KEY"]:
            repo_config_path = METAGPT_ROOT / "config/config2.yaml"
            root_config_path = CONFIG_ROOT / "config2.yaml"
            if root_config_path.exists():
                raise ValueError(
                    f"请在 {root_config_path} 中设置您的 API 密钥。如果您在 {repo_config_path} 中设置了配置，\n"
                    f"则前者会覆盖后者，这可能导致意外的结果。\n"
                )
            elif repo_config_path.exists():
                raise ValueError(f"请在 {repo_config_path} 中设置您的 API 密钥")
            else:
                raise ValueError("请在 config2.yaml 中设置您的 API 密钥")
        return v

    @field_validator("timeout")
    @classmethod
    def check_timeout(cls, v):
        """验证超时时间，确保有值。"""
        return v or LLM_API_TIMEOUT
