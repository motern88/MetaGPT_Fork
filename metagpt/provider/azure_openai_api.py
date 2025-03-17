# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : openai.py
@Modified By: mashenquan, 2023/11/21. Fix bug: ReadTimeout.
@Modified By: mashenquan, 2023/12/1. Fix bug: Unclosed connection caused by openai 0.x.
"""
from openai import AsyncAzureOpenAI
from openai._base_client import AsyncHttpxClientWrapper

from metagpt.configs.llm_config import LLMType
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.provider.openai_api import OpenAILLM


@register_provider(LLMType.AZURE)
class AzureOpenAILLM(OpenAILLM):
    """
    请查看 https://platform.openai.com/examples 以获取示例
    """

    def _init_client(self):
        """初始化客户端"""
        kwargs = self._make_client_kwargs()  # 创建客户端所需的参数
        # 参考文档：https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/migration?tabs=python-new%2Cdalle-fix
        self.aclient = AsyncAzureOpenAI(**kwargs)  # 初始化Azure OpenAI的异步客户端
        self.model = self.config.model  # 模型名称，供_calc_usage和_cons_kwargs使用
        self.pricing_plan = self.config.pricing_plan or self.model  # 定价计划，默认为模型名称

    def _make_client_kwargs(self) -> dict:
        """构建客户端所需的参数"""
        kwargs = dict(
            api_key=self.config.api_key,  # Azure OpenAI API密钥
            api_version=self.config.api_version,  # Azure OpenAI API版本
            azure_endpoint=self.config.base_url,  # Azure OpenAI的端点URL
        )

        # 如果需要使用代理，配置http_client
        proxy_params = self._get_proxy_params()  # 获取代理参数
        if proxy_params:  # 如果有代理参数，则将其传递给http_client
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs
