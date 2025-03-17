#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provider for volcengine.
See Also: https://console.volcengine.com/ark/region:ark+cn-beijing/model

config2.yaml example:
```yaml
llm:
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  api_type: "ark"
  endpoint: "ep-2024080514****-d****"
  api_key: "d47****b-****-****-****-d6e****0fd77"
  pricing_plan: "doubao-lite"
```
"""
from typing import Optional, Union

from pydantic import BaseModel
from volcenginesdkarkruntime import AsyncArk
from volcenginesdkarkruntime._base_client import AsyncHttpxClientWrapper
from volcenginesdkarkruntime._streaming import AsyncStream
from volcenginesdkarkruntime.types.chat import ChatCompletion, ChatCompletionChunk

from metagpt.configs.llm_config import LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.provider.openai_api import OpenAILLM
from metagpt.utils.token_counter import DOUBAO_TOKEN_COSTS


@register_provider(LLMType.ARK)
class ArkLLM(OpenAILLM):
    """
    用于火山方舟的API
    见：https://www.volcengine.com/docs/82379/1263482
    """

    aclient: Optional[AsyncArk] = None

    def _init_client(self):
        """初始化客户端: https://github.com/openai/openai-python#async-usage"""
        self.model = (
            self.config.endpoint or self.config.model
        )  # 模型名称或端点名称，更多信息见: https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint
        self.pricing_plan = self.config.pricing_plan or self.model  # 定价计划，默认为模型名称
        kwargs = self._make_client_kwargs()  # 创建客户端所需的参数
        self.aclient = AsyncArk(**kwargs)  # 初始化火山方舟的异步客户端

    def _make_client_kwargs(self) -> dict:
        """构建客户端所需的参数"""
        kvs = {
            "ak": self.config.access_key,  # 访问密钥
            "sk": self.config.secret_key,  # 秘密密钥
            "api_key": self.config.api_key,  # API密钥
            "base_url": self.config.base_url,  # 基础URL
        }
        kwargs = {k: v for k, v in kvs.items() if v}  # 过滤掉值为None的项

        # 如果需要使用代理，配置http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _update_costs(self, usage: Union[dict, BaseModel], model: str = None, local_calc_usage: bool = True):
        """更新消耗的成本信息"""
        # 如果没有token成本信息，则添加默认的DOUBAO_TOKEN_COSTS
        if next(iter(DOUBAO_TOKEN_COSTS)) not in self.cost_manager.token_costs:
            self.cost_manager.token_costs.update(DOUBAO_TOKEN_COSTS)
        if model in self.cost_manager.token_costs:
            self.pricing_plan = model  # 如果模型在成本管理器中，更新定价计划
        if self.pricing_plan in self.cost_manager.token_costs:
            super()._update_costs(usage, self.pricing_plan, local_calc_usage)  # 更新费用信息

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        """处理火山方舟的流式响应"""
        # 启动流式调用并返回响应
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout)),
            stream=True,
            extra_body={"stream_options": {"include_usage": True}},  # 流式返回时必须增加此参数，以便返回usage信息
        )
        usage = None  # 用于存储usage
        collected_messages = []  # 收集流式消息的内容
        async for chunk in response:
            # 提取每个chunk的消息内容
            chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
            log_llm_stream(chunk_message)  # 记录流式消息
            collected_messages.append(chunk_message)
            if chunk.usage:
                # 火山方舟的流式调用会在最后一个chunk返回usage信息
                usage = chunk.usage

        log_llm_stream("\n")
        full_reply_content = "".join(collected_messages)  # 合并所有的消息内容
        self._update_costs(usage, chunk.model)  # 更新成本信息
        return full_reply_content  # 返回完整的回复内容

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        """处理非流式的完整回复"""
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout))  # 构建请求参数
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)  # 获取完整的回复
        self._update_costs(rsp.usage, rsp.model)  # 更新成本信息
        return rsp  # 返回完整的回复对象
