#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

import json

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.general_api_requestor import GeneralAPIRequestor, OpenAIResponse
from metagpt.provider.llm_provider_registry import register_provider


@register_provider([LLMType.OPENROUTER_REASONING])
class OpenrouterReasoningLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        """初始化函数，设置客户端和配置信息"""
        self.client = GeneralAPIRequestor(base_url=config.base_url)  # 创建API请求客户端
        self.config = config  # 配置对象
        self.model = self.config.model  # 模型名称
        self.http_method = "post"  # HTTP请求方法
        self.base_url = "https://openrouter.ai/api/v1"  # 基本API URL
        self.url_suffix = "/chat/completions"  # API端点
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}  # 请求头部，包含API密钥

    def decode(self, response: OpenAIResponse) -> dict:
        """解码响应内容"""
        return json.loads(response.data.decode("utf-8"))  # 将响应数据解码为字典

    def _const_kwargs(
        self, messages: list[dict], stream: bool = False, timeout=USE_CONFIG_TIMEOUT, **extra_kwargs
    ) -> dict:
        """构建请求参数"""
        kwargs = {
            "messages": messages,  # 消息内容
            "include_reasoning": True,  # 是否包括推理内容
            "max_tokens": self.config.max_token,  # 最大token数
            "temperature": self.config.temperature,  # 温度参数
            "model": self.model,  # 模型名称
            "stream": stream,  # 是否启用流式响应
        }
        return kwargs

    def get_choice_text(self, rsp: dict) -> str:
        """从响应中获取文本内容"""
        if "reasoning" in rsp["choices"][0]["message"]:
            self.reasoning_content = rsp["choices"][0]["message"]["reasoning"]  # 获取推理内容
        return rsp["choices"][0]["message"]["content"]  # 返回消息内容

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> dict:
        """异步发送完成请求并返回响应"""
        payload = self._const_kwargs(messages)  # 构建请求参数
        resp, _, _ = await self.client.arequest(
            url=self.url_suffix, method=self.http_method, params=payload, headers=self.headers  # 发起请求
        )
        resp = resp.decode_asjson()  # 解码响应数据
        self._update_costs(resp["usage"], model=self.model)  # 更新费用
        return resp

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        """异步完成请求，返回响应"""
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))  # 调用异步完成请求方法

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """异步流式完成请求并返回流式响应"""
        self.headers["Content-Type"] = "text/event-stream"  # 更新请求头，以支持流式响应
        payload = self._const_kwargs(messages, stream=True)  # 构建流式请求参数
        resp, _, _ = await self.client.arequest(
            url=self.url_suffix, method=self.http_method, params=payload, headers=self.headers, stream=True  # 发起流式请求
        )
        collected_content = []  # 收集内容的列表
        collected_reasoning_content = []  # 收集推理内容的列表
        usage = {}  # 使用信息

        # 处理流式响应数据
        async for chunk in resp:
            chunk = chunk.decode_asjson()  # 解码每个块
            if not chunk:
                continue
            delta = chunk["choices"][0]["delta"]  # 获取每个块中的变化
            if "reasoning" in delta and delta["reasoning"]:
                collected_reasoning_content.append(delta["reasoning"])  # 收集推理内容
            elif delta["content"]:
                collected_content.append(delta["content"])  # 收集文本内容
                log_llm_stream(delta["content"])  # 记录流内容

            usage = chunk.get("usage")  # 获取使用信息

        log_llm_stream("\n")  # 记录流结束
        self._update_costs(usage, model=self.model)  # 更新费用
        full_content = "".join(collected_content)  # 拼接最终内容
        if collected_reasoning_content:
            self.reasoning_content = "".join(collected_reasoning_content)  # 拼接推理内容
        return full_content  # 返回拼接后的完整内容
