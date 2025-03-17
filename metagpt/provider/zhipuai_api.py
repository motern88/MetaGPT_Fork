#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : zhipuai LLM from https://open.bigmodel.cn/dev/api#sdk

from enum import Enum
from typing import Optional

from zhipuai.types.chat.chat_completion import Completion

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.provider.zhipuai.zhipu_model_api import ZhiPuModelAPI
from metagpt.utils.cost_manager import CostManager


class ZhiPuEvent(Enum):
    """智谱AI事件枚举"""
    ADD = "add"  # 添加事件
    ERROR = "error"  # 错误事件
    INTERRUPTED = "interrupted"  # 中断事件
    FINISH = "finish"  # 完成事件


@register_provider(LLMType.ZHIPUAI)
class ZhiPuAILLM(BaseLLM):
    """
    智谱AI大模型提供者
    参考：`https://open.bigmodel.cn/dev/api#chatglm_turbo`
    目前支持 glm-3-turbo、glm-4，以及系统提示。
    """

    def __init__(self, config: LLMConfig):
        self.config = config  # 初始化配置
        self.__init_zhipuai()  # 初始化智谱AI
        self.cost_manager: Optional[CostManager] = None  # 成本管理器

    def __init_zhipuai(self):
        """初始化智谱AI客户端"""
        assert self.config.api_key  # 确保API Key存在
        self.api_key = self.config.api_key  # API Key
        self.model = self.config.model  # 模型名称，目前支持 glm-3-turbo、glm-4
        self.pricing_plan = self.config.pricing_plan or self.model  # 定价计划
        self.llm = ZhiPuModelAPI(api_key=self.api_key)  # 初始化智谱AI模型API

    def _const_kwargs(self, messages: list[dict], stream: bool = False) -> dict:
        """构造请求参数"""
        max_tokens = self.config.max_token if self.config.max_token > 0 else 1024  # 最大token数
        temperature = self.config.temperature if self.config.temperature > 0.0 else 0.3  # 温度参数
        kwargs = {
            "model": self.model,  # 模型名称
            "max_tokens": max_tokens,  # 最大token数
            "messages": messages,  # 消息列表
            "stream": stream,  # 是否启用流式传输
            "temperature": temperature,  # 温度参数
        }
        return kwargs

    def completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        """同步完成对话"""
        resp: Completion = self.llm.chat.completions.create(**self._const_kwargs(messages))  # 创建对话
        usage = resp.usage.model_dump()  # 获取token使用情况
        self._update_costs(usage)  # 更新成本
        return resp.model_dump()  # 返回响应

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        """异步完成对话"""
        resp = await self.llm.acreate(**self._const_kwargs(messages))  # 异步创建对话
        usage = resp.get("usage", {})  # 获取token使用情况
        self._update_costs(usage)  # 更新成本
        return resp  # 返回响应

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        """异步完成对话的封装方法"""
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))  # 调用异步完成对话

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        """异步流式完成对话"""
        response = await self.llm.acreate_stream(**self._const_kwargs(messages, stream=True))  # 异步流式创建对话
        collected_content = []  # 收集流式响应的内容
        usage = {}  # 初始化token使用情况
        async for chunk in response.stream():  # 遍历流式响应
            finish_reason = chunk.get("choices")[0].get("finish_reason")  # 获取完成原因
            if finish_reason == "stop":  # 如果完成原因为停止
                usage = chunk.get("usage", {})  # 获取token使用情况
            else:
                content = self.get_choice_delta_text(chunk)  # 获取内容
                collected_content.append(content)  # 收集内容
                log_llm_stream(content)  # 记录流式内容

        log_llm_stream("\n")  # 记录换行
        self._update_costs(usage)  # 更新成本
        full_content = "".join(collected_content)  # 合并流式内容
        return full_content  # 返回完整内容
