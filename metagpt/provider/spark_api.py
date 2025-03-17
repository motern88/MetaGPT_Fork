#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sparkai.core.messages import _convert_to_message, convert_to_messages
from sparkai.core.messages.ai import AIMessage
from sparkai.core.messages.base import BaseMessage
from sparkai.core.messages.human import HumanMessage
from sparkai.core.messages.system import SystemMessage
from sparkai.core.outputs.llm_result import LLMResult
from sparkai.llm.llm import ChatSparkLLM

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.utils.common import any_to_str
from metagpt.utils.cost_manager import CostManager
from metagpt.utils.token_counter import SPARK_TOKENS


@register_provider(LLMType.SPARK)
class SparkLLM(BaseLLM):
    """
    用于讯飞星火大模型系列
    参考：https://github.com/iflytek/spark-ai-python"""

    def __init__(self, config: LLMConfig):
        self.config = config  # 初始化配置
        self.cost_manager = CostManager(token_costs=SPARK_TOKENS)  # 成本管理器
        self.model = self.config.domain  # 模型名称
        self._init_client()  # 初始化客户端

    def _init_client(self):
        """初始化星火大模型客户端"""
        self.client = ChatSparkLLM(
            spark_api_url=self.config.base_url,  # API URL
            spark_app_id=self.config.app_id,  # 应用ID
            spark_api_key=self.config.api_key,  # API Key
            spark_api_secret=self.config.api_secret,  # API Secret
            spark_llm_domain=self.config.domain,  # 模型域
            streaming=True,  # 是否启用流式传输
        )

    def _system_msg(self, msg: str) -> SystemMessage:
        """将消息转换为系统消息"""
        return _convert_to_message(msg)

    def _user_msg(self, msg: str, **kwargs) -> HumanMessage:
        """将消息转换为用户消息"""
        return _convert_to_message(msg)

    def _assistant_msg(self, msg: str) -> AIMessage:
        """将消息转换为助手消息"""
        return _convert_to_message(msg)

    def get_choice_text(self, rsp: LLMResult) -> str:
        """从响应中提取生成的文本"""
        return rsp.generations[0][0].text

    def get_usage(self, response: LLMResult):
        """从响应中提取token使用情况"""
        message = response.generations[0][0].message
        if hasattr(message, "additional_kwargs"):
            return message.additional_kwargs.get("token_usage", {})
        else:
            return {}

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """异步完成对话"""
        response = await self.acreate(messages, stream=False)
        usage = self.get_usage(response)  # 获取token使用情况
        self._update_costs(usage)  # 更新成本
        return response

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """异步完成对话的封装方法"""
        return await self._achat_completion(messages, timeout)

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """异步流式完成对话"""
        response = await self.acreate(messages, stream=True)
        collected_content = []  # 收集流式响应的内容
        usage = {}  # 初始化token使用情况
        async for chunk in response:
            collected_content.append(chunk.content)  # 收集内容
            log_llm_stream(chunk.content)  # 记录流式内容
            if hasattr(chunk, "additional_kwargs"):
                usage = chunk.additional_kwargs.get("token_usage", {})  # 更新token使用情况

        log_llm_stream("\n")
        self._update_costs(usage)  # 更新成本
        full_content = "".join(collected_content)  # 合并流式内容
        return full_content

    def _extract_assistant_rsp(self, context: list[BaseMessage]) -> str:
        """从上下文中提取助手的响应"""
        return "\n".join([i.content for i in context if "AIMessage" in any_to_str(i)])

    async def acreate(self, messages: list[dict], stream: bool = True):
        """创建异步对话"""
        messages = convert_to_messages(messages)  # 转换消息格式
        if stream:
            return self.client.astream(messages)  # 流式响应
        else:
            return await self.client.agenerate([messages])  # 非流式响应
