import asyncio
import json
import os
from functools import partial
from typing import List, Literal

import boto3
from botocore.eventstream import EventStream

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream, logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.bedrock.bedrock_provider import get_provider
from metagpt.provider.bedrock.utils import NOT_SUPPORT_STREAM_MODELS, get_max_tokens
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.utils.cost_manager import CostManager
from metagpt.utils.token_counter import BEDROCK_TOKEN_COSTS


@register_provider([LLMType.BEDROCK])
class BedrockLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.__client = self.__init_client("bedrock-runtime")  # 初始化 bedrock-runtime 客户端
        self.__provider = get_provider(
            self.config.model, reasoning=self.config.reasoning, reasoning_max_token=self.config.reasoning_max_token
        )  # 获取模型提供者
        self.cost_manager = CostManager(token_costs=BEDROCK_TOKEN_COSTS)  # 初始化成本管理器
        if self.config.model in NOT_SUPPORT_STREAM_MODELS:
            logger.warning(f"model {self.config.model} doesn't support streaming output!")  # 如果模型不支持流式输出，发出警告

    def __init_client(self, service_name: Literal["bedrock-runtime", "bedrock"]):
        """初始化 boto3 客户端"""
        # 通过 AWS 控制台获取 access key 和 secret key
        self.__credential_kwargs = {
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", self.config.secret_key),
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", self.config.access_key),
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN", self.config.session_token),
            "region_name": os.environ.get("AWS_DEFAULT_REGION", self.config.region_name),
        }
        session = boto3.Session(**self.__credential_kwargs)  # 创建 boto3 会话
        client = session.client(service_name, region_name=self.__credential_kwargs["region_name"])  # 创建服务客户端
        return client

    @property
    def client(self):
        return self.__client  # 返回客户端

    @property
    def provider(self):
        return self.__provider  # 返回模型提供者

    def list_models(self):
        """列出所有可用的文本生成模型"""
        client = self.__init_client("bedrock")  # 初始化客户端
        # 只输出文本生成模型
        response = client.list_foundation_models(byOutputModality="TEXT")
        summaries = [
            f'{summary["modelId"]:50} Support Streaming:{summary["responseStreamingSupported"]}'
            for summary in response["modelSummaries"]
        ]
        logger.info("\n" + "\n".join(summaries))  # 输出模型信息

    async def invoke_model(self, request_body: str) -> dict:
        loop = asyncio.get_running_loop()  # 获取事件循环
        response = await loop.run_in_executor(
            None, partial(self.client.invoke_model, modelId=self.config.model, body=request_body)
        )  # 异步调用模型
        usage = self._get_usage(response)  # 获取使用情况
        self._update_costs(usage, self.config.model)  # 更新成本
        response_body = self._get_response_body(response)  # 获取响应体
        return response_body

    async def invoke_model_with_response_stream(self, request_body: str) -> EventStream:
        loop = asyncio.get_running_loop()  # 获取事件循环
        response = await loop.run_in_executor(
            None, partial(self.client.invoke_model_with_response_stream, modelId=self.config.model, body=request_body)
        )  # 异步调用模型流式响应
        usage = self._get_usage(response)  # 获取使用情况
        self._update_costs(usage, self.config.model)  # 更新成本
        return response

    @property
    def _const_kwargs(self) -> dict:
        model_max_tokens = get_max_tokens(self.config.model)  # 获取模型的最大 token 数
        if self.config.max_token > model_max_tokens:
            max_tokens = model_max_tokens
        else:
            max_tokens = self.config.max_token  # 设置最大 token 数

        return {self.__provider.max_tokens_field_name: max_tokens, "temperature": self.config.temperature}

    def get_choice_text(self, rsp: dict) -> str:
        rsp = self.__provider.get_choice_text(rsp)  # 获取模型响应的文本
        if isinstance(rsp, dict):
            self.reasoning_content = rsp.get("reasoning_content")  # 获取推理内容
            rsp = rsp.get("content")  # 获取模型的主要内容
        return rsp

    async def acompletion(self, messages: list[dict]) -> dict:
        request_body = self.__provider.get_request_body(messages, self._const_kwargs)  # 获取请求体
        response_body = await self.invoke_model(request_body)  # 调用模型
        return response_body

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        return await self.acompletion(messages)  # 调用 acompletion 方法

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        if self.config.model in NOT_SUPPORT_STREAM_MODELS:
            rsp = await self.acompletion(messages)  # 如果模型不支持流式输出，则获取完整文本
            full_text = self.get_choice_text(rsp)  # 获取文本内容
            log_llm_stream(full_text)  # 记录流式输出
            return full_text

        request_body = self.__provider.get_request_body(messages, self._const_kwargs, stream=True)  # 获取流式请求体
        stream_response = await self.invoke_model_with_response_stream(request_body)  # 调用流式模型
        collected_content = await self._get_stream_response_body(stream_response)  # 获取流式响应内容
        log_llm_stream("\n")
        full_text = ("".join(collected_content)).lstrip()  # 合并流式输出
        return full_text

    def _get_response_body(self, response) -> dict:
        response_body = json.loads(response["body"].read())  # 解析响应体
        return response_body

    async def _get_stream_response_body(self, stream_response) -> List[str]:
        def collect_content() -> str:
            collected_content = []  # 存储内容
            collected_reasoning_content = []  # 存储推理内容
            for event in stream_response["body"]:  # 遍历流式响应
                reasoning, chunk_text = self.__provider.get_choice_text_from_stream(event)  # 获取推理和文本
                if reasoning:
                    collected_reasoning_content.append(chunk_text)  # 存储推理内容
                else:
                    collected_content.append(chunk_text)  # 存储其他内容
                    log_llm_stream(chunk_text)  # 记录流式输出
            if collected_reasoning_content:
                self.reasoning_content = "".join(collected_reasoning_content)  # 合并推理内容
            return collected_content

        loop = asyncio.get_running_loop()  # 获取事件循环
        return await loop.run_in_executor(None, collect_content)  # 异步收集流式响应内容

    def _get_usage(self, response) -> dict[str, int]:
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})  # 获取响应头
        prompt_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))  # 获取输入 token 数
        completion_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))  # 获取输出 token 数
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return usage  # 返回使用情况
