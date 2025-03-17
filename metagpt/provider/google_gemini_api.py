#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : Google Gemini LLM from https://ai.google.dev/tutorials/python_quickstart
import json
import os
from dataclasses import asdict
from typing import List, Optional, Union

import google.generativeai as genai
from google.ai import generativelanguage as glm
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import content_types
from google.generativeai.types.generation_types import (
    AsyncGenerateContentResponse,
    BlockedPromptException,
    GenerateContentResponse,
    GenerationConfig,
)

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream, logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider


class GeminiGenerativeModel(GenerativeModel):
    """
    由于`https://github.com/google/generative-ai-python/pull/123`，继承了一个新类。
    如果修复了问题，默认将使用 GenerativeModel。
    """

    def count_tokens(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
        """
        计算给定内容的 token 数量。

        参数：
            contents (content_types.ContentsType): 输入的内容，可以是字符串或其他类型的内容。

        返回：
            glm.CountTokensResponse: 返回一个包含 token 数量的响应。
        """
        contents = content_types.to_contents(contents)
        return self._client.count_tokens(model=self.model_name, contents=contents)

    async def count_tokens_async(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
        """
        异步计算给定内容的 token 数量。

        参数：
            contents (content_types.ContentsType): 输入的内容，可以是字符串或其他类型的内容。

        返回：
            glm.CountTokensResponse: 返回一个包含 token 数量的响应。
        """
        contents = content_types.to_contents(contents)
        return await self._async_client.count_tokens(model=self.model_name, contents=contents)


@register_provider(LLMType.GEMINI)
class GeminiLLM(BaseLLM):
    """
    参考 `https://ai.google.dev/tutorials/python_quickstart` 进行实现。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 GeminiLLM 类。

        参数：
            config (LLMConfig): LLM 配置对象，包含模型信息、API 密钥等配置。
        """
        self.use_system_prompt = False  # Google Gemini API 不使用系统提示

        self.__init_gemini(config)  # 初始化 Gemini 配置
        self.config = config
        self.model = config.model
        self.pricing_plan = self.config.pricing_plan or self.model
        self.llm = GeminiGenerativeModel(model_name=self.model)

    def __init_gemini(self, config: LLMConfig):
        """
        初始化 Gemini 相关配置。

        参数：
            config (LLMConfig): LLM 配置对象，包含 API 密钥、代理等信息。
        """
        if config.proxy:
            logger.info(f"使用代理: {config.proxy}")
            os.environ["http_proxy"] = config.proxy
            os.environ["https_proxy"] = config.proxy
        genai.configure(api_key=config.api_key)

    def _user_msg(self, msg: str, images: Optional[Union[str, list[str]]] = None) -> dict[str, str]:
        """
        构建用户消息格式。

        参数：
            msg (str): 用户输入的消息。
            images (Optional[Union[str, list[str]]]): 可选的图片链接，默认值为 None。

        返回：
            dict[str, str]: 格式化后的用户消息字典。
        """
        return {"role": "user", "parts": [msg]}

    def _assistant_msg(self, msg: str) -> dict[str, str]:
        """
        构建助手（模型）回复消息格式。

        参数：
            msg (str): 模型的回复消息。

        返回：
            dict[str, str]: 格式化后的助手消息字典。
        """
        return {"role": "model", "parts": [msg]}

    def _system_msg(self, msg: str) -> dict[str, str]:
        """
        构建系统消息格式。

        参数：
            msg (str): 系统消息。

        返回：
            dict[str, str]: 格式化后的系统消息字典。
        """
        return {"role": "user", "parts": [msg]}

    def format_msg(self, messages: Union[str, "Message", list[dict], list["Message"], list[str]]) -> list[dict]:
        """
        将消息转换为字典格式。

        参数：
            messages (Union[str, "Message", list[dict], list["Message"], list[str]]): 输入的消息，可以是字符串、字典或消息对象的列表。

        返回：
            list[dict]: 格式化后的消息字典列表。
        """
        from metagpt.schema import Message

        if not isinstance(messages, list):
            messages = [messages]

        # 参考：https://ai.google.dev/tutorials/python_quickstart
        # 消息字典格式要求包含 `role` 和 `parts` 两个键。
        processed_messages = []
        for msg in messages:
            if isinstance(msg, str):
                processed_messages.append({"role": "user", "parts": [msg]})
            elif isinstance(msg, dict):
                assert set(msg.keys()) == set(["role", "parts"])
                processed_messages.append(msg)
            elif isinstance(msg, Message):
                processed_messages.append({"role": "user" if msg.role == "user" else "model", "parts": [msg.content]})
            else:
                raise ValueError(
                    f"只支持以下消息类型: str, Message, dict，但获取到的是 {type(messages).__name__}!"
                )
        return processed_messages

    def _const_kwargs(self, messages: list[dict], stream: bool = False) -> dict:
        """
        构建生成内容请求的常量参数。

        参数：
            messages (list[dict]): 消息内容的字典列表。
            stream (bool): 是否为流式响应。

        返回：
            dict: 请求参数字典。
        """
        kwargs = {"contents": messages, "generation_config": GenerationConfig(temperature=0.3), "stream": stream}
        return kwargs

    def get_choice_text(self, resp: GenerateContentResponse) -> str:
        """
        从响应中提取生成的文本。

        参数：
            resp (GenerateContentResponse): 生成内容的响应对象。

        返回：
            str: 生成的文本。
        """
        return resp.text

    def get_usage(self, messages: list[dict], resp_text: str) -> dict:
        """
        获取消息的 token 使用情况。

        参数：
            messages (list[dict]): 消息内容的字典列表。
            resp_text (str): 模型的回复文本。

        返回：
            dict: 包含 prompt_tokens 和 completion_tokens 的 token 使用情况字典。
        """
        req_text = messages[-1]["parts"][0] if messages else ""
        prompt_resp = self.llm.count_tokens(contents={"role": "user", "parts": [{"text": req_text}]})
        completion_resp = self.llm.count_tokens(contents={"role": "model", "parts": [{"text": resp_text}]})
        usage = {"prompt_tokens": prompt_resp.total_tokens, "completion_tokens": completion_resp.total_tokens}
        return usage

    async def aget_usage(self, messages: list[dict], resp_text: str) -> dict:
        """
        异步获取消息的 token 使用情况。

        参数：
            messages (list[dict]): 消息内容的字典列表。
            resp_text (str): 模型的回复文本。

        返回：
            dict: 包含 prompt_tokens 和 completion_tokens 的 token 使用情况字典。
        """
        req_text = messages[-1]["parts"][0] if messages else ""
        prompt_resp = await self.llm.count_tokens_async(contents={"role": "user", "parts": [{"text": req_text}]})
        completion_resp = await self.llm.count_tokens_async(contents={"role": "model", "parts": [{"text": resp_text}]})
        usage = {"prompt_tokens": prompt_resp.total_tokens, "completion_tokens": completion_resp.total_tokens}
        return usage

    def completion(self, messages: list[dict]) -> "GenerateContentResponse":
        """
        完成消息的生成。

        参数：
            messages (list[dict]): 消息内容的字典列表。

        返回：
            GenerateContentResponse: 生成的内容响应。
        """
        resp: GenerateContentResponse = self.llm.generate_content(**self._const_kwargs(messages))
        usage = self.get_usage(messages, resp.text)
        self._update_costs(usage)
        return resp

    async def _achat_completion(
        self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT
    ) -> "AsyncGenerateContentResponse":
        """
        异步完成消息的生成。

        参数：
            messages (list[dict]): 消息内容的字典列表。
            timeout (int): 请求超时时间。

        返回：
            AsyncGenerateContentResponse: 异步生成的内容响应。
        """
        resp: AsyncGenerateContentResponse = await self.llm.generate_content_async(**self._const_kwargs(messages))
        usage = await self.aget_usage(messages, resp.text)
        self._update_costs(usage)
        return resp

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        """
        异步完成消息的生成（接口方法）。

        参数：
            messages (list[dict]): 消息内容的字典列表。
            timeout (int): 请求超时时间。

        返回：
            dict: 生成的内容字典。
        """
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """
        异步流式生成消息。

        参数：
            messages (list[dict]): 消息内容的字典列表。
            timeout (int): 请求超时时间。

        返回：
            str: 完整生成的文本。
        """
        resp: AsyncGenerateContentResponse = await self.llm.generate_content_async(
            **self._const_kwargs(messages, stream=True)
        )
        collected_content = []
        async for chunk in resp:
            try:
                content = chunk.text
            except Exception as e:
                logger.warning(f"messages: {messages}\nerrors: {e}\n{BlockedPromptException(str(chunk))}")
                raise BlockedPromptException(str(chunk))
            log_llm_stream(content)
            collected_content.append(content)
        log_llm_stream("\n")

        full_content = "".join(collected_content)
        usage = await self.aget_usage(messages, full_content)
        self._update_costs(usage)
        return full_content

    def list_models(self) -> List:
        """
        获取所有可用模型的列表。

        返回：
            List: 模型信息的字典列表。
        """
        models = []
        for model in genai.list_models(page_size=100):
            models.append(asdict(model))
        logger.info(json.dumps(models))
        return models
