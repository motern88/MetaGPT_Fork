# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : openai.py
@Modified By: mashenquan, 2023/11/21. Fix bug: ReadTimeout.
@Modified By: mashenquan, 2023/12/1. Fix bug: Unclosed connection caused by openai 0.x.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Union

from openai import APIConnectionError, AsyncOpenAI, AsyncStream
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream, logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.constant import GENERAL_FUNCTION_SCHEMA
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.utils.common import CodeParser, decode_image, log_and_reraise
from metagpt.utils.cost_manager import CostManager
from metagpt.utils.exceptions import handle_exception
from metagpt.utils.token_counter import (
    count_message_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)


@register_provider(
    [
        LLMType.OPENAI,
        LLMType.FIREWORKS,
        LLMType.OPEN_LLM,
        LLMType.MOONSHOT,
        LLMType.MISTRAL,
        LLMType.YI,
        LLMType.OPEN_ROUTER,
        LLMType.DEEPSEEK,
        LLMType.SILICONFLOW,
        LLMType.OPENROUTER,
    ]
)
class OpenAILLM(BaseLLM):
    """查看 https://platform.openai.com/examples 以获取更多示例"""

    def __init__(self, config: LLMConfig):
        self.config = config  # 初始化配置
        self._init_client()  # 初始化客户端
        self.auto_max_tokens = False  # 不自动设置最大 token 数
        self.cost_manager: Optional[CostManager] = None  # 成本管理器

    def _init_client(self):
        """https://github.com/openai/openai-python#async-usage"""
        self.model = self.config.model  # 使用的模型，用于计算使用量和请求参数
        self.pricing_plan = self.config.pricing_plan or self.model  # 定价计划
        kwargs = self._make_client_kwargs()  # 创建请求参数
        self.aclient = AsyncOpenAI(**kwargs)  # 初始化异步 OpenAI 客户端

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key, "base_url": self.config.base_url}  # API 密钥和基础 URL

        # 如果需要使用代理，OpenAI v1 需要 http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _get_proxy_params(self) -> dict:
        """获取代理参数"""
        params = {}
        if self.config.proxy:
            params = {"proxy": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> str:
        """流式处理消息并返回生成的文本"""
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout)), stream=True
        )
        usage = None
        collected_messages = []
        collected_reasoning_messages = []
        has_finished = False
        async for chunk in response:
            if not chunk.choices:
                continue

            choice0 = chunk.choices[0]
            choice_delta = choice0.delta
            if hasattr(choice_delta, "reasoning_content") and choice_delta.reasoning_content:
                collected_reasoning_messages.append(choice_delta.reasoning_content)  # 深度推理信息
                continue
            chunk_message = choice_delta.content or ""  # 提取消息内容
            finish_reason = choice0.finish_reason if hasattr(choice0, "finish_reason") else None
            log_llm_stream(chunk_message)  # 打印流式消息
            collected_messages.append(chunk_message)
            chunk_has_usage = hasattr(chunk, "usage") and chunk.usage
            if has_finished:
                # 对于 OneAPI，有些服务在 finish_reason 后还会有 usage 数据
                if chunk_has_usage:
                    usage = CompletionUsage(**chunk.usage) if isinstance(chunk.usage, dict) else chunk.usage
            if finish_reason:
                if chunk_has_usage:
                    # 一些服务的 usage 在 chunk 中，如 Fireworks
                    usage = CompletionUsage(**chunk.usage) if isinstance(chunk.usage, dict) else chunk.usage
                elif hasattr(choice0, "usage"):
                    # 一些服务的 usage 在 chunk.choices[0] 中，如 Moonshot
                    usage = CompletionUsage(**choice0.usage)
                has_finished = True

        log_llm_stream("\n")
        full_reply_content = "".join(collected_messages)
        if collected_reasoning_messages:
            self.reasoning_content = "".join(collected_reasoning_messages)
        if not usage:
            # 如果没有获取到 usage，使用默认的计算方式
            usage = self._calc_usage(messages, full_reply_content)

        self._update_costs(usage)  # 更新成本信息
        return full_reply_content

    def _cons_kwargs(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, **extra_kwargs) -> dict:
        """生成请求所需的参数"""
        kwargs = {
            "messages": messages,
            "max_tokens": self._get_max_tokens(messages),
            "temperature": self.config.temperature,
            "model": self.model,
            "timeout": self.get_timeout(timeout),
        }
        if "o1-" in self.model:
            # 兼容 openai o1 系列模型
            kwargs["temperature"] = 1
            kwargs.pop("max_tokens")
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        """获取普通的单次完成消息"""
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout))
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage)  # 更新成本信息
        return rsp

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        """异步获取完成消息"""
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(APIConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT) -> str:
        """获取文本回复，支持流式输出"""
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout)

        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout))
        return self.get_choice_text(rsp)

    async def _achat_completion_function(
        self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT, **chat_configs
    ) -> ChatCompletion:
        """执行带有函数调用的聊天请求"""
        messages = self.format_msg(messages)
        kwargs = self._cons_kwargs(messages=messages, timeout=self.get_timeout(timeout), **chat_configs)
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage)  # 更新成本信息
        return rsp

    async def aask_code(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT, **kwargs) -> dict:
        """使用工具请求生成代码"""
        if "tools" not in kwargs:
            configs = {"tools": [{"type": "function", "function": GENERAL_FUNCTION_SCHEMA}]}
            kwargs.update(configs)
        rsp = await self._achat_completion_function(messages, **kwargs)
        return self.get_choice_function_arguments(rsp)

    def _parse_arguments(self, arguments: str) -> dict:
        """解析 OpenAI 函数调用中的参数"""
        if "language" not in arguments and "code" not in arguments:
            logger.warning(f"未找到 `code` 或 `language`，假设这是纯代码:\n {arguments}\n. ")
            return {"language": "python", "code": arguments}

        # 匹配 language 参数
        language_pattern = re.compile(r'[\"\']?language[\"\']?\s*:\s*["\']([^"\']+?)["\']', re.DOTALL)
        language_match = language_pattern.search(arguments)
        language_value = language_match.group(1) if language_match else "python"

        # 匹配 code 参数
        code_pattern = r'(["\'`]{3}|["\'`])([\s\S]*?)\1'
        try:
            code_value = re.findall(code_pattern, arguments)[-1][-1]
        except Exception as e:
            logger.error(f"{e}, 当 re.findall({code_pattern}, {arguments}) 时发生错误")
            code_value = None

        if code_value is None:
            raise ValueError(f"解析代码时出错: {arguments}")

        # 只有 code 的情况
        return {"language": language_value, "code": code_value}

    def get_choice_function_arguments(self, rsp: ChatCompletion) -> dict:
        """获取函数调用中第一个函数参数

        :param dict rsp: 与 self.get_choice_function 相同
        :return dict: 返回第一个函数参数，例如 {'language': 'python', 'code': "print('Hello, World!')"}
        """
        message = rsp.choices[0].message
        if (
                message.tool_calls is not None
                and message.tool_calls[0].function is not None
                and message.tool_calls[0].function.arguments is not None
        ):
            # 如果返回的是代码
            try:
                return json.loads(message.tool_calls[0].function.arguments, strict=False)
            except json.decoder.JSONDecodeError as e:
                error_msg = (
                    f"在解析时遇到 JSONDecodeError：\n{'--' * 40} \n{message.tool_calls[0].function.arguments}, {str(e)}"
                )
                logger.error(error_msg)
                return self._parse_arguments(message.tool_calls[0].function.arguments)
        elif message.tool_calls is None and message.content is not None:
            # 如果返回的是代码（处理 OpenAI tools_call 响应 bug）
            # 如果响应内容是 `code`，但它出现在 content 中，而不是 arguments 中
            code_formats = "```"
            if message.content.startswith(code_formats) and message.content.endswith(code_formats):
                code = CodeParser.parse_code(text=message.content)
                return {"language": "python", "code": code}
            # 如果响应是文本
            return {"language": "markdown", "code": self.get_choice_text(rsp)}
        else:
            logger.error(f"解析失败：\n {rsp}\n")
            raise Exception(f"解析失败：\n {rsp}\n")

    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """获取第一个选择的文本内容"""
        return rsp.choices[0].message.content if rsp.choices else ""

    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        """计算请求的使用量"""
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if not self.config.calc_usage:
            return usage

        try:
            usage.prompt_tokens = count_message_tokens(messages, self.pricing_plan)
            usage.completion_tokens = count_output_tokens(rsp, self.pricing_plan)
        except Exception as e:
            logger.warning(f"使用量计算失败: {e}")

        return usage

    def _get_max_tokens(self, messages: list[dict]):
        """获取最大 token 数量"""
        if not self.auto_max_tokens:
            return self.config.max_token
        # FIXME: 参考文档链接：https://community.openai.com/t/why-is-gpt-3-5-turbo-1106-max-tokens-limited-to-4096/494973/3
        return min(get_max_completion_tokens(messages, self.model, self.config.max_token), 4096)

    @handle_exception
    async def amoderation(self, content: Union[str, list[str]]):
        """内容审核"""
        return await self.aclient.moderations.create(input=content)

    async def atext_to_speech(self, **kwargs):
        """文本转语音"""
        return await self.aclient.audio.speech.create(**kwargs)

    async def aspeech_to_text(self, **kwargs):
        """语音转文本"""
        return await self.aclient.audio.transcriptions.create(**kwargs)

    async def gen_image(
            self,
            prompt: str,
            size: str = "1024x1024",
            quality: str = "standard",
            model: str = None,
            resp_format: str = "url",
    ) -> list["Image"]:
        """生成图像"""
        assert resp_format in ["url", "b64_json"]
        if not model:
            model = self.model
        res = await self.aclient.images.generate(
            model=model, prompt=prompt, size=size, quality=quality, n=1, response_format=resp_format
        )
        imgs = []
        for item in res.data:
            img_url_or_b64 = item.url if resp_format == "url" else item.b64_json
            imgs.append(decode_image(img_url_or_b64))
        return imgs

    def count_tokens(self, messages: list[dict]) -> int:
        """计算 token 数量"""
        try:
            return count_message_tokens(messages, self.config.model)
        except:
            return super().count_tokens(messages)
