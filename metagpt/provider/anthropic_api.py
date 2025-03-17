#!/usr/bin/env python
# -*- coding: utf-8 -*-

from anthropic import AsyncAnthropic
from anthropic.types import Message, Usage

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider


@register_provider([LLMType.ANTHROPIC, LLMType.CLAUDE])
class AnthropicLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        """
        初始化 AnthropicLLM 实例。

        参数：
            config (LLMConfig): 用于配置 LLM 的配置对象，包括模型名称、API 密钥等信息。
        """
        self.config = config
        self.__init_anthropic()

    def __init_anthropic(self):
        """
        初始化与 Anthropic API 的连接，设置客户端。
        """
        self.model = self.config.model
        self.aclient: AsyncAnthropic = AsyncAnthropic(api_key=self.config.api_key, base_url=self.config.base_url)

    def _const_kwargs(self, messages: list[dict], stream: bool = False) -> dict:
        """
        构造调用 API 时需要的请求参数。

        参数：
            messages (list[dict]): 消息列表，每条消息包含角色和内容。
            stream (bool): 是否使用流式处理（默认为 False）。

        返回：
            dict: 构造的请求参数字典。
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config.max_token,
            "stream": stream,
        }
        # 如果使用系统提示（system prompt），则提取并传递给 API
        if self.use_system_prompt:
            if messages[0]["role"] == "system":
                kwargs["messages"] = messages[1:]
                kwargs["system"] = messages[0]["content"]  # 设置系统提示内容
        # 如果启用了推理（reasoning），添加相关参数
        if self.config.reasoning:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.config.reasoning_max_token}
        return kwargs

    def _update_costs(self, usage: Usage, model: str = None, local_calc_usage: bool = True):
        """
        更新费用信息。

        参数：
            usage (Usage): 费用信息，包括输入和输出的 token 数量。
            model (str): 模型名称（可选）。
            local_calc_usage (bool): 是否进行本地费用计算（默认为 True）。
        """
        usage = {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens}
        super()._update_costs(usage, model)

    def get_choice_text(self, resp: Message) -> str:
        """
        从模型响应中提取生成的文本。

        参数：
            resp (Message): 模型返回的消息对象。

        返回：
            str: 从响应中提取的生成文本。
        """
        if len(resp.content) > 1:
            self.reasoning_content = resp.content[0].thinking
            text = resp.content[1].text
        else:
            text = resp.content[0].text
        return text

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> Message:
        """
        异步调用模型进行文本生成，获取完整的响应。

        参数：
            messages (list[dict]): 消息列表，每条消息包含角色和内容。
            timeout (int): 请求的超时时间。

        返回：
            Message: 模型生成的消息对象。
        """
        resp: Message = await self.aclient.messages.create(**self._const_kwargs(messages))
        self._update_costs(resp.usage, self.model)
        return resp

    async def acompletion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> Message:
        """
        异步调用模型进行文本生成，获取最终结果。

        参数：
            messages (list[dict]): 消息列表，每条消息包含角色和内容。
            timeout (int): 请求的超时时间。

        返回：
            Message: 模型生成的消息对象。
        """
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """
        异步调用模型进行流式生成文本，逐步返回生成的内容。

        参数：
            messages (list[dict]): 消息列表，每条消息包含角色和内容。
            timeout (int): 请求的超时时间。

        返回：
            str: 完整的生成文本内容。
        """
        # 发起流式请求
        stream = await self.aclient.messages.create(**self._const_kwargs(messages, stream=True))
        collected_content = []
        collected_reasoning_content = []
        usage = Usage(input_tokens=0, output_tokens=0)

        # 逐步处理流式返回的数据
        async for event in stream:
            event_type = event.type
            if event_type == "message_start":
                usage.input_tokens = event.message.usage.input_tokens
                usage.output_tokens = event.message.usage.output_tokens
            elif event_type == "content_block_delta":
                delta_type = event.delta.type
                if delta_type == "thinking_delta":
                    collected_reasoning_content.append(event.delta.thinking)
                elif delta_type == "text_delta":
                    content = event.delta.text
                    log_llm_stream(content)
                    collected_content.append(content)
            elif event_type == "message_delta":
                usage.output_tokens = event.usage.output_tokens  # 更新最终的输出 token 数量

        log_llm_stream("\n")
        self._update_costs(usage)
        full_content = "".join(collected_content)

        # 如果有推理内容，记录下来
        if collected_reasoning_content:
            self.reasoning_content = "".join(collected_reasoning_content)

        return full_content
