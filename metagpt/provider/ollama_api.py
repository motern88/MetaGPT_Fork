#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : self-host open llm model with ollama which isn't openai-api-compatible

import json
from enum import Enum, auto
from typing import AsyncGenerator, Optional, Tuple

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.general_api_requestor import GeneralAPIRequestor, OpenAIResponse
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.utils.cost_manager import TokenCostManager


class OllamaMessageAPI(Enum):
    """Ollama API 消息类型的枚举类"""
    CHAT = auto()  # 聊天消息
    GENERATE = auto()  # 生成消息
    EMBED = auto()  # 嵌入消息
    EMBEDDINGS = auto()  # 嵌入向量消息


class OllamaMessageBase:
    """Ollama 消息基类，定义了消息相关的通用操作"""

    api_type = OllamaMessageAPI.CHAT  # 默认消息类型为聊天

    def __init__(self, model: str, **additional_kwargs) -> None:
        self.model = model  # 模型名称
        self.additional_kwargs = additional_kwargs  # 其他附加参数
        self._image_b64_rms = len("data:image/jpeg;base64,")  # 用于处理图像的 base64 数据前缀

    @property
    def api_suffix(self) -> str:
        """获取 API 的后缀路径"""
        raise NotImplementedError

    def apply(self, messages: list[dict]) -> dict:
        """根据消息生成 API 请求数据"""
        raise NotImplementedError

    def decode(self, response: OpenAIResponse) -> dict:
        """解析 API 响应"""
        return json.loads(response.data.decode("utf-8"))

    def get_choice(self, to_choice_dict: dict) -> str:
        """从 API 响应中提取选择内容"""
        raise NotImplementedError

    def _parse_input_msg(self, msg: dict) -> Tuple[Optional[str], Optional[str]]:
        """解析输入消息，返回文本和图像信息"""
        if "type" in msg:
            tpe = msg["type"]
            if tpe == "text":
                return msg["text"], None  # 处理文本消息
            elif tpe == "image_url":
                return None, msg["image_url"]["url"][self._image_b64_rms :]  # 处理图像消息
            else:
                raise ValueError("不支持的消息类型")
        else:
            raise ValueError("消息格式错误")


class OllamaMessageMeta(type):
    """Ollama 消息元类，用于注册和管理不同类型的消息类"""

    registed_message = {}  # 注册的消息类

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # 遍历基类，确保每个子类都注册到 `registed_message` 中
        for base in bases:
            if issubclass(base, OllamaMessageBase):
                api_type = attrs["api_type"]
                assert api_type not in OllamaMessageMeta.registed_message, "api_type 已经存在"
                assert isinstance(api_type, OllamaMessageAPI), "api_type 不支持"
                OllamaMessageMeta.registed_message[api_type] = cls

    @classmethod
    def get_message(cls, input_type: OllamaMessageAPI) -> type[OllamaMessageBase]:
        """根据消息类型获取对应的消息类"""
        return cls.registed_message[input_type]


class OllamaMessageChat(OllamaMessageBase, metaclass=OllamaMessageMeta):
    """聊天消息类，继承自 OllamaMessageBase"""

    api_type = OllamaMessageAPI.CHAT  # 设置消息类型为 CHAT

    @property
    def api_suffix(self) -> str:
        """获取聊天 API 的后缀路径"""
        return "/chat"

    def apply(self, messages: list[dict]) -> dict:
        """根据输入消息生成请求数据"""
        content = messages[0]["content"]
        prompts = []
        images = []
        if isinstance(content, list):  # 如果是消息列表
            for msg in content:
                prompt, image = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
                if image:
                    images.append(image)
        else:
            prompts.append(content)
        messes = []
        for prompt in prompts:
            if len(images) > 0:
                messes.append({"role": "user", "content": prompt, "images": images})
            else:
                messes.append({"role": "user", "content": prompt})
        sends = {"model": self.model, "messages": messes}
        sends.update(self.additional_kwargs)  # 添加附加参数
        return sends

    def get_choice(self, to_choice_dict: dict) -> str:
        """从响应字典中提取用户选择"""
        message = to_choice_dict["message"]
        if message["role"] == "assistant":
            return message["content"]
        else:
            raise ValueError("响应消息的角色不正确")


class OllamaMessageGenerate(OllamaMessageChat, metaclass=OllamaMessageMeta):
    """生成消息类，继承自 OllamaMessageChat"""

    api_type = OllamaMessageAPI.GENERATE  # 设置消息类型为 GENERATE

    @property
    def api_suffix(self) -> str:
        """获取生成 API 的后缀路径"""
        return "/generate"

    def apply(self, messages: list[dict]) -> dict:
        """根据输入消息生成请求数据"""
        content = messages[0]["content"]
        prompts = []
        images = []
        if isinstance(content, list):  # 如果是消息列表
            for msg in content:
                prompt, image = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
                if image:
                    images.append(image)
        else:
            prompts.append(content)
        if len(images) > 0:
            sends = {"model": self.model, "prompt": "\n".join(prompts), "images": images}
        else:
            sends = {"model": self.model, "prompt": "\n".join(prompts)}
        sends.update(self.additional_kwargs)  # 添加附加参数
        return sends

    def get_choice(self, to_choice_dict: dict) -> str:
        """从响应字典中提取生成结果"""
        return to_choice_dict["response"]


class OllamaMessageEmbeddings(OllamaMessageBase, metaclass=OllamaMessageMeta):
    """嵌入消息类，继承自 OllamaMessageBase"""

    api_type = OllamaMessageAPI.EMBEDDINGS  # 设置消息类型为 EMBEDDINGS

    @property
    def api_suffix(self) -> str:
        """获取嵌入 API 的后缀路径"""
        return "/embeddings"

    def apply(self, messages: list[dict]) -> dict:
        """根据输入消息生成嵌入请求数据"""
        content = messages[0]["content"]
        prompts = []  # 嵌入消息不支持图像
        if isinstance(content, list):  # 如果是消息列表
            for msg in content:
                prompt, _ = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
        else:
            prompts.append(content)
        sends = {"model": self.model, "prompt": "\n".join(prompts)}
        sends.update(self.additional_kwargs)  # 添加附加参数
        return sends


class OllamaMessageEmbed(OllamaMessageEmbeddings, metaclass=OllamaMessageMeta):
    """嵌入消息类（另一个形式），继承自 OllamaMessageEmbeddings"""

    api_type = OllamaMessageAPI.EMBED  # 设置消息类型为 EMBED

    @property
    def api_suffix(self) -> str:
        """获取嵌入 API 的后缀路径"""
        return "/embed"

    def apply(self, messages: list[dict]) -> dict:
        """根据输入消息生成嵌入请求数据"""
        content = messages[0]["content"]
        prompts = []  # 嵌入消息不支持图像
        if isinstance(content, list):  # 如果是消息列表
            for msg in content:
                prompt, _ = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
        else:
            prompts.append(content)
        sends = {"model": self.model, "input": prompts}
        sends.update(self.additional_kwargs)  # 添加附加参数
        return sends


@register_provider(LLMType.OLLAMA)
class OllamaLLM(BaseLLM):
    """
    参考 `https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-chat-completion`
    """

    def __init__(self, config: LLMConfig):
        self.client = GeneralAPIRequestor(base_url=config.base_url, key=config.api_key)
        self.config = config
        self.http_method = "post"  # HTTP方法，默认为post
        self.use_system_prompt = False  # 是否使用系统提示符
        self.cost_manager = TokenCostManager()  # 令牌费用管理
        self.__init_ollama(config)  # 初始化Ollama配置

    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        # 返回当前使用的Ollama API类型
        return OllamaMessageAPI.CHAT

    @property
    def _llama_api_kwargs(self) -> dict:
        # 返回Ollama API的额外配置项
        return {"options": {"temperature": 0.3}, "stream": self.config.stream}

    def __init_ollama(self, config: LLMConfig):
        assert config.base_url, "ollama base url is required!"  # 确保配置中有base_url
        self.model = config.model  # 设置模型
        self.pricing_plan = self.model  # 设置定价计划
        ollama_message = OllamaMessageMeta.get_message(self._llama_api_inuse)  # 获取对应API类型的消息处理类
        self.ollama_message = ollama_message(model=self.model, **self._llama_api_kwargs)  # 初始化消息处理

    def get_usage(self, resp: dict) -> dict:
        # 提取响应中的使用情况
        return {"prompt_tokens": resp.get("prompt_eval_count", 0), "completion_tokens": resp.get("eval_count", 0)}

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> dict:
        """
        异步完成聊天请求
        """
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.ollama_message.api_suffix,
            params=self.ollama_message.apply(messages=messages),
            request_timeout=self.get_timeout(timeout),
        )
        if isinstance(resp, AsyncGenerator):
            return await self._processing_openai_response_async_generator(resp)  # 处理流式响应
        elif isinstance(resp, OpenAIResponse):
            return self._processing_openai_response(resp)  # 处理非流式响应
        else:
            raise ValueError

    def get_choice_text(self, rsp):
        # 从响应中提取选择的文本
        return self.ollama_message.get_choice(rsp)

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        # 异步调用聊天完成接口
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """
        异步流式聊天请求
        """
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.ollama_message.api_suffix,
            params=self.ollama_message.apply(messages=messages),
            request_timeout=self.get_timeout(timeout),
            stream=True,
        )
        if isinstance(resp, AsyncGenerator):
            return await self._processing_openai_response_async_generator(resp)
        elif isinstance(resp, OpenAIResponse):
            return self._processing_openai_response(resp)
        else:
            raise ValueError

    def _processing_openai_response(self, openai_resp: OpenAIResponse):
        # 处理OpenAI响应
        resp = self.ollama_message.decode(openai_resp)
        usage = self.get_usage(resp)  # 获取使用情况
        self._update_costs(usage)  # 更新费用
        return resp

    async def _processing_openai_response_async_generator(self, ag_openai_resp: AsyncGenerator[OpenAIResponse, None]):
        # 异步处理OpenAI响应流
        collected_content = []
        usage = {}
        async for raw_chunk in ag_openai_resp:
            chunk = self.ollama_message.decode(raw_chunk)

            if not chunk.get("done", False):
                content = self.ollama_message.get_choice(chunk)
                collected_content.append(content)
                log_llm_stream(content)  # 记录流式数据
            else:
                # 流结束
                usage = self.get_usage(chunk)
        log_llm_stream("\n")

        self._update_costs(usage)  # 更新费用
        full_content = "".join(collected_content)
        return full_content


@register_provider(LLMType.OLLAMA_GENERATE)
class OllamaGenerate(OllamaLLM):
    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.GENERATE  # 使用生成API

    @property
    def _llama_api_kwargs(self) -> dict:
        return {"options": {"temperature": 0.3}, "stream": self.config.stream}  # 设置生成API的参数


@register_provider(LLMType.OLLAMA_EMBEDDINGS)
class OllamaEmbeddings(OllamaLLM):
    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.EMBEDDINGS  # 使用嵌入API

    @property
    def _llama_api_kwargs(self) -> dict:
        return {"options": {"temperature": 0.3}}  # 设置嵌入API的参数

    @property
    def _llama_embedding_key(self) -> str:
        return "embedding"  # 嵌入结果的键名

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> dict:
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.ollama_message.api_suffix,
            params=self.ollama_message.apply(messages=messages),
            request_timeout=self.get_timeout(timeout),
        )
        return self.ollama_message.decode(resp)[self._llama_embedding_key]  # 返回嵌入结果

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    def get_choice_text(self, rsp):
        return rsp  # 返回响应中的嵌入结果


@register_provider(LLMType.OLLAMA_EMBED)
class OllamaEmbed(OllamaEmbeddings):
    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.EMBED  # 使用嵌入API

    @property
    def _llama_embedding_key(self) -> str:
        return "embeddings"  # 嵌入结果的键名
