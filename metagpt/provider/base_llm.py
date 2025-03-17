#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:04
@Author  : alexanderwu
@File    : base_llm.py
@Desc    : mashenquan, 2023/8/22. + try catch
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional, Union

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from metagpt.configs.compress_msg_config import CompressType
from metagpt.configs.llm_config import LLMConfig
from metagpt.const import IMAGES, LLM_API_TIMEOUT, USE_CONFIG_TIMEOUT
from metagpt.logs import logger
from metagpt.provider.constant import MULTI_MODAL_MODELS
from metagpt.utils.common import log_and_reraise
from metagpt.utils.cost_manager import CostManager, Costs
from metagpt.utils.token_counter import TOKEN_MAX


class BaseLLM(ABC):
    """LLM API 抽象类，要求所有继承类提供一系列标准功能"""

    config: LLMConfig  # 配置类
    use_system_prompt: bool = True  # 是否使用系统提示
    system_prompt = "你是一个有帮助的助手。"  # 默认系统提示

    # OpenAI / Azure / 其他
    aclient: Optional[Union[AsyncOpenAI]] = None  # 客户端
    cost_manager: Optional[CostManager] = None  # 成本管理器
    model: Optional[str] = None  # 模型，已弃用
    pricing_plan: Optional[str] = None  # 定价计划

    _reasoning_content: Optional[str] = None  # 推理模式的内容

    @property
    def reasoning_content(self):
        """获取推理内容"""
        return self._reasoning_content

    @reasoning_content.setter
    def reasoning_content(self, value: str):
        """设置推理内容"""
        self._reasoning_content = value

    @abstractmethod
    def __init__(self, config: LLMConfig):
        """初始化方法"""
        pass

    def _user_msg(self, msg: str, images: Optional[Union[str, list[str]]] = None) -> dict[str, Union[str, dict]]:
        """构建用户消息"""
        if images and self.support_image_input():
            # 如果支持图像输入，发送带图像的消息
            return self._user_msg_with_imgs(msg, images)
        else:
            return {"role": "user", "content": msg}

    def _user_msg_with_imgs(self, msg: str, images: Optional[Union[str, list[str]]]):
        """
        处理带图像的用户消息
        images: 可以是http(s) URL或base64编码的图像
        """
        if isinstance(images, str):
            images = [images]  # 确保images是列表
        content = [{"type": "text", "text": msg}]  # 消息内容
        for image in images:
            # 如果是URL或base64
            url = image if image.startswith("http") else f"data:image/jpeg;base64,{image}"
            content.append({"type": "image_url", "image_url": {"url": url}})  # 添加图像信息
        return {"role": "user", "content": content}

    def _assistant_msg(self, msg: str) -> dict[str, str]:
        """构建助手消息"""
        return {"role": "assistant", "content": msg}

    def _system_msg(self, msg: str) -> dict[str, str]:
        """构建系统消息"""
        return {"role": "system", "content": msg}

    def support_image_input(self) -> bool:
        """检查是否支持图像输入"""
        return any([m in self.config.model for m in MULTI_MODAL_MODELS])

    def format_msg(self, messages: Union[str, "Message", list[dict], list["Message"], list[str]]) -> list[dict]:
        """将消息转换为list[dict]格式"""
        from metagpt.schema import Message

        if not isinstance(messages, list):
            messages = [messages]  # 如果只有单个消息，将其转换为列表

        processed_messages = []
        for msg in messages:
            if isinstance(msg, str):
                processed_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                assert set(msg.keys()) == set(["role", "content"])  # 验证消息格式
                processed_messages.append(msg)
            elif isinstance(msg, Message):
                images = msg.metadata.get(IMAGES)
                processed_msg = self._user_msg(msg=msg.content, images=images) if images else msg.to_dict()
                processed_messages.append(processed_msg)
            else:
                raise ValueError(
                    f"只支持的消息类型有: str, Message, dict, 当前类型为 {type(messages).__name__}!"
                )
        return processed_messages

    def _system_msgs(self, msgs: list[str]) -> list[dict[str, str]]:
        """构建多个系统消息"""
        return [self._system_msg(msg) for msg in msgs]

    def _default_system_msg(self):
        """返回默认的系统消息"""
        return self._system_msg(self.system_prompt)

    def _update_costs(self, usage: Union[dict, BaseModel], model: str = None, local_calc_usage: bool = True):
        """更新每个请求的token成本
        参数：
            model (str): 模型名称，或在某些场景下称为端点
            local_calc_usage (bool): 某些模型不计算使用情况，这会覆盖LLMConfig中的calc_usage设置
        """
        calc_usage = self.config.calc_usage and local_calc_usage  # 是否计算使用情况
        model = model or self.pricing_plan  # 如果没有传入model，使用定价计划
        model = model or self.model  # 如果没有定价计划，则使用模型名称
        usage = usage.model_dump() if isinstance(usage, BaseModel) else usage  # 获取使用情况
        if calc_usage and self.cost_manager and usage:
            try:
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                self.cost_manager.update_cost(prompt_tokens, completion_tokens, model)  # 更新成本
            except Exception as e:
                logger.error(f"{self.__class__.__name__} 更新成本失败！错误: {e}")

    def get_costs(self) -> Costs:
        """获取当前成本"""
        if not self.cost_manager:
            return Costs(0, 0, 0, 0)  # 如果没有成本管理器，返回默认值
        return self.cost_manager.get_costs()  # 获取成本

    def mask_base64_data(self, msg: dict) -> dict:
        """处理消息中的base64图像数据，替换为占位符，便于日志记录

        参数：
            msg (dict): 消息字典，格式为OpenAI的消息格式

        返回：
            dict: 处理后的消息字典，图像数据被替换为占位符
        """
        if not isinstance(msg, dict):
            return msg

        new_msg = msg.copy()  # 复制原始消息
        content = new_msg.get("content")  # 获取内容
        img_base64_prefix = "data:image/"

        if isinstance(content, list):
            # 处理多模态内容（例如gpt-4v格式）
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith(img_base64_prefix):
                        item = item.copy()
                        item["image_url"] = {"url": "<图像的base64数据已省略>"}
                new_content.append(item)
            new_msg["content"] = new_content
        elif isinstance(content, str) and img_base64_prefix in content:
            # 处理包含base64图像数据的纯文本消息
            new_msg["content"] = "<包含图像base64数据的消息已省略>"
        return new_msg

    async def aask(
        self,
        msg: Union[str, list[dict[str, str]]],
        system_msgs: Optional[list[str]] = None,
        format_msgs: Optional[list[dict[str, str]]] = None,
        images: Optional[Union[str, list[str]]] = None,
        timeout=USE_CONFIG_TIMEOUT,
        stream=None,
    ) -> str:
        # 如果有系统消息，调用 _system_msgs 方法生成系统消息
        if system_msgs:
            message = self._system_msgs(system_msgs)
        else:
            message = [self._default_system_msg()]  # 默认的系统消息
        # 如果不使用系统提示，则清空 message 列表
        if not self.use_system_prompt:
            message = []
        # 如果有格式化消息，扩展到 message 中
        if format_msgs:
            message.extend(format_msgs)
        # 根据 msg 的类型决定如何处理
        if isinstance(msg, str):
            message.append(self._user_msg(msg, images=images))  # 添加用户消息
        else:
            message.extend(msg)  # 如果 msg 是列表，则直接扩展到 message 中
        # 如果没有指定流，则使用配置中的流设置
        if stream is None:
            stream = self.config.stream

        # 用占位符替换图像数据，避免输出过长
        masked_message = [self.mask_base64_data(m) for m in message]
        logger.debug(masked_message)

        # 压缩消息
        compressed_message = self.compress_messages(message, compress_type=self.config.compress_type)
        # 调用异步方法获取回答
        rsp = await self.acompletion_text(compressed_message, stream=stream, timeout=self.get_timeout(timeout))
        return rsp

    def _extract_assistant_rsp(self, context):
        """从上下文中提取助手的回复"""
        return "\n".join([i["content"] for i in context if i["role"] == "assistant"])

    async def aask_batch(self, msgs: list, timeout=USE_CONFIG_TIMEOUT) -> str:
        """批量顺序提问"""
        context = []
        for msg in msgs:
            umsg = self._user_msg(msg)
            context.append(umsg)
            rsp_text = await self.acompletion_text(context, timeout=self.get_timeout(timeout))
            context.append(self._assistant_msg(rsp_text))
        return self._extract_assistant_rsp(context)

    async def aask_code(
        self, messages: Union[str, "Message", list[dict]], timeout=USE_CONFIG_TIMEOUT, **kwargs
    ) -> dict:
        """代码相关的提问，未实现"""
        raise NotImplementedError

    @abstractmethod
    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """由继承类实现的 _achat_completion 方法"""

    @abstractmethod
    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """异步版本的完成方法，所有 GPTAPI 都必须提供标准的 OpenAI 完成接口"""

    @abstractmethod
    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """由继承类实现的流式返回方法"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(ConnectionError),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(
        self, messages: list[dict], stream: bool = False, timeout: int = USE_CONFIG_TIMEOUT
    ) -> str:
        """异步版本的完成方法，返回字符串。支持流式打印"""
        if stream:
            return await self._achat_completion_stream(messages, timeout=self.get_timeout(timeout))
        resp = await self._achat_completion(messages, timeout=self.get_timeout(timeout))
        return self.get_choice_text(resp)

    def get_choice_text(self, rsp: dict) -> str:
        """获取回答的第一个文本内容"""
        message = rsp.get("choices")[0]["message"]
        if "reasoning_content" in message:
            self.reasoning_content = message["reasoning_content"]
        return message["content"]

    def get_choice_delta_text(self, rsp: dict) -> str:
        """获取流式返回的第一个文本内容"""
        return rsp.get("choices", [{}])[0].get("delta", {}).get("content", "")

    def get_choice_function(self, rsp: dict) -> dict:
        """获取返回的第一个函数调用"""
        return rsp.get("choices")[0]["message"]["tool_calls"][0]["function"]

    def get_choice_function_arguments(self, rsp: dict) -> dict:
        """获取函数调用的参数"""
        return json.loads(self.get_choice_function(rsp)["arguments"], strict=False)

    def messages_to_prompt(self, messages: list[dict]):
        """将消息列表转换为提示文本"""
        return "\n".join([f"{i['role']}: {i['content']}" for i in messages])

    def messages_to_dict(self, messages):
        """将消息对象转换为字典格式"""
        return [i.to_dict() for i in messages]

    def with_model(self, model: str):
        """设置模型并返回当前对象"""
        self.config.model = model
        return self

    def get_timeout(self, timeout: int) -> int:
        """获取超时时间"""
        return timeout or self.config.timeout or LLM_API_TIMEOUT

    def count_tokens(self, messages: list[dict]) -> int:
        """粗略的计算消息的 token 数量"""
        return sum([int(len(msg["content"]) * 0.5) for msg in messages])

    def compress_messages(
        self,
        messages: list[dict],
        compress_type: CompressType = CompressType.NO_COMPRESS,
        max_token: int = 128000,
        threshold: float = 0.8,
    ) -> list[dict]:
        """压缩消息以适应 token 限制"""
        if compress_type == CompressType.NO_COMPRESS:
            return messages

        max_token = TOKEN_MAX.get(self.config.model, max_token)
        keep_token = int(max_token * threshold)
        compressed = []

        # 保留系统消息
        system_msg_val = self._system_msg("")["role"]
        system_msgs = []
        for i, msg in enumerate(messages):
            if msg["role"] == system_msg_val:
                system_msgs.append(msg)
            else:
                user_assistant_msgs = messages[i:]
                break
        compressed.extend(system_msgs)
        current_token_count = self.count_tokens(system_msgs)

        # 根据不同的压缩类型处理消息
        if compress_type in [CompressType.POST_CUT_BY_TOKEN, CompressType.POST_CUT_BY_MSG]:
            # 保留最新的消息，直到达到 token 限制
            for i, msg in enumerate(reversed(user_assistant_msgs)):
                token_count = self.count_tokens([msg])
                if current_token_count + token_count <= keep_token:
                    compressed.insert(len(system_msgs), msg)
                    current_token_count += token_count
                else:
                    if compress_type == CompressType.POST_CUT_BY_TOKEN or len(compressed) == len(system_msgs):
                        # 截断消息以适应剩余的 token 限制
                        truncated_content = msg["content"][-(keep_token - current_token_count):]
                        compressed.insert(len(system_msgs), {"role": msg["role"], "content": truncated_content})
                    logger.warning(
                        f"使用 {compress_type} 截断消息以适应 token 限制。"
                    )
                    break

        elif compress_type in [CompressType.PRE_CUT_BY_TOKEN, CompressType.PRE_CUT_BY_MSG]:
            # 保留最早的消息，直到达到 token 限制
            for i, msg in enumerate(user_assistant_msgs):
                token_count = self.count_tokens([msg])
                if current_token_count + token_count <= keep_token:
                    compressed.append(msg)
                    current_token_count += token_count
                else:
                    if compress_type == CompressType.PRE_CUT_BY_TOKEN or len(compressed) == len(system_msgs):
                        # 截断消息以适应剩余的 token 限制
                        truncated_content = msg["content"][: keep_token - current_token_count]
                        compressed.append({"role": msg["role"], "content": truncated_content})
                    logger.warning(
                        f"使用 {compress_type} 截断消息以适应 token 限制。"
                    )
                    break

        return compressed
