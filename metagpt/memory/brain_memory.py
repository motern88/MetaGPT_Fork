#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : brain_memory.py
@Desc    : Used by AgentStore. Used for long-term storage and automatic compression.
@Modified By: mashenquan, 2023/9/4. + redis memory cache.
@Modified By: mashenquan, 2023/12/25. Simplify Functionality.
"""
import json
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from metagpt.config2 import Config as _Config
from metagpt.const import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_SIZE
from metagpt.logs import logger
from metagpt.provider import MetaGPTLLM
from metagpt.provider.base_llm import BaseLLM
from metagpt.schema import Message, SimpleMessage
from metagpt.utils.redis import Redis


class BrainMemory(BaseModel):
    # 历史对话、知识、历史摘要等信息
    history: List[Message] = Field(default_factory=list)
    knowledge: List[Message] = Field(default_factory=list)
    historical_summary: str = ""  # 历史摘要
    last_history_id: str = ""  # 最后一个历史记录的 ID
    is_dirty: bool = False  # 是否有未保存的数据
    last_talk: Optional[str] = None  # 上一次的对话内容
    cacheable: bool = True  # 是否可以缓存
    llm: Optional[BaseLLM] = Field(default=None, exclude=True)  # 用于生成摘要的 LLM 模型
    config: Optional[_Config] = None  # 配置信息

    # 设置默认配置
    @field_validator("config")
    @classmethod
    def set_default_config(cls, config):
        return config if config else _Config.default()

    class Config:
        arbitrary_types_allowed = True  # 允许使用任意类型

    # 将用户的消息添加到历史中
    def add_talk(self, msg: Message):
        """
        添加用户的消息。
        """
        msg.role = "user"
        self.add_history(msg)
        self.is_dirty = True

    # 将 LLM 的回答消息添加到历史中
    def add_answer(self, msg: Message):
        """添加 LLM 的回答消息"""
        msg.role = "assistant"
        self.add_history(msg)
        self.is_dirty = True

    # 获取知识库内容
    def get_knowledge(self) -> str:
        texts = [m.content for m in self.knowledge]
        return "\n".join(texts)

    # 从 Redis 加载历史数据
    async def loads(self, redis_key: str) -> "BrainMemory":
        redis = Redis(self.config.redis)
        if not redis_key:
            return BrainMemory()
        v = await redis.get(key=redis_key)
        logger.debug(f"REDIS GET {redis_key} {v}")
        if v:
            bm = BrainMemory.parse_raw(v)
            bm.is_dirty = False
            return bm
        return BrainMemory()

    # 将当前数据保存到 Redis
    async def dumps(self, redis_key: str, timeout_sec: int = 30 * 60):
        if not self.is_dirty:
            return
        redis = Redis(self.config.redis)
        if not redis_key:
            return False
        v = self.model_dump_json()
        if self.cacheable:
            await redis.set(key=redis_key, data=v, timeout_sec=timeout_sec)
            logger.debug(f"REDIS SET {redis_key} {v}")
        self.is_dirty = False

    # 生成 Redis 键
    @staticmethod
    def to_redis_key(prefix: str, user_id: str, chat_id: str):
        return f"{prefix}:{user_id}:{chat_id}"

    # 设置历史摘要
    async def set_history_summary(self, history_summary, redis_key):
        if self.historical_summary == history_summary:
            if self.is_dirty:
                await self.dumps(redis_key=redis_key)
                self.is_dirty = False
            return

        self.historical_summary = history_summary
        self.history = []  # 清空历史
        await self.dumps(redis_key=redis_key)
        self.is_dirty = False

    # 将消息添加到历史记录中
    def add_history(self, msg: Message):
        if msg.id:
            if self.to_int(msg.id, 0) <= self.to_int(self.last_history_id, -1):
                return

        self.history.append(msg)
        self.last_history_id = str(msg.id)
        self.is_dirty = True

    # 判断给定文本是否存在于历史记录中
    def exists(self, text) -> bool:
        for m in reversed(self.history):
            if m.content == text:
                return True
        return False

    # 将值转换为整数，无法转换时返回默认值
    @staticmethod
    def to_int(v, default_value):
        try:
            return int(v)
        except:
            return default_value

    # 弹出最后一条对话内容
    def pop_last_talk(self):
        v = self.last_talk
        self.last_talk = None
        return v

    # 进行总结
    async def summarize(self, llm, max_words=200, keep_language: bool = False, limit: int = -1, **kwargs):
        if isinstance(llm, MetaGPTLLM):
            return await self._metagpt_summarize(max_words=max_words)

        self.llm = llm
        return await self._openai_summarize(llm=llm, max_words=max_words, keep_language=keep_language, limit=limit)

    # 使用 OpenAI 模型生成摘要
    async def _openai_summarize(self, llm, max_words=200, keep_language: bool = False, limit: int = -1):
        texts = [self.historical_summary]
        for m in self.history:
            texts.append(m.content)
        text = "\n".join(texts)

        text_length = len(text)
        if limit > 0 and text_length < limit:
            return text
        summary = await self._summarize(text=text, max_words=max_words, keep_language=keep_language, limit=limit)
        if summary:
            await self.set_history_summary(history_summary=summary, redis_key=self.config.redis_key)
            return summary
        raise ValueError(f"text too long:{text_length}")

    # 使用 MetaGPT 模型生成摘要
    async def _metagpt_summarize(self, max_words=200):
        if not self.history:
            return ""

        total_length = 0
        msgs = []
        for m in reversed(self.history):
            delta = len(m.content)
            if total_length + delta > max_words:
                left = max_words - total_length
                if left == 0:
                    break
                m.content = m.content[0:left]
                msgs.append(m)
                break
            msgs.append(m)
            total_length += delta
        msgs.reverse()
        self.history = msgs
        self.is_dirty = True
        await self.dumps(redis_key=self.config.redis.key)
        self.is_dirty = False

        return BrainMemory.to_metagpt_history_format(self.history)

    # 将历史记录格式化为 MetaGPT 格式
    @staticmethod
    def to_metagpt_history_format(history) -> str:
        mmsg = [SimpleMessage(role=m.role, content=m.content).model_dump() for m in history]
        return json.dumps(mmsg, ensure_ascii=False)

    # 获取对话标题
    async def get_title(self, llm, max_words=5, **kwargs) -> str:
        """生成对话标题"""
        if isinstance(llm, MetaGPTLLM):
            return self.history[0].content if self.history else "New"

        summary = await self.summarize(llm=llm, max_words=500)

        language = self.config.language
        command = f"Translate the above summary into a {language} title of less than {max_words} words."
        summaries = [summary, command]
        msg = "\n".join(summaries)
        logger.debug(f"title ask:{msg}")
        response = await llm.aask(msg=msg, system_msgs=[], stream=False)
        logger.debug(f"title rsp: {response}")
        return response

    async def is_related(self, text1, text2, llm):
        # 判断两段文本是否相关
        if isinstance(llm, MetaGPTLLM):
            return await self._metagpt_is_related(text1=text1, text2=text2, llm=llm)
        return await self._openai_is_related(text1=text1, text2=text2, llm=llm)

    @staticmethod
    async def _metagpt_is_related(**kwargs):
        # MetaGPT不进行文本相关性判断，直接返回False
        return False

    @staticmethod
    async def _openai_is_related(text1, text2, llm, **kwargs):
        # 使用OpenAI模型判断文本相关性
        context = f"## Paragraph 1\n{text2}\n---\n## Paragraph 2\n{text1}\n"
        rsp = await llm.aask(
            msg=context,
            system_msgs=[
                "You are a tool capable of determining whether two paragraphs are semantically related."
                'Return "TRUE" if "Paragraph 1" is semantically relevant to "Paragraph 2", otherwise return "FALSE".'
            ],
            stream=False,
        )
        result = True if "TRUE" in rsp else False
        p2 = text2.replace("\n", "")
        p1 = text1.replace("\n", "")
        logger.info(f"IS_RELATED:\nParagraph 1: {p2}\nParagraph 2: {p1}\nRESULT: {result}\n")
        return result

    async def rewrite(self, sentence: str, context: str, llm):
        # 重写给定句子
        if isinstance(llm, MetaGPTLLM):
            return await self._metagpt_rewrite(sentence=sentence, context=context, llm=llm)
        return await self._openai_rewrite(sentence=sentence, context=context, llm=llm)

    @staticmethod
    async def _metagpt_rewrite(sentence: str, **kwargs):
        # MetaGPT不进行重写，直接返回原句子
        return sentence

    @staticmethod
    async def _openai_rewrite(sentence: str, context: str, llm):
        # 使用OpenAI模型重写句子
        prompt = f"## Context\n{context}\n---\n## Sentence\n{sentence}\n"
        rsp = await llm.aask(
            msg=prompt,
            system_msgs=[
                'You are a tool augmenting the "Sentence" with information from the "Context".',
                "Do not supplement the context with information that is not present, especially regarding the subject and object.",
                "Return the augmented sentence.",
            ],
            stream=False,
        )
        logger.info(f"REWRITE:\nCommand: {prompt}\nRESULT: {rsp}\n")
        return rsp

    @staticmethod
    def extract_info(input_string, pattern=r"\[([A-Z]+)\]:\s*(.+)"):
        # 提取信息，匹配输入字符串中的模式
        match = re.match(pattern, input_string)
        if match:
            return match.group(1), match.group(2)
        else:
            return None, input_string

    @property
    def is_history_available(self):
        # 判断历史记录是否可用
        return bool(self.history or self.historical_summary)

    @property
    def history_text(self):
        # 获取历史记录的文本内容
        if len(self.history) == 0 and not self.historical_summary:
            return ""
        texts = [self.historical_summary] if self.historical_summary else []
        for m in self.history[:-1]:
            if isinstance(m, Dict):
                t = Message(**m).content
            elif isinstance(m, Message):
                t = m.content
            else:
                continue
            texts.append(t)

        return "\n".join(texts)

    async def _summarize(self, text: str, max_words=200, keep_language: bool = False, limit: int = -1) -> str:
        # 将文本进行摘要
        max_token_count = DEFAULT_MAX_TOKENS
        max_count = 100
        text_length = len(text)
        if limit > 0 and text_length < limit:
            return text
        summary = ""
        while max_count > 0:
            if text_length < max_token_count:
                summary = await self._get_summary(text=text, max_words=max_words, keep_language=keep_language)
                break

            padding_size = 20 if max_token_count > 20 else 0
            text_windows = self.split_texts(text, window_size=max_token_count - padding_size)
            part_max_words = min(int(max_words / len(text_windows)) + 1, 100)
            summaries = []
            for ws in text_windows:
                response = await self._get_summary(text=ws, max_words=part_max_words, keep_language=keep_language)
                summaries.append(response)
            if len(summaries) == 1:
                summary = summaries[0]
                break

            # 合并并重试
            text = "\n".join(summaries)
            text_length = len(text)

            max_count -= 1  # 防止死循环
        return summary

    async def _get_summary(self, text: str, max_words=20, keep_language: bool = False):
        """生成文本摘要"""
        if len(text) < max_words:
            return text
        system_msgs = [
            "You are a tool for summarizing and abstracting text.",
            f"Return the summarized text to less than {max_words} words.",
        ]
        if keep_language:
            system_msgs.append("The generated summary should be in the same language as the original text.")
        response = await self.llm.aask(msg=text, system_msgs=system_msgs, stream=False)
        logger.debug(f"{text}\nsummary rsp: {response}")
        return response

    @staticmethod
    def split_texts(text: str, window_size) -> List[str]:
        """将长文本分割成滑动窗口"""
        if window_size <= 0:
            window_size = DEFAULT_TOKEN_SIZE
        total_len = len(text)
        if total_len <= window_size:
            return [text]

        padding_size = 20 if window_size > 20 else 0
        windows = []
        idx = 0
        data_len = window_size - padding_size
        while idx < total_len:
            if window_size + idx > total_len:  # 不足一个窗口
                windows.append(text[idx:])
                break
            # 每个窗口少算padding_size自然就可实现滑窗功能, 比如: [1, 2, 3, 4, 5, 6, 7, ....]
            # window_size=3, padding_size=1：：
            # [1, 2, 3], [3, 4, 5], [5, 6, 7], ....
            # idx=2,  |  idx=5   |  idx=8  | ...
            w = text[idx : idx + window_size]
            windows.append(w)
            idx += data_len

        return windows
