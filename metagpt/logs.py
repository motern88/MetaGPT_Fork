#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/1 12:41
@Author  : alexanderwu
@File    : logs.py
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from contextvars import ContextVar
from datetime import datetime
from functools import partial
from typing import Any

from loguru import logger as _logger
from pydantic import BaseModel, Field

from metagpt.const import METAGPT_ROOT

# 定义 LLM（大语言模型）流队列的上下文变量
LLM_STREAM_QUEUE: ContextVar[asyncio.Queue] = ContextVar("llm-stream")


class ToolLogItem(BaseModel):
    """
    工具日志项的数据模型。

    属性：
        type_ (str)：数据类型，对应 `value` 字段的数据类型。
        name (str)：日志项名称。
        value (Any)：日志项的值。
    """
    type_: str = Field(alias="type", default="str", description="Data type of `value` field.")
    name: str
    value: Any


# 特殊的日志项标记，用于表示流日志的结束
TOOL_LOG_END_MARKER = ToolLogItem(
    type="str", name="end_marker", value="\x18\x19\x1B\x18"
)  # A special log item to suggest the end of a stream log

# 日志打印级别，默认为 INFO
_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """
    设置日志级别。

    参数：
        print_level (str)：控制台日志的级别，默认为 "INFO"。
        logfile_level (str)：文件日志的级别，默认为 "DEBUG"。
        name (str)：可选的日志文件前缀名。

    返回：
        Logger 实例。
    """
    global _print_level
    _print_level = print_level

    # 生成当前日期的字符串格式
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")

    # 生成日志文件名
    log_name = f"{name}_{formatted_date}" if name else formatted_date

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)  # 设定控制台日志级别
    _logger.add(METAGPT_ROOT / f"logs/{log_name}.txt", level=logfile_level)  # 设定文件日志级别

    return _logger

# 初始化日志
logger = define_log_level()


def log_llm_stream(msg):
    """
    记录 LLM（大语言模型）流日志。

    参数：
        msg (str)：需要记录的日志信息。

    说明：
        如果 LLM_STREAM_QUEUE 尚未初始化（即 `create_llm_stream_queue` 未被调用），
        该消息不会被添加到 LLM 流队列中。
    """
    queue = get_llm_stream_queue()
    if queue:
        queue.put_nowait(msg)
    _llm_stream_log(msg)  # 调用日志记录函数


def log_tool_output(output: ToolLogItem | list[ToolLogItem], tool_name: str = ""):
    """
    记录工具输出日志。

    参数：
        output (ToolLogItem | list[ToolLogItem])：工具的日志项或日志项列表。
        tool_name (str)：工具名称（可选）。
    """
    _tool_output_log(output=output, tool_name=tool_name)


async def log_tool_output_async(output: ToolLogItem | list[ToolLogItem], tool_name: str = ""):
    """
    异步记录工具输出日志（适用于包含异步对象的情况）。

    参数：
        output (ToolLogItem | list[ToolLogItem])：工具的日志项或日志项列表。
        tool_name (str)：工具名称（可选）。
    """
    await _tool_output_log_async(output=output, tool_name=tool_name)


async def get_human_input(prompt: str = ""):
    """
    获取用户输入。

    参数：
        prompt (str)：提示信息（可选）。

    返回：
        str：用户输入的内容。

    说明：
        允许通过 `set_human_input_func` 替换默认的输入函数，使其可以从不同的来源获取输入。
    """
    if inspect.iscoroutinefunction(_get_human_input):
        return await _get_human_input(prompt)
    else:
        return _get_human_input(prompt)


def set_llm_stream_logfunc(func):
    """设置 LLM 流日志记录函数。"""
    global _llm_stream_log
    _llm_stream_log = func


def set_tool_output_logfunc(func):
    """设置工具输出日志记录函数。"""
    global _tool_output_log
    _tool_output_log = func


async def set_tool_output_logfunc_async(func):
    """设置异步工具输出日志记录函数。"""
    global _tool_output_log_async
    _tool_output_log_async = func


def set_human_input_func(func):
    """设置用户输入函数。"""
    global _get_human_input
    _get_human_input = func


# 默认的 LLM 流日志记录方式，直接打印日志（不换行）
_llm_stream_log = partial(print, end="")


# 默认的工具日志记录方式（空函数，避免未设置时出错）
_tool_output_log = lambda *args, **kwargs: None


async def _tool_output_log_async(*args, **kwargs):
    """异步工具日志记录函数（默认不执行任何操作）。"""
    pass


def create_llm_stream_queue():
    """
    创建新的 LLM 流队列，并将其存储到上下文变量中。

    返回：
        asyncio.Queue：新创建的队列实例。
    """
    queue = asyncio.Queue()
    LLM_STREAM_QUEUE.set(queue)
    return queue


def get_llm_stream_queue():
    """
    获取当前 LLM 流队列。

    返回：
        asyncio.Queue：如果已设置，则返回队列实例，否则返回 None。
    """
    return LLM_STREAM_QUEUE.get(None)


# 默认的用户输入函数，从控制台获取输入
_get_human_input = input


def _llm_stream_log(msg):
    """默认的 LLM 流日志记录函数，仅在日志级别为 INFO 时打印日志。"""
    if _print_level in ["INFO"]:
        print(msg, end="")