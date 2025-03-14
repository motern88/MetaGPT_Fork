#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:07
@Author  : alexanderwu
@File    : common.py
@Modified By: mashenquan, 2023-11-1. According to Chapter 2.2.2 of RFC 116:
        Add generic class-to-string and object-to-string conversion functionality.
@Modified By: mashenquan, 2023/11/27. Bug fix: `parse_recipient` failed to parse the recipient in certain GPT-3.5
        responses.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import csv
import functools
import hashlib
import importlib
import inspect
import json
import mimetypes
import os
import platform
import re
import sys
import time
import traceback
import uuid
from asyncio import iscoroutinefunction
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import quote, unquote

import aiofiles
import aiohttp
import chardet
import loguru
import requests
from PIL import Image
from pydantic_core import to_jsonable_python
from tenacity import RetryCallState, RetryError, _utils

from metagpt.const import MARKDOWN_TITLE_PREFIX, MESSAGE_ROUTE_TO_ALL
from metagpt.logs import logger
from metagpt.utils.exceptions import handle_exception
from metagpt.utils.json_to_markdown import json_to_markdown


def check_cmd_exists(command) -> int:
    """检查命令是否存在
    :param command: 待检查的命令
    :return: 如果命令存在，返回0，如果不存在，返回非0
    """
    if platform.system().lower() == "windows":
        check_command = "where " + command
    else:
        check_command = "command -v " + command + ' >/dev/null 2>&1 || { echo >&2 "no mermaid"; exit 1; }'
    result = os.system(check_command)
    return result


def require_python_version(req_version: Tuple) -> bool:
    """
    检查当前 Python 版本是否高于指定版本。

    :param req_version: 需要的 Python 版本，例如 (3, 9) 或 (3, 10, 13)。
    :return: 如果当前 Python 版本高于 `req_version`，返回 True，否则返回 False。
    """
    if not (2 <= len(req_version) <= 3):
        raise ValueError("req_version should be (3, 9) or (3, 10, 13)")
    return bool(sys.version_info > req_version)


class OutputParser:
    @classmethod
    def parse_blocks(cls, text: str):
        """
        解析文本，将其分割为多个块。

        :param text: 需要解析的文本数据。
        :return: 一个字典，键是块的标题，值是块的内容。
        """
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split(MARKDOWN_TITLE_PREFIX)

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
                # LLM可能出错，在这里做一下修正
                if block_title[-1] == ":":
                    block_title = block_title[:-1]
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, text: str, lang: str = "") -> str:
        """
        提取代码块内容。

        :param text: 需要解析的文本数据。
        :param lang: 代码语言（如 "python"），默认为空字符串。
        :return: 解析出的代码内容（字符串）。
        """
        pattern = rf"```{lang}.*?\s+(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise Exception
        return code

    @classmethod
    def parse_str(cls, text: str):
        """
        解析字符串，将等号后的内容提取出来，并去掉多余的引号。

        :param text: 需要解析的字符串。
        :return: 解析出的字符串。
        """
        text = text.split("=")[-1]
        text = text.strip().strip("'").strip('"')
        return text

    @classmethod
    def parse_file_list(cls, text: str) -> list[str]:
        """
        解析文件列表。

        :param text: 需要解析的文本数据。
        :return: 解析出的文件列表（列表）。
        """
        # Regular expression pattern to find the tasks list.
        pattern = r"\s*(.*=.*)?(\[.*\])"

        # Extract tasks list string using regex.
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            tasks = text.split("\n")
        return tasks

    @staticmethod
    def parse_python_code(text: str) -> str:
        """
        解析 Python 代码块，并验证代码是否有效。

        :param text: 需要解析的文本数据。
        :return: 解析出的 Python 代码。
        :raises ValueError: 如果代码无效，则抛出异常。
        """
        for pattern in (r"(.*?```python.*?\s+)?(?P<code>.*)(```.*?)", r"(.*?```python.*?\s+)?(?P<code>.*)"):
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            code = match.group("code")
            if not code:
                continue
            with contextlib.suppress(Exception):
                ast.parse(code)
                return code
        raise ValueError("Invalid python code")

    @classmethod
    def parse_data(cls, data):
        """
        解析数据，自动提取代码块、列表等格式化内容。

        :param data: 需要解析的文本数据。
        :return: 解析后的数据字典。
        """
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                # 尝试解析list
                try:
                    content = cls.parse_file_list(text=content)
                except Exception:
                    pass
            parsed_data[block] = content
        return parsed_data

    @staticmethod
    def extract_content(text, tag="CONTENT"):
        """
        提取标记 [CONTENT] 和 [/CONTENT] 之间的内容。

        :param text: 需要解析的文本数据。
        :param tag: 需要提取的标签（默认为 "CONTENT"）。
        :return: 提取出的内容。
        :raises ValueError: 如果未找到指定内容，则抛出异常。
        """
        # Use regular expression to extract content between [CONTENT] and [/CONTENT]
        extracted_content = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)

        if extracted_content:
            return extracted_content.group(1).strip()
        else:
            raise ValueError(f"Could not find content between [{tag}] and [/{tag}]")

    @classmethod
    def parse_data_with_mapping(cls, data, mapping):
        """
        按照给定的映射规则解析数据。

        :param data: 需要解析的文本数据。
        :param mapping: 预定义的字段映射规则（字典）。
        :return: 解析后的数据字典。
        """
        if "[CONTENT]" in data:
            data = cls.extract_content(text=data)
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                pass
            typing_define = mapping.get(block, None)
            if isinstance(typing_define, tuple):
                typing = typing_define[0]
            else:
                typing = typing_define
            if typing == List[str] or typing == List[Tuple[str, str]] or typing == List[List[str]]:
                # 尝试解析list
                try:
                    content = cls.parse_file_list(text=content)
                except Exception:
                    pass
            # TODO: 多余的引号去除有风险，后期再解决
            # elif typing == str:
            #     # 尝试去除多余的引号
            #     try:
            #         content = cls.parse_str(text=content)
            #     except Exception:
            #         pass
            parsed_data[block] = content
        return parsed_data

    @classmethod
    def extract_struct(cls, text: str, data_type: Union[type(list), type(dict)]) -> Union[list, dict]:
        """从给定文本中提取并解析指定类型的数据结构（字典或列表）。
        文本仅包含一个列表或字典，并且可能包含嵌套结构。

        参数：
            text: 包含数据结构（字典或列表）的文本。
            data_type: 需要提取的数据类型，可以是 `list` 或 `dict`。

        返回：
            - 如果提取和解析成功，则返回相应的数据结构（列表或字典）。
            - 如果提取失败或解析出错，则抛出异常。

        示例：
            >>> text = 'xxx [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}] xxx'
            >>> result_list = OutputParser.extract_struct(text, list)
            >>> print(result_list)
            >>> # 输出: [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}]

            >>> text = 'xxx {"x": 1, "y": {"a": 2, "b": {"c": 3}}} xxx'
            >>> result_dict = OutputParser.extract_struct(text, dict)
            >>> print(result_dict)
            >>> # 输出: {"x": 1, "y": {"a": 2, "b": {"c": 3}}}
        """
        # 查找第一个 "[" 或 "{" 以及最后一个 "]" 或 "}"
        start_index = text.find("[" if data_type is list else "{")
        end_index = text.rfind("]" if data_type is list else "}")

        if start_index != -1 and end_index != -1:
            # 提取结构部分
            structure_text = text[start_index: end_index + 1]

            try:
                # 使用 ast.literal_eval 将文本转换为 Python 数据结构
                result = ast.literal_eval(structure_text)

                # 确保解析结果与指定数据类型匹配
                if isinstance(result, (list, dict)):
                    return result

                raise ValueError(f"提取的数据结构不是 {data_type} 类型。")

            except (ValueError, SyntaxError) as e:
                raise Exception(f"提取并解析 {data_type} 结构时发生错误: {e}")
        else:
            logger.error(f"在文本中未找到 {data_type} 数据结构。")
            return [] if data_type is list else {}


class CodeParser:
    @classmethod
    def parse_block(cls, block: str, text: str) -> str:
        """
        从文本中提取指定的 block 内容。

        参数：
            block: 需要提取的 block 名称（标题）。
            text: 包含多个 block 的文本。

        返回：
            block 对应的内容字符串，如果找不到则返回空字符串。
        """
        blocks = cls.parse_blocks(text)  # 解析所有 block
        for k, v in blocks.items():
            if block in k:  # 如果 block 名称匹配，则返回对应内容
                return v
        return ""

    @classmethod
    def parse_blocks(cls, text: str):
        """
        解析文本中的所有 block，并以字典的形式返回。

        参数：
            text: 包含多个 block 的文本，每个 block 以 "##" 作为分隔符。

        返回：
            一个字典，键是 block 标题，值是 block 内容。
        """
        # 以 "##" 作为分隔符，将文本拆分成多个 block
        blocks = text.split("##")

        # 创建一个字典，用于存储 block 标题和内容
        block_dict = {}

        # 遍历所有的 block
        for block in blocks:
            if block.strip() == "":  # 忽略空 block
                continue
            if "\n" not in block:  # 仅有标题但无内容
                block_title = block
                block_content = ""
            else:
                # 将 block 标题和内容分开，并去除前后空白字符
                block_title, block_content = block.split("\n", 1)
            block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, text: str, lang: str = "", block: Optional[str] = None) -> str:
        """
        从文本中提取指定语言的代码块内容。

        参数：
            text: 包含代码块的文本。
            lang: 代码块的语言（默认为空，表示匹配任何语言）。
            block: 可选，指定从某个 block 里提取代码。

        返回：
            提取到的代码内容字符串。如果未匹配到代码块，则返回原始文本。
        """
        if block:
            text = cls.parse_block(block, text)  # 先提取指定 block 的内容
        pattern = rf"```{lang}.*?\s+(.*?)\n```"  # 匹配代码块
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)  # 提取代码内容
        else:
            logger.error(f"{pattern} not match following text:")
            logger.error(text)
            return text  # 假设原始文本就是代码
        return code

    @classmethod
    def parse_str(cls, block: str, text: str, lang: str = ""):
        """
        从文本中的指定 block 里提取字符串变量值。

        参数：
            block: 代码所在的 block 名称。
            text: 包含代码的文本。
            lang: 代码语言（默认为空，表示匹配任何语言）。

        返回：
            提取出的字符串值（去掉引号）。
        """
        code = cls.parse_code(block=block, text=text, lang=lang)
        code = code.split("=")[-1]  # 取等号后面的部分
        code = code.strip().strip("'").strip('"')  # 去除前后空白和引号
        return code

    @classmethod
    def parse_file_list(cls, block: str, text: str, lang: str = "") -> list[str]:
        """
        从文本中的指定 block 里提取列表变量。

        参数：
            block: 代码所在的 block 名称。
            text: 包含代码的文本。
            lang: 代码语言（默认为空，表示匹配任何语言）。

        返回：
            解析出的 Python 列表。
        """
        # 获取代码内容
        code = cls.parse_code(block=block, text=text, lang=lang)

        # 正则表达式匹配列表
        pattern = r"\s*(.*=.*)?(\[.*\])"

        # 使用正则表达式提取列表字符串
        match = re.search(pattern, code, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)  # 取列表部分

            # 将字符串转换为 Python 列表
            tasks = ast.literal_eval(tasks_list_str)
        else:
            raise Exception("未能解析出列表")
        return tasks


class NoMoneyException(Exception):
    """当资金不足时抛出的异常"""

    def __init__(self, amount, message="资金不足"):
        self.amount = amount
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} -> 需要金额: {self.amount}"


def print_members(module, indent=0):
    """
    打印模块的所有成员，包括类和方法

    来源: https://stackoverflow.com/questions/1796180/
    """
    prefix = " " * indent
    for name, obj in inspect.getmembers(module):
        print(name, obj)
        if inspect.isclass(obj):
            print(f"{prefix}类: {name}")
            if name in ["__class__", "__base__"]:
                continue
            print_members(obj, indent + 2)
        elif inspect.isfunction(obj):
            print(f"{prefix}函数: {name}")
        elif inspect.ismethod(obj):
            print(f"{prefix}方法: {name}")


def get_function_schema(func: Callable) -> dict[str, Union[dict, Any, str]]:
    """
    获取函数的输入参数、返回类型和文档说明
    """
    sig = inspect.signature(func)
    parameters = sig.parameters
    return_type = sig.return_annotation
    param_schema = {name: parameter.annotation for name, parameter in parameters.items()}
    return {"输入参数": param_schema, "返回类型": return_type, "函数描述": func.__doc__, "函数": func}


def parse_recipient(text):
    """
    解析文本中的收件人信息
    """
    pattern = r"## Send To:\s*([A-Za-z]+)\s*?"
    recipient = re.search(pattern, text)
    if recipient:
        return recipient.group(1)

    pattern = r"Send To:\s*([A-Za-z]+)\s*?"
    recipient = re.search(pattern, text)
    if recipient:
        return recipient.group(1)

    return ""


def remove_comments(code_str: str) -> str:
    """
    移除代码中的注释
    """
    pattern = r"(\".*?\"|\'.*?\')|(\#.*?$)"

    def replace_func(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    clean_code = re.sub(pattern, replace_func, code_str, flags=re.MULTILINE)
    clean_code = os.linesep.join([s.rstrip() for s in clean_code.splitlines() if s.strip()])
    return clean_code


def get_class_name(cls) -> str:
    """
    获取类的完整名称（包括模块名）
    """
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """
    将任意对象转换为字符串:
    - 若为字符串，则直接返回
    - 若为类或对象，则返回类名
    """
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)


def any_to_str_set(val) -> set:
    """
    将任意数据转换为字符串集合
    """
    res = set()
    if isinstance(val, (dict, list, set, tuple)):
        if isinstance(val, dict):
            val = val.values()
        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))
    return res


def is_send_to(message: "Message", addresses: set):
    """
    判断消息是否需要发送到指定地址
    """
    if "ALL" in message.send_to:
        return True
    for i in addresses:
        if i in message.send_to:
            return True
    return False


def any_to_name(val):
    """
    获取对象的名称（去掉模块路径）
    """
    return any_to_str(val).split(".")[-1]


def concat_namespace(*args, delimiter: str = ":") -> str:
    """
    连接多个字段，形成唯一的命名空间前缀
    """
    return delimiter.join(str(value) for value in args)


def split_namespace(ns_class_name: str, delimiter: str = ":", maxsplit: int = 1) -> List[str]:
    """
    将命名空间前缀的名称拆分成前缀和名称部分
    """
    return ns_class_name.split(delimiter, maxsplit=maxsplit)


def auto_namespace(name: str, delimiter: str = ":") -> str:
    """
    自动处理命名空间前缀的名称:
    - 如果为空，则返回默认前缀 `?:?`
    - 如果没有命名空间前缀，则添加 `?` 作为默认前缀
    """
    if not name:
        return f"?{delimiter}?"
    v = split_namespace(name, delimiter=delimiter)
    if len(v) < 2:
        return f"?{delimiter}{name}"
    return name


def add_affix(text: str, affix: Literal["brace", "url", "none"] = "brace"):
    """为字符串添加前后缀封装。

    示例：
        >>> add_affix("data", affix="brace")
        '{data}'

        >>> add_affix("example.com", affix="url")
        '%7Bexample.com%7D'

        >>> add_affix("text", affix="none")
        'text'
    """
    mappings = {
        "brace": lambda x: "{" + x + "}",
        "url": lambda x: quote("{" + x + "}"),
    }
    encoder = mappings.get(affix, lambda x: x)
    return encoder(text)


def remove_affix(text, affix: Literal["brace", "url", "none"] = "brace"):
    """移除字符串的前后缀封装。

    示例：
        >>> remove_affix('{data}', affix="brace")
        'data'

        >>> remove_affix('%7Bexample.com%7D', affix="url")
        'example.com'

        >>> remove_affix('text', affix="none")
        'text'
    """
    mappings = {"brace": lambda x: x[1:-1], "url": lambda x: unquote(x)[1:-1]}
    decoder = mappings.get(affix, lambda x: x)
    return decoder(text)


def general_after_log(i: "loguru.Logger", sec_format: str = "%0.3f") -> Callable[["RetryCallState"], None]:
    """
    生成一个用于日志记录的回调函数，记录重试操作的结果。

    :param i: loguru.Logger 日志实例
    :param sec_format: 时间格式，默认保留三位小数
    :return: 可调用对象，接受 RetryCallState 并进行日志记录
    """

    def log_it(retry_state: "RetryCallState") -> None:
        # 获取调用的函数名称
        fn_name = "<unknown>" if retry_state.fn is None else _utils.get_callback_name(retry_state.fn)

        # 记录错误日志
        i.error(
            f"调用 '{fn_name}' 结束，耗时 {sec_format % retry_state.seconds_since_start} 秒，"
            f"第 {_utils.to_ordinal(retry_state.attempt_number)} 次尝试，异常: {retry_state.outcome.exception()}"
        )

    return log_it


def read_json_file(json_file: str, encoding: str = "utf-8") -> list[Any]:
    """读取 JSON 文件，并返回解析后的数据"""
    if not Path(json_file).exists():
        raise FileNotFoundError(f"文件 {json_file} 不存在")

    with open(json_file, "r", encoding=encoding) as fin:
        try:
            data = json.load(fin)
        except Exception:
            raise ValueError(f"读取 JSON 文件失败: {json_file}")
    return data


def handle_unknown_serialization(x: Any) -> str:
    """处理无法序列化的对象，提供详细错误信息"""
    if inspect.ismethod(x):
        tip = f"无法序列化方法 '{x.__func__.__name__}' (类 '{x.__self__.__class__.__name__}')"
    elif inspect.isfunction(x):
        tip = f"无法序列化函数 '{x.__name__}'"
    elif hasattr(x, "__class__"):
        tip = f"无法序列化 '{x.__class__.__name__}' 的实例"
    else:
        tip = f"无法序列化对象类型 '{type(x).__name__}'"

    raise TypeError(tip)


def write_json_file(json_file: str, data: Any, encoding: str = "utf-8", indent: int = 4, use_fallback: bool = False):
    """写入 JSON 文件"""
    folder_path = Path(json_file).parent
    folder_path.mkdir(parents=True, exist_ok=True)

    custom_default = partial(to_jsonable_python, fallback=handle_unknown_serialization if use_fallback else None)

    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=custom_default)


def read_jsonl_file(jsonl_file: str, encoding="utf-8") -> list[dict]:
    """读取 JSONL（每行一个 JSON 记录）文件"""
    if not Path(jsonl_file).exists():
        raise FileNotFoundError(f"文件 {jsonl_file} 不存在")

    datas = []
    with open(jsonl_file, "r", encoding=encoding) as fin:
        try:
            for line in fin:
                datas.append(json.loads(line))
        except Exception:
            raise ValueError(f"读取 JSONL 文件失败: {jsonl_file}")
    return datas


def add_jsonl_file(jsonl_file: str, data: list[dict], encoding: str = None):
    """向 JSONL 文件追加数据"""
    folder_path = Path(jsonl_file).parent
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, "a", encoding=encoding) as fout:
        for json_item in data:
            fout.write(json.dumps(json_item) + "\n")


def read_csv_to_list(curr_file: str, header=False, strip_trail=True):
    """
    读取 CSV 文件，并返回数据列表。

    :param curr_file: CSV 文件路径
    :param header: 是否包含表头
    :param strip_trail: 是否去除行末空白
    :return: 解析后的数据列表
    """
    logger.debug(f"开始读取 CSV 文件: {curr_file}")
    analysis_list = []
    with open(curr_file) as f_analysis_file:
        data_reader = csv.reader(f_analysis_file, delimiter=",")
        for count, row in enumerate(data_reader):
            if strip_trail:
                row = [i.strip() for i in row]
            analysis_list.append(row)

    return (analysis_list[0], analysis_list[1:]) if header else analysis_list



def import_class(class_name: str, module_name: str) -> type:
    """从模块中导入类"""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def import_class_inst(class_name: str, module_name: str, *args, **kwargs) -> object:
    """从模块中导入类并实例化"""
    a_class = import_class(class_name, module_name)
    return a_class(*args, **kwargs)


def format_trackback_info(limit: int = 2):
    """格式化异常信息"""
    return traceback.format_exc(limit=limit)

def serialize_decorator(func):
    """捕获异常并序列化的装饰器"""
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except KeyboardInterrupt:
            logger.error(f"检测到 KeyboardInterrupt，正在序列化项目，异常详情:\n{format_trackback_info()}")
        except Exception:
            logger.error(f"发生异常，正在序列化项目，异常详情:\n{format_trackback_info()}")
        self.serialize()  # 触发对象的 serialize 方法

    return wrapper


def role_raise_decorator(func):
    """捕获异常并处理角色执行异常的装饰器"""
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except KeyboardInterrupt as kbi:
            logger.error(f"检测到 KeyboardInterrupt: {kbi}，正在序列化项目")
            if self.latest_observed_msg:
                self.rc.memory.delete(self.latest_observed_msg)
            raise Exception(format_trackback_info(limit=None))
        except Exception as e:
            if self.latest_observed_msg:
                logger.exception("角色执行异常，删除最新的消息以便重新处理")
                self.rc.memory.delete(self.latest_observed_msg)
            raise Exception(format_trackback_info(limit=None)) from e

    return wrapper


@handle_exception
async def aread(filename: str | Path, encoding="utf-8") -> str:
    """异步读取文件内容。

    参数：
        filename (str | Path): 文件路径或名称。
        encoding (str): 文件编码格式，默认为 UTF-8。

    返回：
        str: 读取的文件内容。如果文件不存在，则返回空字符串。
    """
    if not filename or not Path(filename).exists():
        return ""
    try:
        async with aiofiles.open(str(filename), mode="r", encoding=encoding) as reader:
            content = await reader.read()
    except UnicodeDecodeError:
        # 如果编码错误，尝试自动检测编码并读取
        async with aiofiles.open(str(filename), mode="rb") as reader:
            raw = await reader.read()
            result = chardet.detect(raw)
            detected_encoding = result["encoding"]
            content = raw.decode(detected_encoding)
    return content


async def awrite(filename: str | Path, data: str, encoding="utf-8"):
    """异步写入文件。

    参数：
        filename (str | Path): 文件路径或名称。
        data (str): 要写入的文本数据。
        encoding (str): 编码格式，默认为 UTF-8。
    """
    pathname = Path(filename)
    pathname.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    async with aiofiles.open(str(pathname), mode="w", encoding=encoding) as writer:
        await writer.write(data)


async def read_file_block(filename: str | Path, lineno: int, end_lineno: int):
    """异步读取文件的指定行范围。

    参数：
        filename (str | Path): 文件路径或名称。
        lineno (int): 起始行号（从 1 开始）。
        end_lineno (int): 结束行号（包含）。

    返回：
        str: 读取的行内容。
    """
    if not Path(filename).exists():
        return ""
    lines = []
    async with aiofiles.open(str(filename), mode="r") as reader:
        ix = 0
        while ix < end_lineno:
            ix += 1
            line = await reader.readline()
            if ix < lineno:
                continue
            if ix > end_lineno:
                break
            lines.append(line)
    return "".join(lines)


def list_files(root: str | Path) -> List[Path]:
    """递归列出指定目录下的所有文件。

    参数：
        root (str | Path): 目录路径。

    返回：
        List[Path]: 该目录下的所有文件路径列表。
    """
    files = []
    try:
        directory_path = Path(root)
        if not directory_path.exists():
            return []
        for file_path in directory_path.iterdir():
            if file_path.is_file():
                files.append(file_path)
            else:
                subfolder_files = list_files(root=file_path)  # 递归获取子目录文件
                files.extend(subfolder_files)
    except Exception as e:
        logger.error(f"错误: {e}")
    return files


def parse_json_code_block(markdown_text: str) -> List[str]:
    """从 Markdown 文本中提取 JSON 代码块。

    参数：
        markdown_text (str): Markdown 格式的文本。

    返回：
        List[str]: JSON 代码块列表。
    """
    json_blocks = (
        re.findall(r"```json(.*?)```", markdown_text, re.DOTALL) if "```json" in markdown_text else [markdown_text]
    )
    return [v.strip() for v in json_blocks]



def remove_white_spaces(v: str) -> str:
    """移除字符串中的多余空格（不影响字符串中的引号内容）。

    参数：
        v (str): 输入字符串。

    返回：
        str: 处理后的字符串。
    """
    return re.sub(r"(?<!['\"])\s|(?<=['\"])\s", "", v)


async def aread_bin(filename: str | Path) -> bytes:
    """异步读取二进制文件。

    参数：
        filename (str | Path): 文件路径或名称。

    返回：
        bytes: 读取的二进制内容。
    """
    async with aiofiles.open(str(filename), mode="rb") as reader:
        content = await reader.read()
    return content


async def awrite_bin(filename: str | Path, data: bytes):
    """异步写入二进制文件。

    参数：
        filename (str | Path): 文件路径或名称。
        data (bytes): 要写入的二进制数据。
    """
    pathname = Path(filename)
    pathname.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(str(pathname), mode="wb") as writer:
        await writer.write(data)


def is_coroutine_func(func: Callable) -> bool:
    """检查函数是否为协程函数。

    参数：
        func (Callable): 目标函数。

    返回：
        bool: 是否为协程函数。
    """
    return inspect.iscoroutinefunction(func)


def load_mc_skills_code(skill_names: list[str] = None, skills_dir: Path = None) -> list[str]:
    """load minecraft skill from js files"""
    if not skills_dir:
        skills_dir = Path(__file__).parent.absolute()
    if skill_names is None:
        skill_names = [skill[:-3] for skill in os.listdir(f"{skills_dir}") if skill.endswith(".js")]
    skills = [skills_dir.joinpath(f"{skill_name}.js").read_text() for skill_name in skill_names]
    return skills


def encode_image(image_path_or_pil: Union[Path, Image, str], encoding: str = "utf-8") -> str:
    """将图片编码为 Base64 字符串。

    参数：
        image_path_or_pil (Union[Path, Image, str]): 图片路径或 PIL.Image 对象。
        encoding (str): 编码格式，默认为 UTF-8。

    返回：
        str: Base64 编码的字符串。
    """
    if isinstance(image_path_or_pil, Image.Image):
        buffer = BytesIO()
        image_path_or_pil.save(buffer, format="JPEG")
        bytes_data = buffer.getvalue()
    else:
        if isinstance(image_path_or_pil, str):
            image_path_or_pil = Path(image_path_or_pil)
        if not image_path_or_pil.exists():
            raise FileNotFoundError(f"{image_path_or_pil} 不存在")
        with open(str(image_path_or_pil), "rb") as image_file:
            bytes_data = image_file.read()
    return base64.b64encode(bytes_data).decode(encoding)


def decode_image(img_url_or_b64: str) -> Image:
    """将 Base64 编码的图片或 URL 下载的图片解码为 PIL.Image。

    参数：
        img_url_or_b64 (str): 图片的 Base64 编码或 URL。

    返回：
        Image: PIL.Image 对象。
    """
    if img_url_or_b64.startswith("http"):
        # 处理 HTTP 图片 URL
        resp = requests.get(img_url_or_b64)
        img = Image.open(BytesIO(resp.content))
    else:
        # 处理 Base64 编码图片
        b64_data = re.sub("^data:image/.+;base64,", "", img_url_or_b64)
        img_data = BytesIO(base64.b64decode(b64_data))
        img = Image.open(img_data)
    return img


def extract_image_paths(content: str) -> bool:
    """从文本中提取图片路径。

    参数：
        content (str): 输入文本。

    返回：
        bool: 识别到的图片路径列表。
    """
    pattern = r"[^\s]+\.(?:png|jpe?g|gif|bmp|tiff|PNG|JPE?G|GIF|BMP|TIFF)"
    image_paths = re.findall(pattern, content)
    return image_paths



def extract_and_encode_images(content: str) -> list[str]:
    images = []
    for path in extract_image_paths(content):
        if os.path.exists(path):
            images.append(encode_image(path))
    return images


def log_and_reraise(retry_state: RetryCallState):
    logger.error(f"Retry attempts exhausted. Last exception: {retry_state.outcome.exception()}")
    logger.warning(
        """
Recommend going to https://deepwisdom.feishu.cn/wiki/MsGnwQBjiif9c3koSJNcYaoSnu4#part-XdatdVlhEojeAfxaaEZcMV3ZniQ
See FAQ 5.8
"""
    )
    raise retry_state.outcome.exception()


async def get_mime_type(filename: str | Path, force_read: bool = False) -> str:
    # 尝试从文件名中推断 MIME 类型
    guess_mime_type, _ = mimetypes.guess_type(filename.name)
    if not guess_mime_type:
        # 如果无法推测 MIME 类型，则根据文件扩展名进行映射
        ext_mappings = {".yml": "text/yaml", ".yaml": "text/yaml"}
        guess_mime_type = ext_mappings.get(filename.suffix)
    if not force_read and guess_mime_type:
        # 如果没有强制读取并且已经推测出 MIME 类型，则直接返回
        return guess_mime_type

    # 避免循环导入
    from metagpt.tools.libs.shell import shell_execute

    # 定义一组文本 MIME 类型
    text_set = {
        "application/json",
        "application/vnd.chipnuts.karaoke-mmd",
        "application/javascript",
        "application/xml",
        "application/x-sh",
        "application/sql",
        "text/yaml",
    }

    try:
        # 使用 shell 命令 `file --mime-type` 获取 MIME 类型
        stdout, stderr, _ = await shell_execute(f"file --mime-type '{str(filename)}'")
        if stderr:
            logger.debug(f"file:{filename}, error:{stderr}")
            return guess_mime_type
        # 提取 MIME 类型
        ix = stdout.rfind(" ")
        mime_type = stdout[ix:].strip()
        # 如果 MIME 类型为 'text/plain' 且推测类型是文本类型，则返回推测的 MIME 类型
        if mime_type == "text/plain" and guess_mime_type in text_set:
            return guess_mime_type
        return mime_type
    except Exception as e:
        logger.debug(f"file:{filename}, error:{e}")
        return "unknown"


def get_markdown_codeblock_type(filename: str = None, mime_type: str = None) -> str:
    """返回与文件扩展名对应的 Markdown 代码块类型。"""
    if not filename and not mime_type:
        raise ValueError("必须提供有效的 filename 或 mime_type。")

    if not mime_type:
        mime_type, _ = mimetypes.guess_type(filename)
    # MIME 类型与 Markdown 代码块类型的映射关系
    mappings = {
        "text/x-shellscript": "bash",
        "text/x-c++src": "cpp",
        "text/css": "css",
        "text/html": "html",
        "text/x-java": "java",
        "text/x-python": "python",
        "text/x-ruby": "ruby",
        "text/x-c": "cpp",
        "text/yaml": "yaml",
        "application/javascript": "javascript",
        "application/json": "json",
        "application/sql": "sql",
        "application/vnd.chipnuts.karaoke-mmd": "mermaid",
        "application/x-sh": "bash",
        "application/xml": "xml",
    }
    return mappings.get(mime_type, "text")


def get_project_srcs_path(workdir: str | Path) -> Path:
    # 获取项目的源代码路径
    src_workdir_path = workdir / ".src_workspace"
    if src_workdir_path.exists():
        # 如果 .src_workspace 文件存在，则读取源代码路径
        with open(src_workdir_path, "r") as file:
            src_name = file.read()
    else:
        # 否则，使用工作目录的名称作为源代码路径
        src_name = Path(workdir).name
    return Path(workdir) / src_name


async def init_python_folder(workdir: str | Path):
    # 初始化 Python 文件夹
    if not workdir:
        return
    workdir = Path(workdir)
    if not workdir.exists():
        return
    init_filename = Path(workdir) / "__init__.py"
    if init_filename.exists():
        return
    # 如果 __init__.py 文件不存在，则创建该文件
    async with aiofiles.open(init_filename, "a"):
        os.utime(init_filename, None)


def get_markdown_code_block_type(filename: str) -> str:
    # 根据文件扩展名返回相应的 Markdown 代码块类型
    if not filename:
        return ""
    ext = Path(filename).suffix
    # 文件扩展名与 Markdown 代码块类型的映射
    types = {
        ".py": "python",
        ".js": "javascript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".html": "html",
        ".css": "css",
        ".xml": "xml",
        ".json": "json",
        ".yaml": "yaml",
        ".md": "markdown",
        ".sql": "sql",
        ".rb": "ruby",
        ".php": "php",
        ".sh": "bash",
        ".swift": "swift",
        ".go": "go",
        ".rs": "rust",
        ".pl": "perl",
        ".asm": "assembly",
        ".r": "r",
        ".scss": "scss",
        ".sass": "sass",
        ".lua": "lua",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".yml": "yaml",
        ".ini": "ini",
        ".toml": "toml",
        ".svg": "xml",  # SVG 常被视为 XML 格式
        # 可以根据需要添加更多文件扩展名和相应的代码块类型
    }
    return types.get(ext, "")


def to_markdown_code_block(val: str, type_: str = "") -> str:
    """
    将字符串转换为 Markdown 代码块。

    该函数将输入的字符串包裹在 Markdown 代码块中。如果提供了类型参数，会将其作为语言标识符，进行语法高亮显示。

    参数：
        val (str): 要转换的字符串。
        type_ (str, 可选): 用于语法高亮的语言标识符。默认为空字符串。

    返回：
        str: 包裹在 Markdown 代码块中的输入字符串。
            如果输入字符串为空，返回空字符串。

    示例：
        >>> to_markdown_code_block("print('Hello, World!')", "python")
        \n```python\nprint('Hello, World!')\n```\n

        >>> to_markdown_code_block("Some text")
        \n```\nSome text\n```\n
    """
    if not val:
        return val or ""
    val = val.replace("```", "\\`\\`\\`")
    return f"\n```{type_}\n{val}\n```\n"


async def save_json_to_markdown(content: str, output_filename: str | Path):
    """
    将提供的 JSON 内容保存为 Markdown 文件。

    该函数将 JSON 字符串转换为 Markdown 格式，并将其写入指定的输出文件。

    参数：
        content (str): 要转换的 JSON 内容。
        output_filename (str 或 Path): 输出 Markdown 文件的路径。

    返回：
        None

    异常：
        None: 任何异常都会被记录并且函数不会抛出异常。

    示例：
        >>> await save_json_to_markdown('{"key": "value"}', Path("/path/to/output.md"))
        这将把转换后的 Markdown 格式的 JSON 内容保存到指定的文件。

    注意：
        - 该函数专门处理 `json.JSONDecodeError` 异常，用于 JSON 解析错误。
        - 其他过程中发生的异常也会被记录并优雅地处理。
    """
    try:
        m = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 内容解码失败: {e}")
        return
    except Exception as e:
        logger.warning(f"发生意外错误: {e}")
        return
    await awrite(filename=output_filename, data=json_to_markdown(m))


def tool2name(cls, methods: List[str], entry) -> Dict[str, Any]:
    """
    生成类方法与给定入口的映射，类名作为前缀。

    参数：
        cls: 类，方法来自该类。
        methods (List[str]): 方法名称列表。
        entry (Any): 映射到每个方法的入口。

    返回：
        Dict[str, Any]: 一个字典，键是类方法的名称，值是给定的入口。
                        如果方法数量少于 2，则字典会包含一个条目，键为类名。

    示例：
        >>> class MyClass:
        >>>     pass
        >>>
        >>> tool2name(MyClass, ['method1', 'method2'], 'some_entry')
        {'MyClass.method1': 'some_entry', 'MyClass.method2': 'some_entry'}

        >>> tool2name(MyClass, ['method1'], 'some_entry')
        {'MyClass': 'some_entry', 'MyClass.method1': 'some_entry'}
    """
    class_name = cls.__name__
    mappings = {f"{class_name}.{i}": entry for i in methods}
    if len(mappings) < 2:
        mappings[class_name] = entry
    return mappings


def new_transaction_id(postfix_len=8) -> str:
    """
    基于当前时间戳和随机 UUID 生成一个新的唯一交易 ID。

    参数：
        postfix_len (int): 随机 UUID 后缀的长度，默认为 8。

    返回：
        str: 一个由时间戳和随机 UUID 组成的唯一交易 ID。
    """
    return datetime.now().strftime("%Y%m%d%H%M%ST") + uuid.uuid4().hex[0:postfix_len]


def log_time(method):
    """一个用于打印执行时长的装饰器。"""

    def before_call():
        start_time, cpu_start_time = time.perf_counter(), time.process_time()
        logger.info(f"[{method.__name__}] 开始执行时间: " f"{datetime.now().strftime('%Y-%m-%d %H:%m:%S')}")
        return start_time, cpu_start_time

    def after_call(start_time, cpu_start_time):
        end_time, cpu_end_time = time.perf_counter(), time.process_time()
        logger.info(
            f"[{method.__name__}] 执行结束。 "
            f"时间耗时: {end_time - start_time:.4} sec, CPU 耗时: {cpu_end_time - cpu_start_time:.4} sec"
        )

    @functools.wraps(method)
    def timeit_wrapper(*args, **kwargs):
        start_time, cpu_start_time = before_call()
        result = method(*args, **kwargs)
        after_call(start_time, cpu_start_time)
        return result

    @functools.wraps(method)
    async def timeit_wrapper_async(*args, **kwargs):
        start_time, cpu_start_time = before_call()
        result = await method(*args, **kwargs)
        after_call(start_time, cpu_start_time)
        return result

    return timeit_wrapper_async if iscoroutinefunction(method) else timeit_wrapper


async def check_http_endpoint(url: str, timeout: int = 3) -> bool:
    """
    检查 HTTP 端点的状态。

    参数：
        url (str): 要检查的 HTTP 端点 URL。
        timeout (int, 可选): HTTP 请求的超时秒数。默认为 3。

    返回：
        bool: 如果端点在线且返回 200 状态码，则返回 True；否则返回 False。
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as response:
                return response.status == 200
        except Exception as e:
            print(f"访问端点 {url} 时发生错误: {e}")
            return False


def rectify_pathname(path: Union[str, Path], default_filename: str) -> Path:
    """
    修正给定路径，确保路径有效。

    如果给定的 `path` 是目录，则创建该目录（如果不存在）并附加 `default_filename`。如果 `path` 是文件路径，则创建父目录（如果不存在）并返回该路径。

    参数：
        path (Union[str, Path]): 输入路径，可以是字符串或 `Path` 对象。
        default_filename (str): 如果 `path` 是目录，则使用的默认文件名。

    返回：
        Path: 修正后的输出路径。
    """
    output_pathname = Path(path)
    if output_pathname.is_dir():
        output_pathname.mkdir(parents=True, exist_ok=True)
        output_pathname = output_pathname / default_filename
    else:
        output_pathname.parent.mkdir(parents=True, exist_ok=True)
    return output_pathname

def generate_fingerprint(text: str) -> str:
    """
    为给定的文本生成指纹值。

    参数:
        text (str): 需要生成指纹的文本

    返回:
        str: 文本的指纹值
    """
    text_bytes = text.encode("utf-8")

    # 计算 SHA-256 哈希值
    sha256 = hashlib.sha256()
    sha256.update(text_bytes)
    fingerprint = sha256.hexdigest()

    return fingerprint


def download_model(file_url: str, target_folder: Path) -> Path:
    """
    从给定的 URL 下载模型文件并保存到目标文件夹。

    参数:
        file_url (str): 模型文件的 URL 地址
        target_folder (Path): 文件保存的目标文件夹路径

    返回:
        Path: 下载并保存的文件路径
    """
    file_name = file_url.split("/")[-1]  # 获取文件名
    file_path = target_folder.joinpath(f"{file_name}")  # 目标文件路径

    if not file_path.exists():  # 如果文件不存在，则进行下载
        file_path.mkdir(parents=True, exist_ok=True)  # 如果目录不存在，则创建目录
        try:
            response = requests.get(file_url, stream=True)  # 以流方式请求文件
            response.raise_for_status()  # 检查请求是否成功
            # 保存文件
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):  # 分块写入文件
                    f.write(chunk)
                logger.info(f"权重文件已下载并保存至 {file_path}")
        except requests.exceptions.HTTPError as err:
            logger.info(f"权重文件下载过程中发生错误: {err}")
    return file_path
