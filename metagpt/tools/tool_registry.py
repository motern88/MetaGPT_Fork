#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/01/12 17:07
@Author  : garylin2099
@File    : tool_registry.py
"""
from __future__ import annotations

import contextlib
import inspect
import os
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from metagpt.const import TOOL_SCHEMA_PATH
from metagpt.logs import logger
from metagpt.tools.tool_convert import (
    convert_code_to_tool_schema,
    convert_code_to_tool_schema_ast,
)
from metagpt.tools.tool_data_type import Tool, ToolSchema


class ToolRegistry(BaseModel):
    tools: dict = {}  # 工具字典，用于存储注册的所有工具
    tools_by_tags: dict = defaultdict(dict)  # 两层字典，用于存储工具标签：{tag: {tool_name: {...}, ...}, ...}

    def register_tool(
        self,
        tool_name: str,
        tool_path: str,
        schemas: dict = None,
        schema_path: str = "",
        tool_code: str = "",
        tags: list[str] = None,
        tool_source_object=None,  # 工具源对象，可以是任何类或函数
        include_functions: list[str] = None,  # 包含的函数列表
        verbose: bool = False,  # 是否打印详细信息
    ):
        """
        注册工具到工具注册表中。
        参数:
            tool_name (str): 工具的名称。
            tool_path (str): 工具代码文件的路径。
            schemas (dict): 工具的架构，包含工具的描述等。
            schema_path (str): 架构文件路径。
            tool_code (str): 工具的源代码。
            tags (list[str]): 工具的标签。
            tool_source_object: 工具的源对象，通常是类或函数。
            include_functions (list[str]): 需要包含的函数列表。
            verbose (bool): 是否输出注册信息。
        """
        if self.has_tool(tool_name):  # 如果工具已注册，直接返回
            return

        schema_path = schema_path or TOOL_SCHEMA_PATH / f"{tool_name}.yml"  # 默认的架构路径

        if not schemas:
            schemas = make_schema(tool_source_object, include_functions, schema_path)

        if not schemas:
            return

        schemas["tool_path"] = tool_path  # 将工具代码文件路径加入架构中
        try:
            ToolSchema(**schemas)  # 校验架构
        except Exception:
            pass  # 如果架构无效，忽略

        tags = tags or []
        tool = Tool(name=tool_name, path=tool_path, schemas=schemas, code=tool_code, tags=tags)  # 创建工具对象
        self.tools[tool_name] = tool  # 注册工具
        for tag in tags:
            self.tools_by_tags[tag].update({tool_name: tool})  # 将工具按标签分类

        if verbose:
            logger.info(f"{tool_name} 已注册")
            logger.info(f"架构文件创建于 {str(schema_path)}，可以用于校验")

    def has_tool(self, key: str) -> Tool:
        """
        检查工具注册表中是否存在指定的工具。
        """
        return key in self.tools

    def get_tool(self, key) -> Tool:
        """
        获取工具注册表中指定的工具。
        """
        return self.tools.get(key)

    def get_tools_by_tag(self, key) -> dict[str, Tool]:
        """
        根据标签获取工具。
        """
        return self.tools_by_tags.get(key, {})

    def get_all_tools(self) -> dict[str, Tool]:
        """
        获取所有工具。
        """
        return self.tools

    def has_tool_tag(self, key) -> bool:
        """
        检查是否存在指定的工具标签。
        """
        return key in self.tools_by_tags

    def get_tool_tags(self) -> list[str]:
        """
        获取所有的工具标签。
        """
        return list(self.tools_by_tags.keys())


# 工具注册表实例
TOOL_REGISTRY = ToolRegistry()


def register_tool(tags: list[str] = None, schema_path: str = "", **kwargs):
    """
    注册工具到工具注册表的装饰器函数。
    """
    def decorator(cls):
        # 获取工具类文件路径及源代码
        file_path = inspect.getfile(cls)
        if "metagpt" in file_path:
            # 处理路径，确保正确
            file_path = "metagpt" + file_path.split("metagpt")[-1]
        source_code = ""
        with contextlib.suppress(OSError):
            source_code = inspect.getsource(cls)

        # 调用工具注册函数
        TOOL_REGISTRY.register_tool(
            tool_name=cls.__name__,
            tool_path=file_path,
            schema_path=schema_path,
            tool_code=source_code,
            tags=tags,
            tool_source_object=cls,
            **kwargs,
        )
        return cls

    return decorator


def make_schema(tool_source_object, include, path):
    """
    生成工具架构的函数。
    """
    try:
        schema = convert_code_to_tool_schema(tool_source_object, include=include)
    except Exception as e:
        schema = {}
        logger.error(f"生成架构失败: {e}")

    return schema


def validate_tool_names(tools: list[str]) -> dict[str, Tool]:
    """
    验证工具名称并返回有效的工具。
    """
    assert isinstance(tools, list), "tools 必须是字符串列表"
    valid_tools = {}
    for key in tools:
        # 可以定义工具名、工具标签或工具路径，获取整个工具集
        if os.path.isdir(key) or os.path.isfile(key):
            valid_tools.update(register_tools_from_path(key))
        elif TOOL_REGISTRY.has_tool(key.split(":")[0]):
            if ":" in key:
                # 处理指定方法的类工具
                class_tool_name = key.split(":")[0]
                method_names = key.split(":")[1].split(",")
                class_tool = TOOL_REGISTRY.get_tool(class_tool_name)

                methods_filtered = {}
                for method_name in method_names:
                    if method_name in class_tool.schemas["methods"]:
                        methods_filtered[method_name] = class_tool.schemas["methods"][method_name]
                    else:
                        logger.warning(f"无效方法 {method_name} 在工具 {class_tool_name} 下，已跳过")
                class_tool_filtered = class_tool.model_copy(deep=True)
                class_tool_filtered.schemas["methods"] = methods_filtered

                valid_tools.update({class_tool_name: class_tool_filtered})

            else:
                valid_tools.update({key: TOOL_REGISTRY.get_tool(key)})
        elif TOOL_REGISTRY.has_tool_tag(key):
            valid_tools.update(TOOL_REGISTRY.get_tools_by_tag(key))
        else:
            logger.warning(f"无效的工具名或工具类型名: {key}, 已跳过")
    return valid_tools


def register_tools_from_file(file_path) -> dict[str, Tool]:
    """
    从文件注册工具。
    """
    file_name = Path(file_path).name
    if not file_name.endswith(".py") or file_name == "setup.py" or file_name.startswith("test"):
        return {}
    registered_tools = {}
    code = Path(file_path).read_text(encoding="utf-8")
    tool_schemas = convert_code_to_tool_schema_ast(code)
    for name, schemas in tool_schemas.items():
        tool_code = schemas.pop("code", "")
        TOOL_REGISTRY.register_tool(
            tool_name=name,
            tool_path=file_path,
            schemas=schemas,
            tool_code=tool_code,
        )
        registered_tools.update({name: TOOL_REGISTRY.get_tool(name)})
    return registered_tools


def register_tools_from_path(path) -> dict[str, Tool]:
    """
    从路径注册工具，可以是文件或目录。
    """
    tools_registered = {}
    if os.path.isfile(path):
        tools_registered.update(register_tools_from_file(path))
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                tools_registered.update(register_tools_from_file(file_path))
    return tools_registered
