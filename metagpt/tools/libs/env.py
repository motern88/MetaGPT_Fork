#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/25
@Author  : mashenquan
@File    : env.py
@Desc: Implement `get_env`. RFC 216 2.4.2.4.2.
"""
import os
from typing import Dict, Optional


class EnvKeyNotFoundError(Exception):
    def __init__(self, info):
        super().__init__(info)


def to_app_key(key: str, app_name: str = None) -> str:
    """
    将环境变量的键转换为与应用名称关联的键。

    参数：
        key (str): 环境变量的键。
        app_name (str, 可选): 应用的名称，如果没有提供则默认为 None。

    返回：
        str: 生成的应用相关环境变量键（格式为 "应用名-键"）。
    """
    return f"{app_name}-{key}" if app_name else key


def split_app_key(app_key: str) -> (str, str):
    """
    将应用相关的键拆分为应用名称和键。

    参数：
        app_key (str): 需要拆分的应用相关环境变量键。

    返回：
        tuple: 应用名称和环境变量的键。如果没有应用名称，则应用名称为空字符串。
    """
    if "-" not in app_key:
        return "", app_key
    app_name, key = app_key.split("-", 1)
    return app_name, key


async def default_get_env(key: str, app_name: str = None) -> str:
    """
    获取指定环境变量键的值。首先检查环境变量中是否存在该键，若没有，则检查 Context 对象中是否存在。

    参数：
        key (str): 环境变量的键。
        app_name (str, 可选): 应用的名称，如果没有提供则默认为 None。

    返回：
        str: 环境变量的值。

    异常：
        如果环境变量或 Context 中没有找到该键，将抛出 `EnvKeyNotFoundError` 异常。
    """
    app_key = to_app_key(key=key, app_name=app_name)
    if app_key in os.environ:
        return os.environ[app_key]

    # 替换 "-" 为 "_"，因为 Linux 环境变量不支持 "-" 字符
    env_app_key = app_key.replace("-", "_")
    if env_app_key in os.environ:
        return os.environ[env_app_key]

    from metagpt.context import Context

    context = Context()
    val = context.kwargs.get(app_key, None)
    if val is not None:
        return val

    raise EnvKeyNotFoundError(f"EnvKeyNotFoundError: {key}, app_name:{app_name or ''}")


async def default_get_env_description() -> Dict[str, str]:
    """
    获取所有环境变量的描述。

    返回：
        dict: 一个字典，包含环境变量的描述信息，键是调用 `get_env` 函数的示例，值是该环境变量的描述。
    """
    result = {}
    for k in os.environ.keys():
        app_name, key = split_app_key(k)
        call = f'await get_env(key="{key}", app_name="{app_name}")'
        result[call] = f"返回环境变量 `{k}` 的值。"

    from metagpt.context import Context

    context = Context()
    for k in context.kwargs.__dict__.keys():
        app_name, key = split_app_key(k)
        call = f'await get_env(key="{key}", app_name="{app_name}")'
        result[call] = f"获取环境变量 `{k}` 的值。"
    return result


_get_env_entry = default_get_env
_get_env_description_entry = default_get_env_description


async def get_env(key: str, app_name: str = None) -> str:
    """
    异步获取指定的环境变量的值。

    参数：
        key (str): 环境变量的键。
        app_name (str, 可选): 应用的名称，如果没有提供则默认为 None。

    返回：
        str: 对应的环境变量值。

    示例：
        >>> api_key = await get_env("API_KEY")
        >>> print(api_key)
        <API_KEY>

    注意：
        该函数是异步的，必须使用 `await` 调用。
    """
    global _get_env_entry
    if _get_env_entry:
        return await _get_env_entry(key=key, app_name=app_name)

    return await default_get_env(key=key, app_name=app_name)


async def get_env_default(key: str, app_name: str = None, default_value: str = None) -> Optional[str]:
    """
    获取指定的环境变量的值。如果没有找到该环境变量，则返回默认值。

    参数：
        key (str): 环境变量的键。
        app_name (str, 可选): 应用的名称。
        default_value (str, 可选): 如果没有找到环境变量，则返回的默认值。

    返回：
        str or None: 环境变量的值，或者是默认值。

    示例：
        >>> api_key = await get_env_default(key="NOT_EXISTS_API_KEY", default_value="<API_KEY>")
        >>> print(api_key)
        <API_KEY>
    """
    try:
        return await get_env(key=key, app_name=app_name)
    except EnvKeyNotFoundError:
        return default_value


async def get_env_description() -> Dict[str, str]:
    """
    获取所有环境变量及其描述的字典。

    返回：
        dict: 环境变量描述信息的字典。
    """
    global _get_env_description_entry

    if _get_env_description_entry:
        return await _get_env_description_entry()

    return await default_get_env_description()


def set_get_env_entry(value, description):
    """
    修改 `get_env` 和 `get_env_description` 的实现。

    参数：
        value (function): 新的 `get_env` 实现函数。
        description (str): `get_env` 实现的描述。
    """
    global _get_env_entry
    global _get_env_description_entry
    _get_env_entry = value
    _get_env_description_entry = description
