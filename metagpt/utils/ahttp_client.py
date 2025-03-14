#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : pure async http_client

from typing import Any, Mapping, Optional, Union

import aiohttp
from aiohttp.client import DEFAULT_TIMEOUT


async def apost(
    url: str,
    params: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    headers: Optional[dict] = None,
    as_json: bool = False,
    encoding: str = "utf-8",
    timeout: int = DEFAULT_TIMEOUT.total,
) -> Union[str, dict]:
    """
    发送异步 POST 请求。

    参数：
        url (str): 目标 URL。
        params (Optional[Mapping[str, str]]): 可选的 URL 查询参数，默认 None。
        json (Any): 可选的 JSON 数据，默认 None。
        data (Any): 可选的请求体数据，默认 None。
        headers (Optional[dict]): 可选的 HTTP 头部信息，默认 None。
        as_json (bool): 是否将响应解析为 JSON，默认 False。
        encoding (str): 响应的解码方式，默认 "utf-8"。
        timeout (int): 超时时间（秒），默认值为 DEFAULT_TIMEOUT。

    返回：
        Union[str, dict]: 如果 `as_json=True`，返回 JSON 对象，否则返回解码后的字符串。
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(url=url, params=params, json=json, data=data, headers=headers, timeout=timeout) as resp:
            if as_json:
                data = await resp.json()
            else:
                data = await resp.read()
                data = data.decode(encoding)
    return data


async def apost_stream(
    url: str,
    params: Optional[Mapping[str, str]] = None,
    json: Any = None,
    data: Any = None,
    headers: Optional[dict] = None,
    encoding: str = "utf-8",
    timeout: int = DEFAULT_TIMEOUT.total,
) -> Any:
    """
    发送异步 POST 请求，并以流式方式获取响应内容。

    用法：
        result = apost_stream(url="xx")
        async for line in result:
            deal_with(line)

    参数：
        url (str): 目标 URL。
        params (Optional[Mapping[str, str]]): 可选的 URL 查询参数，默认 None。
        json (Any): 可选的 JSON 数据，默认 None。
        data (Any): 可选的请求体数据，默认 None。
        headers (Optional[dict]): 可选的 HTTP 头部信息，默认 None。
        encoding (str): 响应的解码方式，默认 "utf-8"。
        timeout (int): 超时时间（秒），默认值为 DEFAULT_TIMEOUT。

    返回：
        Any: 以流式方式返回每一行解码后的响应数据。
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(url=url, params=params, json=json, data=data, headers=headers, timeout=timeout) as resp:
            async for line in resp.content:
                yield line.decode(encoding)
