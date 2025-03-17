#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : General Async API for http-based LLM model

import asyncio
from typing import AsyncGenerator, Iterator, Optional, Tuple, Union

import aiohttp
import requests

from metagpt.logs import logger
from metagpt.provider.general_api_base import APIRequestor, OpenAIResponse


def parse_stream_helper(line: bytes) -> Optional[bytes]:
    """
    解析流中的一行数据，返回处理后的数据。

    参数:
        line (bytes): 输入的字节行数据。

    返回:
        Optional[bytes]: 如果行有效且不是结束标志，则返回数据字节；如果是结束标志则返回 None；否则返回 None。
    """
    if line and line.startswith(b"data:"):
        if line.startswith(b"data: "):
            # 如果以 "data: " 开头，去掉前缀并保留后续内容
            line = line[len(b"data: ") :]
        else:
            # 如果以 "data:" 开头，去掉前缀并保留后续内容
            line = line[len(b"data:") :]
        if line.strip() == b"[DONE]":
            # 如果行内容是 "[DONE]"，表示流结束，返回 None
            return None
        else:
            return line
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[bytes]:
    """
    解析流的每一行数据，并返回有效的数据。

    参数:
        rbody (Iterator[bytes]): 响应体的字节流。

    返回:
        Iterator[bytes]: 生成有效的数据行。
    """
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


class GeneralAPIRequestor(APIRequestor):
    """
    通用API请求类，用于发起和处理API请求。

    使用示例:
        # full_url = "{base_url}{url}"
        requester = GeneralAPIRequestor(base_url=base_url)
        result, _, api_key = await requester.arequest(
            method=method,
            url=url,
            headers=headers,
            stream=stream,
            params=kwargs,
            request_timeout=120
        )
    """

    def _interpret_response_line(self, rbody: bytes, rcode: int, rheaders: dict, stream: bool) -> OpenAIResponse:
        """
        处理并返回响应数据，封装成 OpenAIResponse 对象。

        参数:
            rbody (bytes): 响应体内容。
            rcode (int): 响应状态码。
            rheaders (dict): 响应头部。
            stream (bool): 是否为流响应。

        返回:
            OpenAIResponse: 封装后的响应数据。
        """
        return OpenAIResponse(rbody, rheaders)

    def _interpret_response(
        self, result: requests.Response, stream: bool
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]:
        """
        解析同步响应。

        参数:
            result (requests.Response): 响应对象。
            stream (bool): 是否为流响应。

        返回:
            Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]: 响应内容和是否为流的标志。
        """
        content_type = result.headers.get("Content-Type", "")
        if stream and ("text/event-stream" in content_type or "application/x-ndjson" in content_type):
            return (
                (
                    self._interpret_response_line(line, result.status_code, result.headers, stream=True)
                    for line in parse_stream(result.iter_lines())
                ),
                True,
            )
        else:
            return (
                self._interpret_response_line(
                    result.content,  # 让调用者解码消息
                    result.status_code,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    async def _interpret_async_response(
        self, result: aiohttp.ClientResponse, stream: bool
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool]:
        """
        解析异步响应。

        参数:
            result (aiohttp.ClientResponse): 响应对象。
            stream (bool): 是否为流响应。

        返回:
            Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool]: 响应内容和是否为流的标志。
        """
        content_type = result.headers.get("Content-Type", "")
        if stream and (
            "text/event-stream" in content_type or "application/x-ndjson" in content_type or content_type == ""
        ):
            return (
                (
                    self._interpret_response_line(line, result.status, result.headers, stream=True)
                    async for line in result.content
                ),
                True,
            )
        else:
            try:
                response_content = await result.read()
            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                raise TimeoutError("请求超时") from e
            except aiohttp.ClientError as exp:
                logger.warning(f"响应: {result}, 异常: {exp}")
                response_content = b""
            return (
                self._interpret_response_line(
                    response_content,  # 让调用者解码消息
                    result.status,
                    result.headers,
                    stream=False,
                ),
                False,
            )
