#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : refs to openai 0.x sdk

import asyncio
import json
import os
import platform
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
)
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import requests

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import logging

import openai
from openai import version

logger = logging.getLogger("openai")

TIMEOUT_SECS = 600  # 超时秒数
MAX_SESSION_LIFETIME_SECS = 180  # 最大会话生命周期（秒）
MAX_CONNECTION_RETRIES = 2  # 最大连接重试次数

# 每个线程有一个属性，'session'。
_thread_context = threading.local()

LLM_LOG = os.environ.get("LLM_LOG", "debug")  # 从环境变量获取日志级别，默认为 "debug"


class ApiType(Enum):
    AZURE = 1  # Azure API类型
    OPEN_AI = 2  # OpenAI API类型
    AZURE_AD = 3  # Azure AD API类型

    @staticmethod
    def from_str(label):
        # 将字符串标签转换为 ApiType 枚举类型
        if label.lower() == "azure":
            return ApiType.AZURE
        elif label.lower() in ("azure_ad", "azuread"):
            return ApiType.AZURE_AD
        elif label.lower() in ("open_ai", "openai"):
            return ApiType.OPEN_AI
        else:
            raise openai.OpenAIError(
                "提供的API类型无效。请选择支持的API类型：'azure', 'azure_ad', 'open_ai'"
            )


# 根据不同的 API 类型，选择合适的 API 密钥头
api_key_to_header = (
    lambda api, key: {"Authorization": f"Bearer {key}"}
    if api in (ApiType.OPEN_AI, ApiType.AZURE_AD)
    else {"api-key": f"{key}"}
)


def _console_log_level():
    # 返回控制台日志级别
    if LLM_LOG in ["debug", "info"]:
        return LLM_LOG
    else:
        return None


def log_debug(message, **params):
    # 打印调试日志
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() == "debug":
        print(msg, file=sys.stderr)
    logger.debug(msg)


def log_info(message, **params):
    # 打印信息日志
    msg = logfmt(dict(message=message, **params))
    if _console_log_level() in ["debug", "info"]:
        print(msg, file=sys.stderr)
    logger.info(msg)


def log_warn(message, **params):
    # 打印警告日志
    msg = logfmt(dict(message=message, **params))
    print(msg, file=sys.stderr)
    logger.warning(msg)


def logfmt(props):
    # 格式化日志输出
    def fmt(key, val):
        # 处理值为字节或字节数组的情况
        if hasattr(val, "decode"):
            val = val.decode("utf-8")
        # 检查 val 是否已经是字符串，以避免重复编码
        if not isinstance(val, str):
            val = str(val)
        if re.search(r"\s", val):
            val = repr(val)
        # 如果 key 包含空格，需要将其转换为字符串表示
        if re.search(r"\s", key):
            key = repr(key)
        return "{key}={val}".format(key=key, val=val)

    return " ".join([fmt(key, val) for key, val in sorted(props.items())])


class OpenAIResponse:
    def __init__(self, data: Union[bytes, Any], headers: dict):
        self._headers = headers
        self.data = data

    @property
    def request_id(self) -> Optional[str]:
        return self._headers.get("request-id")

    @property
    def retry_after(self) -> Optional[int]:
        try:
            return int(self._headers.get("retry-after"))
        except TypeError:
            return None

    @property
    def operation_location(self) -> Optional[str]:
        return self._headers.get("operation-location")

    @property
    def organization(self) -> Optional[str]:
        return self._headers.get("LLM-Organization")

    @property
    def response_ms(self) -> Optional[int]:
        h = self._headers.get("Openai-Processing-Ms")
        return None if h is None else round(float(h))

    def decode_asjson(self) -> Optional[dict]:
        # 解码响应数据为 JSON 格式
        bstr = self.data.strip()
        if bstr.startswith(b"{") and bstr.endswith(b"}"):
            bstr = bstr.decode("utf-8")
        else:
            bstr = parse_stream_helper(bstr)
        return json.loads(bstr) if bstr else None


def _build_api_url(url, query):
    # 构建 API URL，合并查询参数
    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = "%s&%s" % (base_query, query)

    return urlunsplit((scheme, netloc, path, query, fragment))


def _requests_proxies_arg(proxy) -> Optional[Dict[str, str]]:
    """返回适用于 'requests.request' 的 'proxies' 参数"""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http": proxy, "https": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise ValueError(
            "'openai.proxy' 必须指定为字符串 URL 或字典形式，字典中包含 https 和/或 http 键。"
        )


def _aiohttp_proxies_arg(proxy) -> Optional[str]:
    """返回适用于 'aiohttp.ClientSession.request' 的 'proxies' 参数"""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return proxy
    elif isinstance(proxy, dict):
        return proxy["https"] if "https" in proxy else proxy["http"]
    else:
        raise ValueError(
            "'openai.proxy' 必须指定为字符串 URL 或字典形式，字典中包含 https 和/或 http 键。"
        )


def _make_session() -> requests.Session:
    # 创建一个新的 HTTP 会话
    s = requests.Session()
    s.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=MAX_CONNECTION_RETRIES),
    )
    return s


def parse_stream_helper(line: bytes) -> Optional[str]:
    # 解析数据流中的一行
    if line:
        if line.strip() == b"data: [DONE]":
            # 如果数据流结束，返回 None
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: ") :]
            return line.decode("utf-8")
        else:
            return None
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    # 解析整个数据流，逐行处理
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


async def parse_stream_async(rbody: aiohttp.StreamReader):
    # 异步解析流数据
    async for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


class APIRequestor:
    def __init__(
        self,
        key=None,
        base_url=None,
        api_type=None,
        api_version=None,
        organization=None,
    ):
        # 初始化 API 请求对象
        self.base_url = base_url or openai.base_url  # 设置基础 URL，默认使用 OpenAI 的 URL
        self.api_key = key or openai.api_key  # 设置 API 密钥
        self.api_type = ApiType.from_str(api_type) if api_type else ApiType.from_str("openai")  # 设置 API 类型
        self.api_version = api_version or openai.api_version  # 设置 API 版本
        self.organization = organization or openai.organization  # 设置组织信息

    @overload
    def request(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        # 重载方法：处理请求，返回一个生成器
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[OpenAIResponse, bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        pass

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        # 处理实际的 API 请求并返回响应
        result = self.request_raw(
            method.lower(),
            url,
            params=params,
            supplied_headers=headers,
            files=files,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool, str]:
        # 异步请求重载方法
        pass

    async def arequest(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool, str]:
        # 处理实际的异步 API 请求并返回响应
        ctx = aiohttp_session()  # 获取 aiohttp 会话
        session = await ctx.__aenter__()  # 异步打开会话
        try:
            result = await self.arequest_raw(
                method.lower(),
                url,
                session,
                params=params,
                supplied_headers=headers,
                files=files,
                request_id=request_id,
                request_timeout=request_timeout,
            )
            resp, got_stream = await self._interpret_async_response(result, stream)
        except Exception:
            await ctx.__aexit__(None, None, None)  # 异常处理，关闭会话
            raise
        if got_stream:

            async def wrap_resp():
                # 包装生成器，确保流式响应正确处理
                assert isinstance(resp, AsyncGenerator)
                try:
                    async for r in resp:
                        yield r
                finally:
                    await ctx.__aexit__(None, None, None)

            return wrap_resp(), got_stream, self.api_key
        else:
            await ctx.__aexit__(None, None, None)  # 关闭会话
            return resp, got_stream, self.api_key

    def request_headers(self, method: str, extra, request_id: Optional[str]) -> Dict[str, str]:
        """
        准备请求头部信息。
        :param method: 请求方法（GET, POST, etc.）
        :param extra: 额外的请求头
        :param request_id: 请求的 ID
        :return: 准备好的请求头部
        """
        # 设置默认的 User-Agent
        user_agent = "LLM/v1 PythonBindings/%s" % (version.VERSION,)

        # 获取当前平台的基本信息
        uname_without_node = " ".join(v for k, v in platform.uname()._asdict().items() if k != "node")

        # 设置 User-Agent 的详细信息
        ua = {
            "bindings_version": version.VERSION,
            "httplib": "requests",
            "lang": "python",
            "lang_version": platform.python_version(),
            "platform": platform.platform(),
            "publisher": "openai",
            "uname": uname_without_node,
        }

        # 设置请求头
        headers = {
            "X-LLM-Client-User-Agent": json.dumps(ua),
            "User-Agent": user_agent,
        }

        # 如果存在 API 密钥，则将其添加到请求头
        if self.api_key:
            headers.update(api_key_to_header(self.api_type, self.api_key))

        # 如果存在组织 ID，则将其添加到请求头
        if self.organization:
            headers["LLM-Organization"] = self.organization

        # 如果 API 版本存在且为 OpenAI 类型，则将版本添加到请求头
        if self.api_version is not None and self.api_type == ApiType.OPEN_AI:
            headers["LLM-Version"] = self.api_version

        # 如果请求 ID 存在，将其添加到请求头
        if request_id is not None:
            headers["X-Request-Id"] = request_id

        # 合并额外的请求头
        headers.update(extra)

        return headers

    def _validate_headers(self, supplied_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        验证请求头部信息的有效性。
        :param supplied_headers: 用户传入的请求头
        :return: 验证后的请求头
        """
        headers: Dict[str, str] = {}
        if supplied_headers is None:
            return headers

        if not isinstance(supplied_headers, dict):
            raise TypeError("Headers must be a dictionary")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings")
            headers[k] = v

        return headers

    def _prepare_request_raw(
            self,
            url,
            supplied_headers,
            method,
            params,
            files,
            request_id: Optional[str],
    ) -> Tuple[str, Dict[str, str], Optional[bytes]]:
        """
        准备原始请求的数据（URL、头部、数据）。
        :param url: 请求的 URL
        :param supplied_headers: 用户提供的请求头
        :param method: 请求方法（GET, POST, etc.）
        :param params: 请求参数
        :param files: 上传的文件
        :param request_id: 请求 ID
        :return: 准备好的 URL、头部和数据
        """
        abs_url = "%s%s" % (self.base_url, url)
        headers = self._validate_headers(supplied_headers)

        data = None
        if method == "get" or method == "delete":
            if params:
                encoded_params = urlencode([(k, v) for k, v in params.items() if v is not None])
                abs_url = _build_api_url(abs_url, encoded_params)
        elif method in {"post", "put"}:
            if params and files:
                data = params
            if params and not files:
                data = json.dumps(params).encode()
                headers["Content-Type"] = "application/json"
        else:
            raise openai.APIConnectionError(
                message=f"Unrecognized HTTP method {method}. This may indicate a bug in the LLM bindings.",
                request=None,
            )

        headers = self.request_headers(method, headers, request_id)

        return abs_url, headers, data

    def request_raw(
            self,
            method,
            url,
            *,
            params=None,
            supplied_headers: Optional[Dict[str, str]] = None,
            files=None,
            stream: bool = False,
            request_id: Optional[str] = None,
            request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> requests.Response:
        """
        发送原始的 HTTP 请求。
        :param method: 请求方法（GET, POST, etc.）
        :param url: 请求的 URL
        :param params: 请求参数
        :param supplied_headers: 用户提供的请求头
        :param files: 上传的文件
        :param stream: 是否需要流式响应
        :param request_id: 请求 ID
        :param request_timeout: 请求超时设置
        :return: 请求响应对象
        """
        abs_url, headers, data = self._prepare_request_raw(url, supplied_headers, method, params, files, request_id)

        if not hasattr(_thread_context, "session"):
            _thread_context.session = _make_session()
            _thread_context.session_create_time = time.time()
        elif time.time() - getattr(_thread_context, "session_create_time", 0) >= MAX_SESSION_LIFETIME_SECS:
            _thread_context.session.close()
            _thread_context.session = _make_session()
            _thread_context.session_create_time = time.time()

        try:
            result = _thread_context.session.request(
                method,
                abs_url,
                headers=headers,
                data=data,
                files=files,
                stream=stream,
                timeout=request_timeout if request_timeout else TIMEOUT_SECS,
                proxies=_thread_context.session.proxies,
            )
        except requests.exceptions.Timeout as e:
            raise openai.APITimeoutError("Request timed out: {}".format(e)) from e
        except requests.exceptions.RequestException as e:
            raise openai.APIConnectionError(message="Error communicating with LLM: {}".format(e),
                                            request=None) from e

        return result

    async def arequest_raw(
            self,
            method,
            url,
            session,
            *,
            params=None,
            supplied_headers: Optional[Dict[str, str]] = None,
            files=None,
            request_id: Optional[str] = None,
            request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> aiohttp.ClientResponse:
        """
        异步发送原始 HTTP 请求。
        :param method: 请求方法（GET, POST, etc.）
        :param url: 请求的 URL
        :param session: aiohttp 客户端会话
        :param params: 请求参数
        :param supplied_headers: 用户提供的请求头
        :param files: 上传的文件
        :param request_id: 请求 ID
        :param request_timeout: 请求超时设置
        :return: 请求响应对象
        """
        abs_url, headers, data = self._prepare_request_raw(url, supplied_headers, method, params, files, request_id)

        if isinstance(request_timeout, tuple):
            timeout = aiohttp.ClientTimeout(
                connect=request_timeout[0],
                total=request_timeout[1],
            )
        else:
            timeout = aiohttp.ClientTimeout(total=request_timeout or TIMEOUT_SECS)

        if files:
            # TODO: 使用 `aiohttp.MultipartWriter` 来创建 multipart 数据。
            # 当前使用了 requests 中的私有方法，这个方法已知可行。
            data, content_type = requests.models.RequestEncodingMixin._encode_files(files, data)  # type: ignore
            headers["Content-Type"] = content_type

        request_kwargs = {
            "method": method,
            "url": abs_url,
            "headers": headers,
            "data": data,
            "timeout": timeout,
        }

        try:
            result = await session.request(**request_kwargs)
            return result
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise openai.APITimeoutError("Request timed out") from e
        except aiohttp.ClientError as e:
            raise openai.APIConnectionError(message="Error communicating with LLM", request=None) from e

    def _interpret_response(
            self, result: requests.Response, stream: bool
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]:
        """
        解释 HTTP 响应，返回响应内容和流的标志。
        :param result: 响应对象
        :param stream: 是否为流式响应
        :return: 响应内容和流标志
        """
        pass

    async def _interpret_async_response(
            self, result: aiohttp.ClientResponse, stream: bool
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool]:
        """
        异步解释 HTTP 响应，返回响应内容和流的标志。
        :param result: 响应对象
        :param stream: 是否为流式响应
        :return: 响应内容和流标志
        """
        pass

    def _interpret_response_line(self, rbody: str, rcode: int, rheaders, stream: bool) -> OpenAIResponse:
        """
        解释单行响应内容。
        :param rbody: 响应体内容
        :param rcode: 响应码
        :param rheaders: 响应头
        :param stream: 是否为流式响应
        :return: 解释后的响应
        """
        pass


@asynccontextmanager
async def aiohttp_session() -> AsyncIterator[aiohttp.ClientSession]:
    """
    创建一个异步上下文管理器，用于管理 aiohttp 客户端会话。
    这个上下文管理器会在执行完后自动关闭会话。

    使用方法：
        使用 `async with` 语句来确保会话正确关闭。

    示例：
        async with aiohttp_session() as session:
            # 在此处使用 `session` 进行请求
            pass
    """
    # 使用 aiohttp.ClientSession 创建一个会话
    async with aiohttp.ClientSession() as session:
        # 在上下文管理器中返回会话
        yield session
