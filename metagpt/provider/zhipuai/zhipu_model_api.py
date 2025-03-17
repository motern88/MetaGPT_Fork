#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : zhipu model api to support sync & async for invoke & sse_invoke

import json

from zhipuai import ZhipuAI
from zhipuai.core._http_client import ZHIPUAI_DEFAULT_TIMEOUT

from metagpt.provider.general_api_requestor import GeneralAPIRequestor
from metagpt.provider.zhipuai.async_sse_client import AsyncSSEClient


class ZhiPuModelAPI(ZhipuAI):
    def split_zhipu_api_url(self):
        """
        用于分割 Zhipu API 的 URL，以防止 Zhipu API 升级导致版本变化。
        该方法返回基础 API URL 和具体的 API 路径。

        返回：
            str: 基础 URL 和 API 路径的分割字符串
        """
        # 使用固定的 Zhipu API URL
        zhipu_api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        arr = zhipu_api_url.split("/api/")
        # 返回基础 URL 和 API 路径
        return f"{arr[0]}/api", f"/{arr[1]}"

    async def arequest(self, stream: bool, method: str, headers: dict, kwargs):
        """
        异步请求方法，支持不同 HTTP 方法（POST 和 GET）。

        参数：
            stream (bool): 是否使用流式请求
            method (str): HTTP 请求方法，支持 "post" 和 "get"
            headers (dict): 请求头
            kwargs (dict): 请求参数

        返回：
            dict: 请求结果
        """
        # 确保请求方法是支持的
        assert method in ["post", "get"]

        # 获取分割后的基础 URL 和 API 路径
        base_url, url = self.split_zhipu_api_url()

        # 创建 API 请求对象
        requester = GeneralAPIRequestor(base_url=base_url)

        # 异步发起请求并返回结果
        result, _, api_key = await requester.arequest(
            method=method,
            url=url,
            headers=headers,
            stream=stream,
            params=kwargs,
            request_timeout=ZHIPUAI_DEFAULT_TIMEOUT.read,
        )
        return result

    async def acreate(self, **kwargs) -> dict:
        """
        异步调用方法，直接获取最终结果。

        该方法与原始方法 `async_invoke` 不同，`async_invoke` 会通过 task_id 获取最终结果。

        参数：
            kwargs (dict): 请求参数

        返回：
            dict: 返回的请求结果

        异常：
            - 如果返回数据包含错误信息，则抛出 `RuntimeError`。
        """
        headers = self._default_headers

        # 发起异步请求并获取结果
        resp = await self.arequest(stream=False, method="post", headers=headers, kwargs=kwargs)

        # 解码并加载响应数据
        resp = resp.data.decode("utf-8")
        resp = json.loads(resp)

        # 如果响应中包含错误，抛出异常
        if "error" in resp:
            raise RuntimeError(
                f"请求失败，错误信息: {resp}, 请参考 `https://open.bigmodel.cn/dev/api#error-code-v3`"
            )
        return resp

    async def acreate_stream(self, **kwargs) -> AsyncSSEClient:
        """
        异步流式请求方法，使用 SSE（Server-Sent Events）获取结果。

        参数：
            kwargs (dict): 请求参数

        返回：
            AsyncSSEClient: 返回一个 `AsyncSSEClient` 实例，用于处理流式数据。
        """
        headers = self._default_headers

        # 返回一个包含流式数据的异步客户端
        return AsyncSSEClient(await self.arequest(stream=True, method="post", headers=headers, kwargs=kwargs))