#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import warnings
from concurrent import futures
from typing import Optional
from urllib.parse import urlparse

import httplib2
from pydantic import BaseModel, ConfigDict, model_validator

try:
    from googleapiclient.discovery import build  # 尝试导入Google API客户端
except ImportError:
    raise ImportError(
        "要使用此模块，您需要安装 `google-api-python-client` Python包。"
        "您可以运行命令：`pip install -e.[search-google]` 来安装该包"
    )  # 如果没有安装google-api-python-client库，抛出错误并给出安装提示


class GoogleAPIWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str  # Google API 密钥
    cse_id: str  # 自定义搜索引擎ID
    discovery_service_url: Optional[str] = None  # 可选的发现服务URL

    loop: Optional[asyncio.AbstractEventLoop] = None  # 可选的事件循环，用于异步任务
    executor: Optional[futures.Executor] = None  # 可选的执行器，用于在线程池中运行任务
    proxy: Optional[str] = None  # 可选的代理设置

    @model_validator(mode="before")
    @classmethod
    def validate_google(cls, values: dict) -> dict:
        """验证API密钥和自定义搜索引擎ID是否存在"""
        if "google_api_key" in values:
            values.setdefault("api_key", values["google_api_key"])
            warnings.warn("`google_api_key` 已过时，请使用 `api_key`", DeprecationWarning, stacklevel=2)

        if "api_key" not in values:
            raise ValueError(
                "要使用Google搜索引擎，必须在构造对象时提供 `api_key`。您可以从 https://console.cloud.google.com/apis/credentials 获取API密钥。"
            )

        if "google_cse_id" in values:
            values.setdefault("cse_id", values["google_cse_id"])
            warnings.warn("`google_cse_id` 已过时，请使用 `cse_id`", DeprecationWarning, stacklevel=2)

        if "cse_id" not in values:
            raise ValueError(
                "要使用Google搜索引擎，必须在构造对象时提供 `cse_id`。您可以从 https://programmablesearchengine.google.com/controlpanel/create 获取cse_id。"
            )
        return values

    @property
    def google_api_client(self):
        """返回Google API客户端实例"""
        build_kwargs = {"developerKey": self.api_key, "discoveryServiceUrl": self.discovery_service_url}
        if self.proxy:
            parse_result = urlparse(self.proxy)
            proxy_type = parse_result.scheme
            if proxy_type == "https":
                proxy_type = "http"
            build_kwargs["http"] = httplib2.Http(
                proxy_info=httplib2.ProxyInfo(
                    getattr(httplib2.socks, f"PROXY_TYPE_{proxy_type.upper()}"),
                    parse_result.hostname,
                    parse_result.port,
                ),
            )
        service = build("customsearch", "v1", **build_kwargs)
        return service.cse()  # 返回自定义搜索引擎实例

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
        focus: list[str] | None = None,
    ) -> str | list[dict]:
        """使用Google API返回搜索结果

        参数：
            query: 搜索查询字符串。
            max_results: 返回的最大结果数，默认为8。
            as_string: 布尔值，决定返回结果的格式。如果为True，返回格式化的字符串；如果为False，返回包含详细信息的字典列表。
            focus: 要从每个搜索结果中提取的特定信息。

        返回：
            搜索结果，可以是字符串或字典列表。
        """
        loop = self.loop or asyncio.get_event_loop()  # 获取事件循环
        future = loop.run_in_executor(
            self.executor, self.google_api_client.list(q=query, num=max_results, cx=self.cse_id).execute
        )
        result = await future  # 等待任务完成并获取结果
        # 从响应中提取搜索结果
        search_results = result.get("items", [])

        focus = focus or ["snippet", "link", "title"]  # 默认聚焦在摘要、链接和标题上
        details = [{i: j for i, j in item_dict.items() if i in focus} for item_dict in search_results]
        # 如果需要返回字符串格式的结果
        if as_string:
            return safe_google_results(details)

        return details  # 返回字典列表形式的结果


def safe_google_results(results: str | list) -> str:
    """以安全的格式返回Google搜索结果

    参数：
        results: 搜索结果。

    返回：
        搜索结果的安全格式。
    """
    if isinstance(results, list):
        safe_message = json.dumps([result for result in results])
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message


if __name__ == "__main__":
    import fire

    fire.Fire(GoogleAPIWrapper().run)  # 使用fire库创建命令行接口，调用run方法
