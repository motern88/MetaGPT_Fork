#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import warnings
from typing import Optional

import aiohttp
from pydantic import BaseModel, ConfigDict, model_validator


class BingAPIWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str  # 用于认证的API密钥
    bing_url: str = "https://api.bing.microsoft.com/v7.0/search"  # Bing搜索API的URL
    aiosession: Optional[aiohttp.ClientSession] = None  # aiohttp会话，用于异步请求
    proxy: Optional[str] = None  # 可选的代理设置

    @model_validator(mode="before")
    @classmethod
    def validate_api_key(cls, values: dict) -> dict:
        """验证API密钥，若没有则警告并使用默认值"""
        if "api_key" in values:
            values.setdefault("api_key", values["api_key"])  # 默认值设置
            warnings.warn("`api_key` is deprecated, use `api_key` instead", DeprecationWarning, stacklevel=2)  # 发出API密钥弃用的警告
        return values

    @property
    def header(self):
        """设置请求头，使用API密钥进行认证"""
        return {"Ocp-Apim-Subscription-Key": self.api_key}

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
        focus: list[str] | None = None,
    ) -> str | list[dict]:
        """使用官方Bing API进行搜索，并返回结果。

        参数：
            query: 搜索查询词。
            max_results: 返回的结果数量，默认为8。
            as_string: 是否以字符串形式返回结果。如果为True，返回格式化的字符串。如果为False，返回包含详细信息的字典列表。
            focus: 可选，指定关注的搜索结果字段（例如：snippet、link、title）。

        返回：
            返回搜索结果，类型可以是字符串或字典列表。
        """
        params = {
            "q": query,  # 搜索查询
            "count": max_results,  # 返回的最大结果数
            "textFormat": "HTML",  # 设置返回格式为HTML
        }
        result = await self.results(params)  # 获取搜索结果
        search_results = result["webPages"]["value"]  # 获取网页搜索结果
        focus = focus or ["snippet", "link", "title"]  # 默认关注“snippet”, “link”, “title”字段
        for item_dict in search_results:
            item_dict["link"] = item_dict["url"]  # 将链接字段重命名为“link”
            item_dict["title"] = item_dict["name"]  # 将标题字段重命名为“title”
        details = [{i: j for i, j in item_dict.items() if i in focus} for item_dict in search_results]  # 根据focus筛选字段
        if as_string:
            return safe_results(details)  # 如果需要字符串格式，使用safe_results函数
        return details

    async def results(self, params: dict) -> dict:
        """使用aiohttp异步请求获取搜索结果。

        参数：
            params: 请求参数字典。

        返回：
            返回的JSON格式的搜索结果。
        """
        if not self.aiosession:
            # 如果没有现成的会话，则创建一个新的会话进行请求
            async with aiohttp.ClientSession() as session:
                async with session.get(self.bing_url, params=params, headers=self.header, proxy=self.proxy) as response:
                    response.raise_for_status()  # 如果请求失败，则抛出异常
                    res = await response.json()  # 解析响应的JSON数据
        else:
            # 如果已有会话，则复用现有会话进行请求
            async with self.aiosession.get(
                self.bing_url, params=params, headers=self.header, proxy=self.proxy
            ) as response:
                response.raise_for_status()  # 如果请求失败，则抛出异常
                res = await response.json()  # 解析响应的JSON数据

        return res

def safe_results(results: str | list) -> str:
    """以安全格式返回Bing搜索结果。

    参数：
        results: 搜索结果，可以是字符串或字典列表。

    返回：
        格式化后的搜索结果字符串。
    """
    if isinstance(results, list):
        safe_message = json.dumps([result for result in results])  # 将字典列表转换为JSON格式的字符串
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")  # 处理字符串中的非UTF-8字符
    return safe_message

if __name__ == "__main__":
    import fire

    fire.Fire(BingAPIWrapper().run)  # 使用fire库创建命令行接口，调用run方法
