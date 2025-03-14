#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 18:27
@Author  : alexanderwu
@File    : search_engine_serpapi.py
"""
import json
import warnings
from typing import Any, Dict, Optional

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SerperWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str  # Serper API 的 API 密钥
    url: str = "https://google.serper.dev/search"  # Serper 搜索引擎的 URL
    payload: dict = Field(default_factory=lambda: {"page": 1, "num": 10})  # 默认请求参数
    aiosession: Optional[aiohttp.ClientSession] = None  # 可选的 aiohttp 会话
    proxy: Optional[str] = None  # 可选的代理

    @model_validator(mode="before")
    @classmethod
    def validate_serper(cls, values: dict) -> dict:
        """验证 Serper 配置，确保提供了有效的 API 密钥。"""
        if "serper_api_key" in values:
            values.setdefault("api_key", values["serper_api_key"])
            warnings.warn("`serper_api_key` 已弃用，请改用 `api_key`", DeprecationWarning, stacklevel=2)

        if "api_key" not in values:
            raise ValueError(
                "要使用 Serper 搜索引擎，请确保在构造对象时提供 `api_key`。你可以从 https://serper.dev/ 获取 API 密钥。"
            )

        return values

    async def run(self, query: str, max_results: int = 8, as_string: bool = True, **kwargs: Any) -> str:
        """异步运行查询并解析结果。"""
        if isinstance(query, str):
            return self._process_response((await self.results([query], max_results))[0], as_string=as_string)
        else:
            results = [self._process_response(res, as_string) for res in await self.results(query, max_results)]
        return "\n".join(results) if as_string else results

    async def results(self, queries: list[str], max_results: int = 8) -> dict:
        """使用 aiohttp 通过 Serper 执行查询并异步返回结果。"""

        payloads = self.get_payloads(queries, max_results)
        headers = self.get_headers()

        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, data=payloads, headers=headers, proxy=self.proxy) as response:
                    response.raise_for_status()
                    res = await response.json()
        else:
            async with self.aiosession.post(self.url, data=payloads, headers=headers, proxy=self.proxy) as response:
                response.raise_for_status()
                res = await response.json()

        return res

    def get_payloads(self, queries: list[str], max_results: int) -> Dict[str, str]:
        """生成 Serper 查询的请求负载。"""
        payloads = []
        for query in queries:
            _payload = {
                "q": query,
                "num": max_results,
            }
            payloads.append({**self.payload, **_payload})
        return json.dumps(payloads, sort_keys=True)

    def get_headers(self) -> Dict[str, str]:
        """获取 Serper 请求所需的 HTTP 头部。"""
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        return headers

    @staticmethod
    def _process_response(res: dict, as_string: bool = False) -> str:
        """处理从 Serper 返回的响应。"""
        # logger.debug(res)
        focus = ["title", "snippet", "link"]

        def get_focused(x):
            return {i: j for i, j in x.items() if i in focus}

        if "error" in res.keys():
            raise ValueError(f"从 Serper 获取错误：{res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif "answer_box" in res.keys() and "snippet_highlighted_words" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif "sports_results" in res.keys() and "game_spotlight" in res["sports_results"].keys():
            toret = res["sports_results"]["game_spotlight"]
        elif "knowledge_graph" in res.keys() and "description" in res["knowledge_graph"].keys():
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic"][0].keys():
            toret = res["organic"][0]["snippet"]
        else:
            toret = "没有找到好的搜索结果"

        toret_l = []
        if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret_l += [get_focused(res["answer_box"])]
        if res.get("organic"):
            toret_l += [get_focused(i) for i in res.get("organic")]

        return str(toret) + "\n" + str(toret_l) if as_string else toret_l


if __name__ == "__main__":
    import fire

    fire.Fire(SerperWrapper().run)
