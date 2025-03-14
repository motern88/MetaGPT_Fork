#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 18:27
@Author  : alexanderwu
@File    : search_engine_serpapi.py
"""
import warnings
from typing import Any, Dict, Optional

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SerpAPIWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str  # API密钥，用于访问SerpAPI
    params: dict = Field(
        default_factory=lambda: {
            "engine": "google",  # 使用Google搜索引擎
            "google_domain": "google.com",  # 设置Google的域名
            "gl": "us",  # 设置Google的区域（美国）
            "hl": "en",  # 设置语言为英语
        }
    )
    url: str = "https://serpapi.com/search"  # SerpAPI的URL地址
    aiosession: Optional[aiohttp.ClientSession] = None  # 可选的aiohttp客户端会话
    proxy: Optional[str] = None  # 可选的代理设置

    @model_validator(mode="before")
    @classmethod
    def validate_serpapi(cls, values: dict) -> dict:
        """验证SerpAPI参数的有效性"""
        # 检查是否存在已废弃的`serpapi_api_key`，并发出警告
        if "serpapi_api_key" in values:
            values.setdefault("api_key", values["serpapi_api_key"])
            warnings.warn("`serpapi_api_key` is deprecated, use `api_key` instead", DeprecationWarning, stacklevel=2)

        # 如果没有提供`api_key`，抛出异常
        if "api_key" not in values:
            raise ValueError(
                "To use serpapi search engine, make sure you provide the `api_key` when constructing an object. You can obtain"
                " an API key from https://serpapi.com/."
            )
        return values

    async def run(self, query, max_results: int = 8, as_string: bool = True, **kwargs: Any) -> str:
        """通过SerpAPI运行查询并异步处理结果"""
        result = await self.results(query, max_results)  # 获取查询结果
        return self._process_response(result, as_string=as_string)  # 处理并返回结果

    async def results(self, query: str, max_results: int) -> dict:
        """使用aiohttp异步执行查询并返回结果"""
        params = self.get_params(query)  # 获取查询参数
        params["source"] = "python"  # 设置源为python
        params["num"] = max_results  # 设置返回结果的数量
        params["output"] = "json"  # 设置返回格式为json

        # 如果没有提供aiohttp客户端会话，则创建新的会话
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, params=params, proxy=self.proxy) as response:
                    response.raise_for_status()  # 检查请求是否成功
                    res = await response.json()  # 获取JSON响应
        else:
            # 使用提供的aiohttp客户端会话
            async with self.aiosession.get(self.url, params=params, proxy=self.proxy) as response:
                response.raise_for_status()
                res = await response.json()

        return res

    def get_params(self, query: str) -> Dict[str, str]:
        """获取查询参数"""
        _params = {
            "api_key": self.api_key,  # API密钥
            "q": query,  # 查询关键词
        }
        params = {**self.params, **_params}  # 合并默认参数和查询参数
        return params

    @staticmethod
    def _process_response(res: dict, as_string: bool) -> str:
        """处理来自SerpAPI的响应"""
        # logger.debug(res)
        focus = ["title", "snippet", "link"]  # 关注的字段
        get_focused = lambda x: {i: j for i, j in x.items() if i in focus}  # 提取关注的字段

        # 错误处理
        if "error" in res.keys():
            if res["error"] == "Google hasn't returned any results for this query.":
                toret = "No good search result found"  # 如果没有找到有效结果
            else:
                raise ValueError(f"Got error from SerpAPI: {res['error']}")  # 抛出错误信息
        # 如果存在回答框（answer_box）
        elif "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif "answer_box" in res.keys() and "snippet_highlighted_words" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        # 处理体育比赛结果
        elif "sports_results" in res.keys() and "game_spotlight" in res["sports_results"].keys():
            toret = res["sports_results"]["game_spotlight"]
        # 处理知识图谱
        elif "knowledge_graph" in res.keys() and "description" in res["knowledge_graph"].keys():
            toret = res["knowledge_graph"]["description"]
        # 返回搜索结果的snippet
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]
        else:
            toret = "No good search result found"  # 如果没有有效结果

        toret_l = []
        # 如果回答框中有snippet，加入到结果中
        if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret_l += [get_focused(res["answer_box"])]
        # 如果有有机搜索结果，加入到结果中
        if res.get("organic_results"):
            toret_l += [get_focused(i) for i in res.get("organic_results")]

        # 根据as_string返回格式化的结果或原始结果列表
        return str(toret) + "\n" + str(toret_l) if as_string else toret_l


if __name__ == "__main__":
    import fire

    fire.Fire(SerpAPIWrapper().run)  # 使用fire库启动命令行接口
