#!/usr/bin/env python

from __future__ import annotations

import asyncio
import json
from concurrent import futures
from typing import Literal, Optional, overload

from pydantic import BaseModel, ConfigDict

try:
    from duckduckgo_search import DDGS  # 尝试导入duckduckgo_search库中的DDGS类
except ImportError:
    raise ImportError(
        "要使用此模块，您需要安装 `duckduckgo_search` Python包。"
        "您可以运行命令：`pip install -e.[search-ddg]` 来安装该包"
    )  # 如果没有安装duckduckgo_search库，抛出错误并给出安装提示


class DDGAPIWrapper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loop: Optional[asyncio.AbstractEventLoop] = None  # 可选的事件循环，用于异步任务
    executor: Optional[futures.Executor] = None  # 可选的执行器，用于在指定线程池中运行任务
    proxy: Optional[str] = None  # 可选的代理设置

    @property
    def ddgs(self):
        """返回DuckDuckGo搜索实例，支持代理设置"""
        return DDGS(proxies=self.proxy)

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
        focus: list[str] | None = None,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
        focus: list[str] | None = None,
    ) -> list[dict[str, str]]:
        ...

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
    ) -> str | list[dict]:
        """使用DuckDuckGo API返回搜索结果

        参数：
            query: 搜索查询字符串。
            max_results: 返回的最大结果数，默认为8。
            as_string: 布尔值，决定返回结果的格式。如果为True，返回格式化的字符串；如果为False，返回包含详细信息的字典列表。

        返回：
            搜索结果，可以是字符串或字典列表。
        """
        loop = self.loop or asyncio.get_event_loop()  # 获取事件循环
        future = loop.run_in_executor(
            self.executor,  # 在指定执行器中运行任务
            self._search_from_ddgs,  # 调用内部方法进行DuckDuckGo搜索
            query,
            max_results,
        )
        search_results = await future  # 等待任务完成并获取结果

        # 如果需要返回字符串格式的结果
        if as_string:
            return json.dumps(search_results, ensure_ascii=False)
        return search_results  # 返回字典列表形式的结果

    def _search_from_ddgs(self, query: str, max_results: int):
        """使用DuckDuckGo搜索引擎获取结果"""
        return [
            {"link": i["href"], "snippet": i["body"], "title": i["title"]}
            for (_, i) in zip(range(max_results), self.ddgs.text(query))
        ]  # 从DuckDuckGo返回的结果中提取链接、摘要和标题

if __name__ == "__main__":
    import fire

    fire.Fire(DDGAPIWrapper().run)  # 使用fire库创建命令行接口，调用run方法