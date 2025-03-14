#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 20:15
@Author  : alexanderwu
@File    : search_engine.py
"""
import importlib
from typing import Annotated, Callable, Coroutine, Literal, Optional, Union, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

from metagpt.configs.search_config import SearchConfig
from metagpt.logs import logger
from metagpt.tools import SearchEngineType


class SearchEngine(BaseModel):
    """用于配置和执行不同搜索引擎的模型。

    属性:
        model_config: 配置模型，允许使用任意类型。
        engine: 要使用的搜索引擎类型。
        run_func: 可选的执行搜索的函数。如果未提供，则根据引擎类型自动确定。
        api_key: 可选的搜索引擎 API 密钥。
        proxy: 可选的代理，用于搜索引擎请求。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    engine: SearchEngineType = SearchEngineType.SERPER_GOOGLE  # 默认使用 Serper 搜索引擎
    run_func: Annotated[  # 可选的运行函数，异步执行搜索
        Optional[Callable[[str, int, bool], Coroutine[None, None, Union[str, list[str]]]]], Field(exclude=True)
    ] = None
    api_key: Optional[str] = None  # 可选的 API 密钥
    proxy: Optional[str] = None  # 可选的代理

    @model_validator(mode="after")
    def validate_extra(self):
        """验证传递给模型的额外字段，并相应地更新运行函数。"""
        data = self.model_dump(exclude={"engine"}, exclude_none=True, exclude_defaults=True)
        if self.model_extra:
            data.update(self.model_extra)
        self._process_extra(**data)  # 处理额外的配置
        return self

    def _process_extra(
        self,
        run_func: Optional[Callable[[str, int, bool], Coroutine[None, None, Union[str, list[str]]]]] = None,
        **kwargs,
    ):
        """根据搜索引擎类型处理额外的配置，并更新运行函数。

        参数:
            run_func: 可选的执行搜索的函数。如果未提供，将根据搜索引擎类型自动确定。
        """
        # 根据引擎类型加载相应的模块并设置运行函数
        if self.engine == SearchEngineType.SERPAPI_GOOGLE:
            module = "metagpt.tools.search_engine_serpapi"
            run_func = importlib.import_module(module).SerpAPIWrapper(**kwargs).run
        elif self.engine == SearchEngineType.SERPER_GOOGLE:
            module = "metagpt.tools.search_engine_serper"
            run_func = importlib.import_module(module).SerperWrapper(**kwargs).run
        elif self.engine == SearchEngineType.DIRECT_GOOGLE:
            module = "metagpt.tools.search_engine_googleapi"
            run_func = importlib.import_module(module).GoogleAPIWrapper(**kwargs).run
        elif self.engine == SearchEngineType.DUCK_DUCK_GO:
            module = "metagpt.tools.search_engine_ddg"
            run_func = importlib.import_module(module).DDGAPIWrapper(**kwargs).run
        elif self.engine == SearchEngineType.CUSTOM_ENGINE:
            run_func = self.run_func  # 使用自定义的搜索函数
        elif self.engine == SearchEngineType.BING:
            module = "metagpt.tools.search_engine_bing"
            run_func = importlib.import_module(module).BingAPIWrapper(**kwargs).run
        else:
            raise NotImplementedError("该搜索引擎类型尚未实现")
        self.run_func = run_func

    @classmethod
    def from_search_config(cls, config: SearchConfig, **kwargs):
        """从 SearchConfig 创建 SearchEngine 实例。

        参数:
            config: 用于创建 SearchEngine 实例的搜索配置。
        """
        data = config.model_dump(exclude={"api_type", "search_func"})
        if config.search_func is not None:
            data["run_func"] = config.search_func  # 如果配置了搜索函数，使用它

        return cls(engine=config.api_type, **data, **kwargs)

    @classmethod
    def from_search_func(
        cls, search_func: Callable[[str, int, bool], Coroutine[None, None, Union[str, list[str]]]], **kwargs
    ):
        """从自定义搜索函数创建 SearchEngine 实例。

        参数:
            search_func: 执行搜索的可调用函数。
        """
        return cls(engine=SearchEngineType.CUSTOM_ENGINE, run_func=search_func, **kwargs)

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
    ) -> list[dict[str, str]]:
        ...

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
        ignore_errors: bool = False,
    ) -> Union[str, list[dict[str, str]]]:
        """执行搜索查询。

        参数:
            query: 搜索查询的字符串。
            max_results: 返回的最大结果数量，默认为 8。
            as_string: 是否将结果作为字符串返回，默认为 True。如果为 False，将返回字典列表。
            ignore_errors: 是否忽略搜索中的错误，默认为 False。

        返回:
            搜索结果，可以是字符串或字典列表。
        """
        try:
            # 执行搜索查询
            return await self.run_func(query, max_results=max_results, as_string=as_string)
        except Exception as e:
            # 捕获异常并记录错误
            logger.exception(f"搜索失败: {query}，错误信息: {e}")
            if not ignore_errors:
                raise e
            # 如果忽略错误，返回空字符串或空列表
            return "" if as_string else []
