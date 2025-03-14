#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from typing import Annotated, Any, Callable, Coroutine, Optional, Union, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

from metagpt.configs.browser_config import BrowserConfig
from metagpt.tools import WebBrowserEngineType
from metagpt.utils.parse_html import WebPage


class WebBrowserEngine(BaseModel):
    """定义一个网页浏览器引擎配置，用于自动化浏览和数据提取。

    该类封装了不同网页浏览器引擎（如 Playwright、Selenium 或自定义实现）的配置和操作逻辑，
    提供了一个统一的接口来运行浏览器自动化任务。

    属性：
        model_config: 配置字典，允许任意类型和额外字段。
        engine: 要使用的网页浏览器引擎类型。
        run_func: 可选的协程函数，用于运行浏览器引擎。
        proxy: 可选的代理服务器 URL，用于浏览器引擎。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    engine: WebBrowserEngineType = WebBrowserEngineType.PLAYWRIGHT  # 浏览器引擎类型，默认使用 Playwright
    run_func: Annotated[  # 可选的协程函数，运行浏览器引擎的任务
        Optional[Callable[..., Coroutine[Any, Any, Union[WebPage, list[WebPage]]]]],
        Field(exclude=True),
    ] = None
    proxy: Optional[str] = None  # 可选的代理服务器 URL

    @model_validator(mode="after")
    def validate_extra(self):
        """验证并处理模型初始化后的额外配置数据。

        该方法由 Pydantic 自动调用，用于验证和处理提供给模型的任何额外配置数据。
        确保额外数据正确集成到模型的配置和操作逻辑中。

        返回：
            处理后的模型实例。
        """
        data = self.model_dump(exclude={"engine"}, exclude_none=True, exclude_defaults=True)
        if self.model_extra:
            data.update(self.model_extra)  # 将额外的数据合并到配置中
        self._process_extra(**data)  # 处理额外的配置数据
        return self

    def _process_extra(self, **kwargs):
        """处理额外配置数据以设置浏览器引擎的运行函数。

        根据指定的引擎类型，此方法动态导入并配置相应的浏览器引擎封装器及其运行函数。

        参数：
            **kwargs: 任意关键字参数，表示额外的配置数据。

        异常：
            NotImplementedError: 如果引擎类型不被支持，则抛出该异常。
        """
        if self.engine is WebBrowserEngineType.PLAYWRIGHT:
            module = "metagpt.tools.web_browser_engine_playwright"  # 使用 Playwright 引擎
            run_func = importlib.import_module(module).PlaywrightWrapper(**kwargs).run
        elif self.engine is WebBrowserEngineType.SELENIUM:
            module = "metagpt.tools.web_browser_engine_selenium"  # 使用 Selenium 引擎
            run_func = importlib.import_module(module).SeleniumWrapper(**kwargs).run
        elif self.engine is WebBrowserEngineType.CUSTOM:
            run_func = self.run_func  # 使用自定义引擎
        else:
            raise NotImplementedError  # 如果引擎类型不支持，则抛出异常
        self.run_func = run_func  # 设置运行函数

    @classmethod
    def from_browser_config(cls, config: BrowserConfig, **kwargs):
        """通过 BrowserConfig 对象和额外的关键字参数创建 WebBrowserEngine 实例。

        该类方法通过从 BrowserConfig 对象中提取配置数据，并可选地与额外的关键字参数合并，
        来创建 WebBrowserEngine 实例。

        参数：
            config: 一个包含基础配置数据的 BrowserConfig 对象。
            **kwargs: 可选的额外关键字参数，用于覆盖或扩展配置。

        返回：
            根据提供的配置参数创建的 WebBrowserEngine 实例。
        """
        data = config.model_dump()  # 提取 BrowserConfig 配置数据
        return cls(**data, **kwargs)  # 创建 WebBrowserEngine 实例

    @overload
    async def run(self, url: str, per_page_timeout: float = None) -> WebPage:
        ...

    @overload
    async def run(self, url: str, *urls: str, per_page_timeout: float = None) -> list[WebPage]:
        ...

    async def run(self, url: str, *urls: str, per_page_timeout: float = None) -> WebPage | list[WebPage]:
        """运行浏览器引擎以加载一个或多个网页。

        该方法是重载的 run 方法的实现，它将加载网页的任务委托给配置的运行函数，
        处理单个 URL 或多个 URL 的加载。

        参数：
            url: 第一个网页的 URL。
            *urls: 其他网页的 URL（如果有的话）。
            per_page_timeout: 每个网页的最大加载时间（秒）。

        返回：
            如果提供单个 URL，则返回 WebPage 对象；如果提供多个 URL，则返回 WebPage 对象列表。
        """
        return await self.run_func(url, *urls, per_page_timeout=per_page_timeout)
