#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import importlib
from concurrent import futures
from copy import deepcopy
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.core.download_manager import WDMDownloadManager
from webdriver_manager.core.http import WDMHttpClient

from metagpt.utils.parse_html import WebPage


class SeleniumWrapper(BaseModel):
    """Selenium的封装类。

    使用此模块前，请确保以下几点：

    1. 运行命令：`pip install metagpt[selenium]`。
    2. 确保已安装兼容的浏览器，并为该浏览器设置了适当的WebDriver。例如，如果你安装了Mozilla Firefox，可以将SELENIUM_BROWSER_TYPE配置为firefox。然后，你就可以使用Selenium WebBrowserEngine来抓取网页内容。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 浏览器类型，默认为Chrome
    browser_type: Literal["chrome", "firefox", "edge", "ie"] = "chrome"

    # 启动参数
    launch_kwargs: dict = Field(default_factory=dict)

    # 代理地址
    proxy: Optional[str] = None

    # 事件循环
    loop: Optional[asyncio.AbstractEventLoop] = None

    # 执行器
    executor: Optional[futures.Executor] = None

    # 用于检查浏览器和WebDriver是否准备好的标记
    _has_run_precheck: bool = PrivateAttr(False)

    # 用于获取WebDriver的函数
    _get_driver: Optional[Callable] = PrivateAttr(None)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # 如果提供了代理地址，并且启动参数中没有"proxy-server"，则将代理添加到启动参数中
        if self.proxy and "proxy-server" not in self.launch_kwargs:
            self.launch_kwargs["proxy-server"] = self.proxy

    @property
    def launch_args(self):
        """返回启动参数列表"""
        return [f"--{k}={v}" for k, v in self.launch_kwargs.items() if k != "executable_path"]

    @property
    def executable_path(self):
        """返回WebDriver的可执行路径"""
        return self.launch_kwargs.get("executable_path")

    async def run(self, url: str, *urls: str, per_page_timeout: float = None) -> WebPage | list[WebPage]:
        """运行抓取任务，返回指定URL的网页内容"""
        await self._run_precheck()

        # 定义抓取网页的函数
        _scrape = lambda url, per_page_timeout: self.loop.run_in_executor(
            self.executor, self._scrape_website, url, per_page_timeout
        )

        # 如果有多个URL，将并行抓取所有URL
        if urls:
            return await asyncio.gather(_scrape(url, per_page_timeout), *(_scrape(i, per_page_timeout) for i in urls))

        # 否则，只抓取一个URL
        return await _scrape(url, per_page_timeout)

    async def _run_precheck(self):
        """检查WebDriver是否准备好，如果没有准备好则进行初始化"""
        if self._has_run_precheck:
            return

        # 如果未指定事件循环，使用默认的事件循环
        self.loop = self.loop or asyncio.get_event_loop()

        # 初始化获取WebDriver的函数
        self._get_driver = await self.loop.run_in_executor(
            self.executor,
            lambda: _gen_get_driver_func(
                self.browser_type, *self.launch_args, executable_path=self.executable_path, proxy=self.proxy
            ),
        )

        self._has_run_precheck = True

    def _scrape_website(self, url, timeout: float = None):
        """使用Selenium抓取网页内容"""
        with self._get_driver() as driver:
            try:
                # 打开URL并等待页面加载
                driver.get(url)
                WebDriverWait(driver, timeout or 30).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # 获取页面的HTML和文本内容
                inner_text = driver.execute_script("return document.body.innerText;")
                html = driver.page_source
            except Exception as e:
                # 如果抓取失败，返回错误信息
                inner_text = f"Fail to load page content for {e}"
                html = ""
            return WebPage(inner_text=inner_text, html=html, url=url)


# 浏览器WebDriver管理器类型
_webdriver_manager_types = {
    "chrome": ("webdriver_manager.chrome", "ChromeDriverManager"),
    "firefox": ("webdriver_manager.firefox", "GeckoDriverManager"),
    "edge": ("webdriver_manager.microsoft", "EdgeChromiumDriverManager"),
    "ie": ("webdriver_manager.microsoft", "IEDriverManager"),
}


class WDMHttpProxyClient(WDMHttpClient):
    """使用代理的WebDriver管理器HTTP客户端"""

    def __init__(self, proxy: str = None):
        super().__init__()
        self.proxy = proxy

    def get(self, url, **kwargs):
        """发送GET请求时使用代理"""
        if "proxies" not in kwargs and self.proxy:
            kwargs["proxies"] = {"all": self.proxy}
        return super().get(url, **kwargs)


def _gen_get_driver_func(browser_type, *args, executable_path=None, proxy=None):
    """生成获取WebDriver实例的函数"""
    # 导入WebDriver相关模块
    WebDriver = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.webdriver"), "WebDriver")
    Service = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.service"), "Service")
    Options = getattr(importlib.import_module(f"selenium.webdriver.{browser_type}.options"), "Options")

    # 如果没有指定可执行路径，则使用WebDriverManager自动下载并获取WebDriver路径
    if not executable_path:
        module_name, type_name = _webdriver_manager_types[browser_type]
        DriverManager = getattr(importlib.import_module(module_name), type_name)
        driver_manager = DriverManager(download_manager=WDMDownloadManager(http_client=WDMHttpProxyClient(proxy=proxy)))
        executable_path = driver_manager.install()

    def _get_driver():
        """创建WebDriver实例并返回"""
        options = Options()
        options.add_argument("--headless")  # 无头模式
        options.add_argument("--enable-javascript")  # 启用JavaScript
        if browser_type == "chrome":
            options.add_argument("--disable-gpu")  # 禁用GPU加速
            options.add_argument("--disable-dev-shm-usage")  # 解决资源不足问题
            options.add_argument("--no-sandbox")  # 禁用沙盒模式
        for i in args:
            options.add_argument(i)
        return WebDriver(options=deepcopy(options), service=Service(executable_path=executable_path))

    return _get_driver
