#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal, Optional

from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, PrivateAttr

from metagpt.logs import logger
from metagpt.utils.parse_html import WebPage


class PlaywrightWrapper(BaseModel):
    """Playwright的封装类。

    要使用此模块，您需要安装 `playwright` Python 包，并确保安装所需的浏览器。可以通过运行
    `pip install metagpt[playwright]` 来安装playwright，并通过第一次运行时执行 `playwright install`
    命令来下载所需的浏览器二进制文件。
    """

    browser_type: Literal["chromium", "firefox", "webkit"] = "chromium"  # 浏览器类型，默认为"chromium"
    launch_kwargs: dict = Field(default_factory=dict)  # 浏览器启动时的配置参数
    proxy: Optional[str] = None  # 可选的代理服务器地址
    context_kwargs: dict = Field(default_factory=dict)  # 浏览器上下文配置
    _has_run_precheck: bool = PrivateAttr(False)  # 标记是否已执行过浏览器预检查

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        launch_kwargs = self.launch_kwargs
        if self.proxy and "proxy" not in launch_kwargs:  # 如果设置了代理，且启动参数中没有代理配置，则添加代理
            args = launch_kwargs.get("args", [])
            if not any(str.startswith(i, "--proxy-server=") for i in args):
                launch_kwargs["proxy"] = {"server": self.proxy}

        # 根据传入的kwargs，设置浏览器上下文中的一些额外配置项
        for key in ["ignore_https_errors", "java_script_enabled", "extra_http_headers", "user_agent"]:
            if key in kwargs:
                self.context_kwargs[key] = kwargs[key]

    async def run(self, url: str, *urls: str, per_page_timeout: float = None) -> WebPage | list[WebPage]:
        """启动浏览器并加载网页。

        根据传入的URL列表，加载一个或多个网页，并返回网页内容。

        Args:
            url: 第一个网页的URL。
            *urls: 其他网页的URL（如果有）。
            per_page_timeout: 每个网页加载的最大超时时间，单位为秒。

        Returns:
            单个WebPage对象或多个WebPage对象的列表。
        """
        async with async_playwright() as ap:
            browser_type = getattr(ap, self.browser_type)  # 获取指定的浏览器类型
            await self._run_precheck(browser_type)  # 执行浏览器的预检查
            browser = await browser_type.launch(**self.launch_kwargs)  # 启动浏览器
            _scrape = self._scrape  # 页面抓取方法

            if urls:
                return await asyncio.gather(
                    _scrape(browser, url, per_page_timeout), *(_scrape(browser, i, per_page_timeout) for i in urls)
                )
            return await _scrape(browser, url, per_page_timeout)

    async def _scrape(self, browser, url, timeout: float = None):
        """抓取网页内容。

        通过浏览器加载网页，并获取网页的HTML和文本内容。

        Args:
            browser: Playwright浏览器实例。
            url: 要加载的网页URL。
            timeout: 页面加载的超时时间，单位为秒。

        Returns:
            返回包含网页文本和HTML内容的WebPage对象。
        """
        context = await browser.new_context(**self.context_kwargs)  # 创建新的浏览器上下文

        if timeout is not None:
            context.set_default_timeout(timeout * 1000)  # playwright使用毫秒作为超时单位

        page = await context.new_page()  # 创建新页面
        async with page:
            try:
                await page.goto(url)  # 跳转到指定的URL
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # 滚动页面到底部
                html = await page.content()  # 获取页面的HTML内容
                inner_text = await page.evaluate("() => document.body.innerText")  # 获取页面的文本内容
            except Exception as e:
                inner_text = f"Fail to load page content for {e}"  # 如果加载失败，返回错误信息
                html = ""
            return WebPage(inner_text=inner_text, html=html, url=url)

    async def _run_precheck(self, browser_type):
        """执行浏览器的预检查。

        确保浏览器已正确安装，如果未安装，尝试自动安装浏览器。

        Args:
            browser_type: Playwright浏览器类型。
        """
        if self._has_run_precheck:
            return

        executable_path = Path(browser_type.executable_path)  # 获取浏览器可执行文件的路径
        if not executable_path.exists() and "executable_path" not in self.launch_kwargs:
            kwargs = {}
            if self.proxy:
                kwargs["env"] = {"ALL_PROXY": self.proxy}
            await _install_browsers(self.browser_type, **kwargs)  # 如果浏览器未安装，调用安装函数

            if self._has_run_precheck:
                return

            if not executable_path.exists():
                # 如果浏览器仍未找到，尝试使用备用的构建版本
                parts = executable_path.parts
                available_paths = list(Path(*parts[:-3]).glob(f"{self.browser_type}-*"))
                if available_paths:
                    logger.warning(
                        "It seems that your OS is not officially supported by Playwright. "
                        "Try to set executable_path to the fallback build version."
                    )
                    executable_path = available_paths[0].joinpath(*parts[-2:])
                    self.launch_kwargs["executable_path"] = str(executable_path)
        self._has_run_precheck = True


def _get_install_lock():
    """获取安装锁。

    用于确保在安装浏览器时只有一个并发任务。

    Returns:
        asyncio.Lock: 安装锁。
    """
    global _install_lock
    if _install_lock is None:
        _install_lock = asyncio.Lock()  # 创建一个新的异步锁
    return _install_lock


async def _install_browsers(*browsers, **kwargs) -> None:
    """安装Playwright所需的浏览器。

    如果指定的浏览器未安装，则会下载并安装它们。

    Args:
        *browsers: 要安装的浏览器类型。
        **kwargs: 额外的配置参数。
    """
    async with _get_install_lock():
        browsers = [i for i in browsers if i not in _install_cache]  # 检查哪些浏览器未安装
        if not browsers:
            return
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "playwright",
            "install",
            *browsers,  # 安装浏览器
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )

        await asyncio.gather(_log_stream(process.stdout, logger.info), _log_stream(process.stderr, logger.warning))

        if await process.wait() == 0:
            logger.info("Install browser for playwright successfully.")  # 安装成功
        else:
            logger.warning("Fail to install browser for playwright.")  # 安装失败
        _install_cache.update(browsers)  # 更新已安装的浏览器缓存


async def _log_stream(sr, log_func):
    """异步日志流。

    用于读取和记录安装过程中产生的输出流。

    Args:
        sr: 流对象，通常为stdout或stderr。
        log_func: 用于记录日志的函数。
    """
    while True:
        line = await sr.readline()
        if not line:
            return
        log_func(f"[playwright install browser]: {line.decode().strip()}")  # 记录日志


_install_lock: asyncio.Lock = None  # 安装锁的初始化
_install_cache = set()  # 安装缓存，用于存储已安装的浏览器类型
