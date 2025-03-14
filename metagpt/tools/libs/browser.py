from __future__ import annotations

import time
from typing import Literal, Optional

from playwright.async_api import Browser as Browser_
from playwright.async_api import (
    BrowserContext,
    Frame,
    Page,
    Playwright,
    Request,
    async_playwright,
)
from pydantic import BaseModel, ConfigDict, Field

from metagpt.tools.tool_registry import register_tool
from metagpt.utils.a11y_tree import (
    click_element,
    get_accessibility_tree,
    get_backend_node_id,
    hover_element,
    key_press,
    parse_accessibility_tree,
    scroll_page,
    type_text,
)
from metagpt.utils.proxy_env import get_proxy_from_env
from metagpt.utils.report import BrowserReporter


@register_tool(
    tags=["web", "browse"],
    include_functions=[
        "click",
        "close_tab",
        "go_back",
        "go_forward",
        "goto",
        "hover",
        "press",
        "scroll",
        "tab_focus",
        "type",
    ],
)
class Browser(BaseModel):
    """浏览网页的工具类。若已存在该类的实例，则不要再次初始化。

    注意：如果计划使用浏览器来帮助完成任务，浏览器应作为独立任务使用，每次基于网页上的内容执行操作，然后再继续下一步。

    ## 示例
    问题：查看 geekan/MetaGPT 仓库中最新问题的详细信息。
    计划：使用浏览器打开 geekan/MetaGPT 仓库的问题页面。
    解决方案：
    首先，我们使用 `Browser.goto` 命令打开 MetaGPT 仓库的问题页面。

    >>> await browser.goto("https://github.com/geekan/MetaGPT/issues")

    从网页输出中，我们发现可以通过点击 ID 为 "1141" 的元素访问最新问题。

    >>> await browser.click(1141)

    最后，我们找到了最新问题的网页，可以关闭当前标签页并完成任务。

    >>> await browser.close_tab()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    playwright: Optional[Playwright] = Field(default=None, exclude=True)  # Playwright 实例
    browser_instance: Optional[Browser_] = Field(default=None, exclude=True)  # 浏览器实例
    browser_ctx: Optional[BrowserContext] = Field(default=None, exclude=True)  # 浏览器上下文
    page: Optional[Page] = Field(default=None, exclude=True)  # 当前页面
    accessibility_tree: list = Field(default_factory=list)  # 可访问性树，用于辅助功能分析
    headless: bool = Field(default=True)  # 是否无头浏览器，默认启用
    proxy: Optional[dict] = Field(default_factory=get_proxy_from_env)  # 代理设置
    is_empty_page: bool = Field(default=True)  # 页面是否为空
    reporter: BrowserReporter = Field(default_factory=BrowserReporter)  # 浏览器报告实例

    async def start(self) -> None:
        """启动 Playwright 并启动浏览器"""
        if self.playwright is None:
            self.playwright = playwright = await async_playwright().start()  # 启动 Playwright
            browser = self.browser_instance = await playwright.chromium.launch(headless=self.headless, proxy=self.proxy)  # 启动 Chromium 浏览器
            browser_ctx = self.browser_ctx = await browser.new_context()  # 创建新的浏览器上下文
            self.page = await browser_ctx.new_page()  # 创建新页面

    async def stop(self):
        """停止浏览器并释放资源"""
        if self.playwright:
            playwright = self.playwright
            self.playwright = None
            self.browser_instance = None
            self.browser_ctx = None
            await playwright.stop()  # 停止 Playwright

    async def click(self, element_id: int):
        """点击页面上指定 ID 的元素"""
        await click_element(self.page, get_backend_node_id(element_id, self.accessibility_tree))  # 模拟点击元素
        return await self._wait_page()  # 等待页面加载完成

    async def type(self, element_id: int, content: str, press_enter_after: bool = False):
        """在指定 ID 的输入框中输入内容"""
        if press_enter_after:
            content += "\n"  # 如果需要按回车，添加换行符
        await click_element(self.page, get_backend_node_id(element_id, self.accessibility_tree))  # 点击输入框
        await type_text(self.page, content)  # 输入文本
        return await self._wait_page()  # 等待页面加载完成

    async def hover(self, element_id: int):
        """鼠标悬停在指定 ID 的元素上"""
        await hover_element(self.page, get_backend_node_id(element_id, self.accessibility_tree))  # 模拟鼠标悬停
        return await self._wait_page()  # 等待页面加载完成

    async def press(self, key_comb: str):
        """模拟按下键盘上的组合键（例如 Ctrl+v）"""
        await key_press(self.page, key_comb)  # 模拟键盘按键
        return await self._wait_page()  # 等待页面加载完成

    async def scroll(self, direction: Literal["down", "up"]):
        """向上或向下滚动页面"""
        await scroll_page(self.page, direction)  # 滚动页面
        return await self._wait_page()  # 等待页面加载完成

    async def goto(self, url: str, timeout: float = 90000):
        """访问指定的 URL"""
        if self.page is None:
            await self.start()  # 如果页面未启动，则启动浏览器
        async with self.reporter as reporter:
            await reporter.async_report(url, "url")  # 记录 URL
            await self.page.goto(url, timeout=timeout)  # 访问 URL
            self.is_empty_page = False  # 标记页面已加载
            return await self._wait_page()  # 等待页面加载完成

    async def go_back(self):
        """返回到上一页"""
        await self.page.go_back()  # 返回到上一页
        return await self._wait_page()  # 等待页面加载完成

    async def go_forward(self):
        """前进到下一页"""
        await self.page.go_forward()  # 前进到下一页
        return await self._wait_page()  # 等待页面加载完成

    async def tab_focus(self, page_number: int):
        """切换到指定的浏览器标签页"""
        page = self.browser_ctx.pages[page_number]  # 获取指定的标签页
        await page.bring_to_front()  # 将标签页置前
        return await self._wait_page()  # 等待页面加载完成

    async def close_tab(self):
        """关闭当前活动的标签页"""
        await self.page.close()  # 关闭当前标签页
        if len(self.browser_ctx.pages) > 0:
            self.page = self.browser_ctx.pages[-1]  # 切换到最后一个标签页
        else:
            self.page = await self.browser_ctx.new_page()  # 如果没有标签页，创建新标签页
            self.is_empty_page = True  # 标记页面为空
        return await self._wait_page()  # 等待页面加载完成

    async def _wait_page(self):
        """等待页面加载完成"""
        page = self.page
        await self._wait_until_page_idle(page)  # 等待页面空闲
        self.accessibility_tree = await get_accessibility_tree(page)  # 获取可访问性树
        await self.reporter.async_report(page, "page")  # 记录页面信息
        return f"SUCCESS, URL: {page.url} have been loaded."  # 返回成功信息

    def _register_page_event(self, page: Page):
        """注册页面事件以追踪页面状态"""
        page.last_busy_time = time.time()  # 记录页面的最后忙碌时间
        page.requests = set()  # 页面请求集合
        page.on("domcontentloaded", self._update_page_last_busy_time)  # DOM内容加载事件
        page.on("load", self._update_page_last_busy_time)  # 页面加载完成事件
        page.on("request", self._on_page_request)  # 请求事件
        page.on("requestfailed", self._on_page_requestfinished)  # 请求失败事件
        page.on("requestfinished", self._on_page_requestfinished)  # 请求完成事件
        page.on("frameattached", self._on_frame_change)  # 帧附加事件
        page.on("framenavigated", self._on_frame_change)  # 帧导航事件

    async def _wait_until_page_idle(self, page) -> None:
        """等待页面空闲"""
        if not hasattr(page, "last_busy_time"):
            self._register_page_event(page)  # 注册页面事件
        else:
            page.last_busy_time = time.time()  # 更新最后忙碌时间
        while time.time() - page.last_busy_time < 0.5:  # 等待 0.5 秒
            await page.wait_for_timeout(100)  # 等待页面处理请求

    async def _update_page_last_busy_time(self, page: Page):
        """更新页面最后忙碌时间"""
        page.last_busy_time = time.time()

    async def _on_page_request(self, request: Request):
        """处理页面请求"""
        page = request.frame.page
        page.requests.add(request)  # 将请求添加到页面请求集合
        await self._update_page_last_busy_time(page)  # 更新页面最后忙碌时间

    async def _on_page_requestfinished(self, request: Request):
        """处理请求完成事件"""
        request.frame.page.requests.discard(request)  # 从请求集合中移除请求

    async def _on_frame_change(self, frame: Frame):
        """处理帧变化事件"""
        await self._update_page_last_busy_time(frame.page)  # 更新页面最后忙碌时间

    async def view(self):
        """查看当前页面的可访问性树并返回"""
        observation = parse_accessibility_tree(self.accessibility_tree)  # 解析可访问性树
        return f"当前浏览器视图\n URL: {self.page.url}\n观察信息:\n{observation[0]}\n"

    async def __aenter__(self):
        """进入异步上下文管理器，启动浏览器"""
        await self.start()
        return self

    async def __aexit__(self, *args, **kwargs):
        """退出异步上下文管理器，停止浏览器"""
        await self.stop()
