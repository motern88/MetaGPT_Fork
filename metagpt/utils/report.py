import asyncio
import os
import typing
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union
from urllib.parse import unquote, urlparse, urlunparse
from uuid import UUID, uuid4

from aiohttp import ClientSession, UnixConnector
from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage
from pydantic import BaseModel, Field, PrivateAttr

from metagpt.const import METAGPT_REPORTER_DEFAULT_URL
from metagpt.logs import create_llm_stream_queue, get_llm_stream_queue

if typing.TYPE_CHECKING:
    from metagpt.roles.role import Role

try:
    import requests_unixsocket as requests
except ImportError:
    import requests

from contextvars import ContextVar

CURRENT_ROLE: ContextVar["Role"] = ContextVar("role")


class BlockType(str, Enum):
    """枚举类，表示不同类型的块（如任务、终端等）。"""

    TERMINAL = "Terminal"
    TASK = "Task"
    BROWSER = "Browser"
    BROWSER_RT = "Browser-RT"
    EDITOR = "Editor"
    GALLERY = "Gallery"
    NOTEBOOK = "Notebook"
    DOCS = "Docs"
    THOUGHT = "Thought"


END_MARKER_NAME = "end_marker"
END_MARKER_VALUE = "\x18\x19\x1B\x18\n"


class ResourceReporter(BaseModel):
    """资源报告基类，用于同步和异步报告资源数据。"""

    block: BlockType = Field(description="报告资源的块类型")
    uuid: UUID = Field(default_factory=uuid4, description="资源的唯一标识符")
    enable_llm_stream: bool = Field(False, description="是否连接到LLM流以进行报告")
    callback_url: str = Field(METAGPT_REPORTER_DEFAULT_URL, description="报告将发送到的URL")
    _llm_task: Optional[asyncio.Task] = PrivateAttr(None)

    def report(self, value: Any, name: str, extra: Optional[dict] = None):
        """同步报告资源观察数据。

        参数:
            value: 要报告的数据。
            name: 数据的类型名称。
        """
        return self._report(value, name, extra)

    async def async_report(self, value: Any, name: str, extra: Optional[dict] = None):
        """异步报告资源观察数据。

        参数:
            value: 要报告的数据。
            name: 数据的类型名称。
        """
        return await self._async_report(value, name, extra)

    @classmethod
    def set_report_fn(cls, fn: Callable):
        """设置同步报告函数。

        参数:
            fn: 用于同步报告的可调用函数。例如：

                >>> def _report(self, value: Any, name: str):
                ...     print(value, name)
        """
        cls._report = fn

    @classmethod
    def set_async_report_fn(cls, fn: Callable):
        """设置异步报告函数。

        参数:
            fn: 用于异步报告的可调用函数。例如：

                ```python
                >>> async def _report(self, value: Any, name: str):
                ...     print(value, name)
                ```
        """
        cls._async_report = fn

    def _report(self, value: Any, name: str, extra: Optional[dict] = None):
        if not self.callback_url:
            return

        data = self._format_data(value, name, extra)
        resp = requests.post(self.callback_url, json=data)
        resp.raise_for_status()
        return resp.text

    async def _async_report(self, value: Any, name: str, extra: Optional[dict] = None):
        if not self.callback_url:
            return

        data = self._format_data(value, name, extra)
        url = self.callback_url
        _result = urlparse(url)
        sessiion_kwargs = {}
        if _result.scheme.endswith("+unix"):
            parsed_list = list(_result)
            parsed_list[0] = parsed_list[0][:-5]
            parsed_list[1] = "fake.org"
            url = urlunparse(parsed_list)
            sessiion_kwargs["connector"] = UnixConnector(path=unquote(_result.netloc))

        async with ClientSession(**sessiion_kwargs) as client:
            async with client.post(url, json=data) as resp:
                resp.raise_for_status()
                return await resp.text()

    def _format_data(self, value, name, extra):
        data = self.model_dump(mode="json", exclude=("callback_url", "llm_stream"))
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        elif isinstance(value, Path):
            value = str(value)

        if name == "path":
            value = os.path.abspath(value)
        data["value"] = value
        data["name"] = name
        role = CURRENT_ROLE.get(None)
        if role:
            role_name = role.name
        else:
            role_name = os.environ.get("METAGPT_ROLE")
        data["role"] = role_name
        if extra:
            data["extra"] = extra
        return data

    def __enter__(self):
        """进入同步流报告的回调上下文。"""
        return self

    def __exit__(self, *args, **kwargs):
        """退出同步流报告的回调上下文。"""
        self.report(None, END_MARKER_NAME)

    async def __aenter__(self):
        """进入异步流报告的回调上下文。"""
        if self.enable_llm_stream:
            queue = create_llm_stream_queue()
            self._llm_task = asyncio.create_task(self._llm_stream_report(queue))
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        """退出异步流报告的回调上下文。"""
        if self.enable_llm_stream and exc_type != asyncio.CancelledError:
            await get_llm_stream_queue().put(None)
            await self._llm_task
            self._llm_task = None
        await self.async_report(None, END_MARKER_NAME)

    async def _llm_stream_report(self, queue: asyncio.Queue):
        while True:
            data = await queue.get()
            if data is None:
                return
            await self.async_report(data, "content")

    async def wait_llm_stream_report(self):
        """等待LLM流报告完成。"""
        queue = get_llm_stream_queue()
        while self._llm_task:
            if queue.empty():
                break
            await asyncio.sleep(0.01)

class TerminalReporter(ResourceReporter):
    """终端输出回调，用于命令和输出的流式报告。

    终端有状态，每个代理可以打开多个终端并输入不同的命令。为了正确显示这些状态，每个终端应该有自己唯一的ID，因此在实际应用中，每个终端
    应该实例化自己的 TerminalReporter 对象。
    """

    block: Literal[BlockType.TERMINAL] = BlockType.TERMINAL

    def report(self, value: str, name: Literal["cmd", "output"]):
        """同步报告终端命令或输出."""
        return super().report(value, name)

    async def async_report(self, value: str, name: Literal["cmd", "output"]):
        """异步报告终端命令或输出."""
        return await super().async_report(value, name)


class BrowserReporter(ResourceReporter):
    """浏览器输出回调，用于报告请求的 URL 和页面内容的流式报告。

    浏览器有状态，因此在实际应用中，每个浏览器应该实例化自己的 BrowserReporter 对象。
    """

    block: Literal[BlockType.BROWSER] = BlockType.BROWSER

    def report(self, value: Union[str, SyncPage], name: Literal["url", "page"]):
        """同步报告浏览器的 URL 或页面内容."""
        if name == "page":
            value = {"page_url": value.url, "title": value.title(), "screenshot": str(value.screenshot())}
        return super().report(value, name)

    async def async_report(self, value: Union[str, AsyncPage], name: Literal["url", "page"]):
        """异步报告浏览器的 URL 或页面内容."""
        if name == "page":
            value = {"page_url": value.url, "title": await value.title(), "screenshot": str(await value.screenshot())}
        return await super().async_report(value, name)


class ServerReporter(ResourceReporter):
    """用于服务器部署报告的回调."""

    block: Literal[BlockType.BROWSER_RT] = BlockType.BROWSER_RT

    def report(self, value: str, name: Literal["local_url"] = "local_url"):
        """同步报告服务器部署的 URL."""
        return super().report(value, name)

    async def async_report(self, value: str, name: Literal["local_url"] = "local_url"):
        """异步报告服务器部署的 URL."""
        return await super().async_report(value, name)


class ObjectReporter(ResourceReporter):
    """用于报告完整对象资源的回调."""

    def report(self, value: dict, name: Literal["object"] = "object"):
        """同步报告对象资源."""
        return super().report(value, name)

    async def async_report(self, value: dict, name: Literal["object"] = "object"):
        """异步报告对象资源."""
        return await super().async_report(value, name)


class TaskReporter(ObjectReporter):
    """报告任务块对象资源的回调."""

    block: Literal[BlockType.TASK] = BlockType.TASK


class ThoughtReporter(ObjectReporter):
    """报告思考块对象资源的回调."""

    block: Literal[BlockType.THOUGHT] = BlockType.THOUGHT


class FileReporter(ResourceReporter):
    """文件资源回调，用于报告完整文件路径。

    有两种情况：如果文件需要一次性输出全部内容，使用非流式回调；
    如果文件可以先部分输出以便展示，则使用流式回调。
    """

    def report(
        self,
        value: Union[Path, dict, Any],
        name: Literal["path", "meta", "content"] = "path",
        extra: Optional[dict] = None,
    ):
        """同步报告文件资源."""
        return super().report(value, name, extra)

    async def async_report(
        self,
        value: Union[Path, dict, Any],
        name: Literal["path", "meta", "content"] = "path",
        extra: Optional[dict] = None,
    ):
        """异步报告文件资源."""
        return await super().async_report(value, name, extra)


class NotebookReporter(FileReporter):
    """等同于 FileReporter(block=BlockType.NOTEBOOK)."""

    block: Literal[BlockType.NOTEBOOK] = BlockType.NOTEBOOK


class DocsReporter(FileReporter):
    """等同于 FileReporter(block=BlockType.DOCS)."""

    block: Literal[BlockType.DOCS] = BlockType.DOCS


class EditorReporter(FileReporter):
    """等同于 FileReporter(block=BlockType.EDITOR)."""

    block: Literal[BlockType.EDITOR] = BlockType.EDITOR


class GalleryReporter(FileReporter):
    """图像资源回调，用于报告完整文件路径。

    由于图像需要在显示之前完整，因此每个回调都是完整的文件路径。然而，Gallery 需要展示图像类型和提示信息，
    如果有元数据信息，则应以流式方式报告。
    """

    block: Literal[BlockType.GALLERY] = BlockType.GALLERY

    def report(self, value: Union[dict, Path], name: Literal["meta", "path"] = "path"):
        """同步报告图像资源."""
        return super().report(value, name)

    async def async_report(self, value: Union[dict, Path], name: Literal["meta", "path"] = "path"):
        """异步报告图像资源."""
        return await super().async_report(value, name)
