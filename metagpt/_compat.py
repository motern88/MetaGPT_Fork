import platform
import sys
import warnings

# 如果当前 Python 解释器是 CPython 且运行在 Windows 平台上
if sys.implementation.name == "cpython" and platform.system() == "Windows":
    import asyncio

    # 如果 Python 版本是 3.9，则修复 asyncio 在 Windows 上的 _ProactorBasePipeTransport 资源释放问题
    if sys.version_info[:2] == (3, 9):
        from asyncio.proactor_events import _ProactorBasePipeTransport

        # 参考修复方案：https://github.com/python/cpython/pull/92842
        def pacth_del(self, _warn=warnings.warn):
            if self._sock is not None:
                _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)
                self._sock.close()

        # 修改 _ProactorBasePipeTransport 的析构方法，确保资源正确释放
        _ProactorBasePipeTransport.__del__ = pacth_del

    # 如果 Python 版本大于等于 3.9.0，则设置 Windows 平台的 asyncio 事件循环策略
    if sys.version_info >= (3, 9, 0):
        from semantic_kernel.orchestration import sk_function as _  # noqa: F401

        # 由于 https://github.com/microsoft/semantic-kernel/pull/1416 引发的问题
        # 在 Windows 平台上使用 WindowsProactorEventLoopPolicy 作为默认事件循环策略
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
