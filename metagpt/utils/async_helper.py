import asyncio
import threading
from typing import Any


def run_coroutine_in_new_loop(coroutine) -> Any:
    """
    在一个新的独立事件循环中运行协程（在不同的线程中）。

    这个函数适用于在同步函数中执行异步函数时，遇到 `RuntimeError: This event loop is already running` 错误的情况。

    参数：
        coroutine: 需要执行的异步协程对象。

    返回：
        Any: 协程执行后的返回结果。
    """
    # 创建一个新的事件循环
    new_loop = asyncio.new_event_loop()
    # 在新线程中运行事件循环
    t = threading.Thread(target=lambda: new_loop.run_forever())
    t.start()

    # 通过线程安全的方式在新的事件循环中运行协程
    future = asyncio.run_coroutine_threadsafe(coroutine, new_loop)

    try:
        return future.result()  # 获取协程执行结果
    finally:
        # 让新线程中的事件循环停止
        new_loop.call_soon_threadsafe(new_loop.stop)
        t.join()  # 等待线程结束
        new_loop.close()  # 关闭事件循环


class NestAsyncio:
    """允许 asyncio 事件循环重新进入（解决嵌套运行问题）。"""

    is_applied = False  # 记录是否已经应用过

    @classmethod
    def apply_once(cls):
        """
        确保 `nest_asyncio.apply()` 仅被调用一次，以避免重复应用。
        """
        if not cls.is_applied:
            import nest_asyncio

            nest_asyncio.apply()  # 允许在已有事件循环中嵌套运行新的事件循环
            cls.is_applied = True  # 标记为已应用
