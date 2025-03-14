#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 14:46
@Author  : alexanderwu
@File    : exceptions.py
"""


import asyncio
import functools
import traceback
from typing import Any, Callable, Tuple, Type, TypeVar, Union

from metagpt.logs import logger

ReturnType = TypeVar("ReturnType")


def handle_exception(
        _func: Callable[..., ReturnType] = None,
        *,
        exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        exception_msg: str = "",
        default_return: Any = None,
) -> Callable[..., ReturnType]:
    """处理异常并返回默认值。

    该装饰器用于捕获函数中的异常并进行处理，当指定的异常发生时，记录错误日志并返回一个默认值。

    参数:
        _func (Callable[..., ReturnType], optional): 要装饰的函数。如果未提供，将返回一个装饰器。
        exception_type (Union[Type[Exception], Tuple[Type[Exception], ...]], optional): 需要捕获的异常类型，默认捕获所有的 Exception 类型。如果传递多个异常类型，装饰器会捕获这些类型的异常。
        exception_msg (str, optional): 发生异常时的附加消息，默认是空字符串。
        default_return (Any, optional): 发生异常时返回的默认值，默认值为 None。

    返回:
        Callable[..., ReturnType]: 一个函数装饰器，能够处理异常并返回默认值。

    示例:
        @handle_exception(exception_type=ValueError, exception_msg="An error occurred", default_return="default")
        async def my_func():
            # 可能会抛出异常的代码
            pass

        该装饰器会捕获 ValueError 异常，记录错误日志，并返回 "default"。
    """

    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                logger.opt(depth=1).error(
                    f"{e}: {exception_msg}, "
                    f"\nCalling {func.__name__} with args: {args}, kwargs: {kwargs} "
                    f"\nStack: {traceback.format_exc()}"
                )
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                logger.opt(depth=1).error(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs} failed: {e}, "
                    f"stack: {traceback.format_exc()}"
                )
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
