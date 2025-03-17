#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : async_sse_client to make keep the use of Event to access response
#           refs to `zhipuai/core/_sse_client.py`

import json
from typing import Any, Iterator


class AsyncSSEClient(object):
    def __init__(self, event_source: Iterator[Any]):
        """
        初始化异步 SSE 客户端。

        参数：
            event_source (Iterator[Any]): 事件源，通常是一个异步迭代器，提供 SSE 数据流。
        """
        self._event_source = event_source

    async def stream(self) -> dict:
        """
        异步流式处理事件数据。

        如果事件源是字节类型，将抛出运行时错误并提示错误消息。

        返回：
            dict: 从事件数据中解析出的 JSON 数据。

        异常：
            - 如果事件源是字节类型，将抛出 `RuntimeError`。
        """
        if isinstance(self._event_source, bytes):
            raise RuntimeError(
                f"请求失败，错误信息: {self._event_source.decode('utf-8')}, 请参考 `https://open.bigmodel.cn/dev/api#error-code-v3`"
            )

        # 异步迭代事件源中的每一个数据块
        async for chunk in self._event_source:
            line = chunk.data.decode("utf-8")

            # 跳过空行和以冒号开头的行
            if line.startswith(":") or not line:
                return

            field, _p, value = line.partition(":")

            # 清理值中的前导空格
            if value.startswith(" "):
                value = value[1:]

            # 如果字段是 "data"，则处理其中的内容
            if field == "data":
                # 遇到结束标记 [DONE]，则跳出循环
                if value.startswith("[DONE]"):
                    break
                # 解析数据并返回
                data = json.loads(value)
                yield data
