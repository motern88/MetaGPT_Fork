# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 10:00
# @Author  : leiwu30
# @File    : stream_pipe.py
# @Version : None
# @Description : None

import json
import time
from multiprocessing import Pipe


class StreamPipe:
    def __init__(self, name=None):
        """
        初始化 StreamPipe 对象。

        Args:
        - name (str, optional): 设置 StreamPipe 的名称，默认为 None。
        """
        self.name = name
        # 创建一个管道，用于父进程和子进程之间的通信
        self.parent_conn, self.child_conn = Pipe()
        self.finish: bool = False

    # 用于模拟一个格式化的数据模板，通常用于发送流数据
    format_data = {
        "id": "chatcmpl-96bVnBOOyPFZZxEoTIGbdpFcVEnur",
        "object": "chat.completion.chunk",
        "created": 1711361191,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_3bc1b5746c",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "content"}, "logprobs": None, "finish_reason": None}
        ],
    }

    def set_message(self, msg):
        """
        通过父连接发送消息。

        Args:
        - msg (any): 需要发送的消息内容。
        """
        self.parent_conn.send(msg)

    def get_message(self, timeout: int = 3):
        """
        从子连接接收消息，如果超时则返回 None。

        Args:
        - timeout (int, optional): 等待消息的最大时间，默认为 3 秒。

        Returns:
        - msg (any): 接收到的消息内容，若超时则返回 None。
        """
        if self.child_conn.poll(timeout):
            return self.child_conn.recv()
        else:
            return None

    def msg2stream(self, msg):
        """
        将消息格式化为流数据格式。

        Args:
        - msg (str): 需要发送的消息内容。

        Returns:
        - str: 格式化后的流数据，包含时间戳和消息内容。
        """
        self.format_data["created"] = int(time.time())
        self.format_data["choices"][0]["delta"]["content"] = msg
        return f"data: {json.dumps(self.format_data, ensure_ascii=False)}\n".encode("utf-8")