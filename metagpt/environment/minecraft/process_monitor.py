#!/usr/bin/env python
# -*- coding: utf-8 -*-
# refs to `voyager process_monitor.py`

import re
import subprocess
import threading
import warnings
from typing import List

import psutil

from metagpt.logs import define_log_level


class SubprocessMonitor:
    def __init__(
            self,
            commands: List[str],
            name: str,
            ready_match: str = r".*",  # 默认的匹配正则表达式，用于检查子进程是否准备就绪
            callback_match: str = r"^(?!x)x$",  # 默认的回调匹配正则表达式，不会匹配任何内容
            callback: callable = None,  # 匹配时调用的回调函数
            finished_callback: callable = None,  # 子进程结束时调用的回调函数
    ):
        # 初始化子进程监控器
        self.commands = commands  # 子进程命令列表
        self.name = name  # 子进程名称
        self.logger = define_log_level(name=name)  # 日志记录器
        self.process = None  # 子进程实例
        self.ready_match = ready_match  # 用于判断子进程是否准备就绪的正则表达式
        self.ready_event = None  # 事件标志，用于通知子进程是否准备就绪
        self.ready_line = None  # 子进程输出的准备就绪的行
        self.callback_match = callback_match  # 匹配回调的正则表达式
        self.callback = callback  # 匹配到回调的函数
        self.finished_callback = finished_callback  # 子进程结束后的回调函数
        self.thread = None  # 线程，用于启动子进程

    def _start(self):
        # 启动子进程的实际过程
        self.logger.info(f"Starting subprocess with commands: {self.commands}")

        # 使用 psutil 启动子进程
        self.process = psutil.Popen(
            self.commands,
            stdout=subprocess.PIPE,  # 将标准输出流连接到管道
            stderr=subprocess.STDOUT,  # 将标准错误输出流重定向到标准输出流
            universal_newlines=True,  # 使用文本模式读取输出
        )
        self.logger.info(f"Subprocess {self.name} started with PID {self.process.pid}.")

        # 逐行读取子进程的输出
        for line in iter(self.process.stdout.readline, ""):
            self.logger.info(line.strip())  # 记录输出
            # 如果输出匹配准备就绪的正则表达式，则标记子进程准备就绪
            if re.search(self.ready_match, line):
                self.ready_line = line
                self.logger.info("Subprocess is ready.")
                self.ready_event.set()  # 设置事件标志，表示子进程准备就绪
            # 如果输出匹配回调正则表达式，则调用回调函数
            if re.search(self.callback_match, line):
                self.callback()  # 调用回调函数
        # 如果子进程没有准备就绪，触发警告
        if not self.ready_event.is_set():
            self.ready_event.set()
            warnings.warn(f"Subprocess {self.name} failed to start.")
        # 如果有子进程结束的回调函数，则调用它
        if self.finished_callback:
            self.finished_callback()

    def run(self):
        # 启动并等待子进程准备就绪
        self.ready_event = threading.Event()  # 创建事件标志
        self.ready_line = None  # 重置准备就绪的输出行
        self.thread = threading.Thread(target=self._start)  # 启动子进程的线程
        self.thread.start()  # 启动线程
        self.ready_event.wait()  # 等待子进程准备就绪

    def stop(self):
        # 停止子进程
        self.logger.info("Stopping subprocess.")
        if self.process and self.process.is_running():
            self.process.terminate()  # 终止子进程
            self.process.wait()  # 等待子进程结束

    @property
    def is_running(self):
        # 检查子进程是否正在运行
        if self.process is None:
            return False  # 如果没有子进程，返回 False
        return self.process.is_running()  # 返回子进程的运行状态
