#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 19:07
@Author  : alexanderwu
@File    : mermaid_config.py
"""
from typing import Literal

from metagpt.utils.yaml_model import YamlModel


class MermaidConfig(YamlModel):
    """Mermaid 配置类

    该类用于配置生成 Mermaid 图表的工具和相关设置。

    属性：
    engine: str - 设置使用的引擎，支持的选项包括 "nodejs"、"ink"、"playwright"、"pyppeteer" 和 "none"。
    path: str - 指定生成 Mermaid 图表的可执行文件路径，默认为 "mmdc"。
    puppeteer_config: str - Puppeteer 配置文件路径，默认为空字符串。
    pyppeteer_path: str - pyppeteer 所需的 Chrome 浏览器路径，默认为 "/usr/bin/google-chrome-stable"。
    """

    engine: Literal["nodejs", "ink", "playwright", "pyppeteer", "none"] = "nodejs"  # 使用的引擎类型
    path: str = "mmdc"  # 可执行文件路径，默认为 mmdc
    puppeteer_config: str = ""  # Puppeteer 配置文件路径
    pyppeteer_path: str = "/usr/bin/google-chrome-stable"  # pyppeteer 使用的 Chrome 浏览器路径
