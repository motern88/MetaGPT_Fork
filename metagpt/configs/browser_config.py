#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 19:06
@Author  : alexanderwu
@File    : browser_config.py
"""
from enum import Enum
from typing import Literal

from metagpt.utils.yaml_model import YamlModel


class WebBrowserEngineType(Enum):
    PLAYWRIGHT = "playwright"  # 使用 Playwright 引擎
    SELENIUM = "selenium"      # 使用 Selenium 引擎
    CUSTOM = "custom"          # 自定义引擎

    @classmethod
    def __missing__(cls, key):
        """默认类型转换，当传入的类型无法匹配时，返回 CUSTOM 类型"""
        return cls.CUSTOM


class BrowserConfig(YamlModel):
    """浏览器配置类，用于指定浏览器的引擎和类型"""

    engine: WebBrowserEngineType = WebBrowserEngineType.PLAYWRIGHT  # 默认使用 Playwright 引擎
    browser_type: Literal["chromium", "firefox", "webkit", "chrome", "firefox", "edge", "ie"] = "chromium"
    """浏览器类型配置
    如果引擎是 Playwright，浏览器类型应为 "chromium", "firefox" 或 "webkit" 之一。
    如果引擎是 Selenium，浏览器类型应为 "chrome", "firefox", "edge" 或 "ie" 之一。
    """
