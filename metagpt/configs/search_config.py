#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 19:06
@Author  : alexanderwu
@File    : search_config.py
"""
from enum import Enum
from typing import Callable, Optional

from pydantic import ConfigDict, Field

from metagpt.utils.yaml_model import YamlModel


class SearchEngineType(Enum):
    """搜索引擎类型枚举"""
    SERPAPI_GOOGLE = "serpapi"  # 使用 SerpAPI 搜索引擎
    SERPER_GOOGLE = "serper"  # 使用 Serper 搜索引擎
    DIRECT_GOOGLE = "google"  # 使用 Google 搜索引擎
    DUCK_DUCK_GO = "ddg"  # 使用 DuckDuckGo 搜索引擎
    CUSTOM_ENGINE = "custom"  # 使用自定义搜索引擎
    BING = "bing"  # 使用 Bing 搜索引擎


class SearchConfig(YamlModel):
    """搜索配置类

    该类用于配置搜索引擎的API设置和其他相关配置。

    属性：
        model_config (ConfigDict): 配置字典，可以通过额外的字段来扩展。
        api_type (SearchEngineType): 搜索引擎的类型，默认为 DuckDuckGo。
        api_key (str): 用于认证的API密钥。
        cse_id (str): Google的自定义搜索引擎ID。
        search_func (Optional[Callable]): 可选的自定义搜索函数。
        params (dict): 用于配置搜索引擎的额外参数，默认为Google配置。
    """

    model_config = ConfigDict(extra="allow")  # 允许扩展额外的配置字段

    api_type: SearchEngineType = SearchEngineType.DUCK_DUCK_GO  # 搜索引擎类型，默认使用 DuckDuckGo
    api_key: str = ""  # 搜索引擎API密钥
    cse_id: str = ""  # Google自定义搜索引擎ID
    search_func: Optional[Callable] = None  # 自定义搜索函数，可选
    params: dict = Field(
        default_factory=lambda: {
            "engine": "google",  # 默认使用Google引擎
            "google_domain": "google.com",  # Google域名
            "gl": "us",  # 地理位置，默认为美国
            "hl": "en",  # 语言，默认为英文
        }
    )  # 搜索引擎的额外参数
