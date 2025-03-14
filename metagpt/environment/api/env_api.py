#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :  the environment api store

from typing import Any, Callable, Union

from pydantic import BaseModel, Field


class EnvAPIAbstract(BaseModel):
    """API/接口的摘要描述"""

    api_name: str = Field(default="", description="API函数的名称或ID")
    args: set = Field(default={}, description="API函数的`args`参数")
    kwargs: dict = Field(default=dict(), description="API函数的`kwargs`参数")


class EnvAPIRegistry(BaseModel):
    """用于存储环境中读写API/接口的注册表"""

    registry: dict[str, Callable] = Field(default=dict(), exclude=True)

    def get(self, api_name: str):
        """根据API名称获取API函数"""
        if api_name not in self.registry:
            raise KeyError(f"API名称: {api_name} 未找到")
        return self.registry.get(api_name)

    def __getitem__(self, api_name: str) -> Callable:
        """通过索引访问API函数"""
        return self.get(api_name)

    def __setitem__(self, api_name: str, func: Callable):
        """将API函数添加到注册表中"""
        self.registry[api_name] = func

    def __len__(self):
        """返回注册表中API函数的数量"""
        return len(self.registry)

    def get_apis(self, as_str=True) -> dict[str, dict[str, Union[dict, Any, str]]]:
        """返回API函数的schema，而不是函数实例"""
        apis = dict()
        for func_name, func_schema in self.registry.items():
            new_func_schema = dict()
            for key, value in func_schema.items():
                if key == "func":
                    continue  # 跳过函数本身
                new_func_schema[key] = str(value) if as_str else value
            apis[func_name] = new_func_schema
        return apis


class WriteAPIRegistry(EnvAPIRegistry):
    """专门用于写操作的API注册表"""

    pass


class ReadAPIRegistry(EnvAPIRegistry):
    """专门用于读操作的API注册表"""

    pass
