#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 19:06
@Author  : alexanderwu
@File    : redis_config.py
"""
from metagpt.utils.yaml_model import YamlModelWithoutDefault


class RedisConfig(YamlModelWithoutDefault):
    """
    Redis 配置类，用于存储和管理 Redis 连接相关的配置信息。

    属性：
        host (str): Redis 服务器的主机地址。
        port (int): Redis 服务器的端口号。
        username (str): Redis 服务器的用户名，默认为空字符串。
        password (str): Redis 服务器的密码。
        db (str): Redis 使用的数据库名称或编号。

    方法：
        to_url(): 将 Redis 配置信息转换为连接 URL 字符串。
        to_kwargs(): 将 Redis 配置信息转换为字典，适用于通过关键字参数传递给 Redis 连接函数。
    """

    host: str  # Redis 服务器的主机地址
    port: int  # Redis 服务器的端口号
    username: str = ""  # Redis 服务器的用户名，默认为空字符串
    password: str  # Redis 服务器的密码
    db: str  # Redis 使用的数据库名称或编号

    def to_url(self):
        """
        将 Redis 配置信息转换为连接 URL 字符串。

        返回：
            str: 格式化后的 Redis 连接 URL。
        """
        return f"redis://{self.host}:{self.port}"

    def to_kwargs(self):
        """
        将 Redis 配置信息转换为字典，适用于通过关键字参数传递给 Redis 连接函数。

        返回：
            dict: 包含 Redis 配置信息的字典。
        """
        return {
            "username": self.username,
            "password": self.password,
            "db": self.db,
        }
