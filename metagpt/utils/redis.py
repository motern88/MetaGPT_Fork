# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/27
@Author  : mashenquan
@File    : redis.py
"""
from __future__ import annotations

import traceback
from datetime import timedelta

import redis.asyncio as aioredis

from metagpt.configs.redis_config import RedisConfig
from metagpt.logs import logger


class Redis:
    def __init__(self, config: RedisConfig = None):
        """
        初始化 Redis 客户端。

        参数:
            config (RedisConfig): Redis 配置对象，包含连接参数。
        """
        self.config = config
        self._client = None  # 用于存储 Redis 客户端实例

    async def _connect(self, force=False):
        """
        异步连接到 Redis 服务器。

        参数:
            force (bool): 是否强制重新连接，即使已存在连接。

        返回:
            bool: 是否成功连接。
        """
        # 如果已连接且不强制重新连接，直接返回 True
        if self._client and not force:
            return True

        try:
            # 使用 aioredis 连接 Redis
            self._client = await aioredis.from_url(
                self.config.to_url(),
                username=self.config.username,
                password=self.config.password,
                db=self.config.db,
            )
            return True
        except Exception as e:
            logger.warning(f"Redis 初始化失败: {e}")
        return False

    async def get(self, key: str) -> bytes | None:
        """
        从 Redis 获取指定键的值。

        参数:
            key (str): Redis 键。

        返回:
            bytes | None: 如果成功，返回存储的数据；如果失败，返回 None。
        """
        # 确保连接已建立且键不为空
        if not await self._connect() or not key:
            return None
        try:
            # 从 Redis 获取键值
            v = await self._client.get(key)
            return v
        except Exception as e:
            logger.exception(f"获取数据失败: {e}, stack: {traceback.format_exc()}")
            return None

    async def set(self, key: str, data: str, timeout_sec: int = None):
        """
        将数据存储到 Redis。

        参数:
            key (str): Redis 键。
            data (str): 要存储的数据。
            timeout_sec (int): 键值对的过期时间，单位秒。如果为 None，则表示永不过期。
        """
        # 确保连接已建立且键不为空
        if not await self._connect() or not key:
            return
        try:
            # 如果设置过期时间，则转换为 timedelta 格式
            ex = None if not timeout_sec else timedelta(seconds=timeout_sec)
            # 设置 Redis 键值对
            await self._client.set(key, data, ex=ex)
        except Exception as e:
            logger.exception(f"设置数据失败: {e}, stack: {traceback.format_exc()}")

    async def close(self):
        """
        关闭 Redis 连接。
        """
        if not self._client:
            return
        # 关闭 Redis 客户端连接
        await self._client.close()
        self._client = None