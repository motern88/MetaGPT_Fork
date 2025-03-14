#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/28 00:01
@Author  : alexanderwu
@File    : base_store.py
"""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseStore(ABC):
    """基础存储类，供其他存储类继承

    该类定义了存储操作的基本接口，包括搜索、写入和添加数据。
    """

    @abstractmethod
    def search(self, *args, **kwargs):
        """搜索数据的抽象方法

        具体的子类需要实现搜索操作。

        Args:
            *args: 可变参数
            **kwargs: 可变关键字参数

        Returns:
            搜索结果
        """
        raise NotImplementedError

    @abstractmethod
    def write(self, *args, **kwargs):
        """写入数据的抽象方法

        具体的子类需要实现写入操作。

        Args:
            *args: 可变参数
            **kwargs: 可变关键字参数

        Returns:
            写入的结果
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, *args, **kwargs):
        """添加数据的抽象方法

        具体的子类需要实现添加操作。

        Args:
            *args: 可变参数
            **kwargs: 可变关键字参数

        Returns:
            添加的数据
        """
        raise NotImplementedError


class LocalStore(BaseStore, ABC):
    """本地存储类，继承自 BaseStore，实现本地数据的加载、写入、添加等操作

    该类用于操作本地存储，包括加载原始数据、缓存文件以及存储数据。
    """

    def __init__(self, raw_data_path: Path, cache_dir: Path = None):
        """初始化本地存储

        如果没有提供缓存目录，则默认为原始数据的父目录。初始化时，加载存储的数据，
        如果没有数据，则调用写入方法生成新的存储数据。

        Args:
            raw_data_path (Path): 原始数据文件路径
            cache_dir (Path, optional): 缓存目录，默认为 None。若为 None，则使用原始数据的父目录
        """
        if not raw_data_path:
            raise FileNotFoundError  # 如果没有原始数据路径，抛出文件未找到异常
        self.raw_data_path = raw_data_path
        self.fname = self.raw_data_path.stem  # 获取原始数据的文件名（不包含扩展名）
        if not cache_dir:
            cache_dir = raw_data_path.parent  # 如果没有提供缓存目录，默认使用原始数据的父目录
        self.cache_dir = cache_dir
        self.store = self._load()  # 加载存储的数据
        if not self.store:
            self.store = self.write()  # 如果存储数据为空，调用写入方法

    def _get_index_and_store_fname(self, index_ext=".json", docstore_ext=".json"):
        """获取索引文件和存储文件的文件名

        该方法生成并返回索引文件和文档存储文件的路径。

        Args:
            index_ext (str): 索引文件扩展名，默认为 ".json"
            docstore_ext (str): 文档存储文件扩展名，默认为 ".json"

        Returns:
            tuple: 包含索引文件路径和存储文件路径的元组
        """
        index_file = self.cache_dir / "default__vector_store" / index_ext
        store_file = self.cache_dir / "docstore" / docstore_ext
        return index_file, store_file

    @abstractmethod
    def _load(self):
        """加载存储数据的抽象方法

        具体的子类需要实现该方法来加载存储的数据。

        Returns:
            存储的数据
        """
        raise NotImplementedError

    @abstractmethod
    def _write(self, docs, metadatas):
        """写入数据的抽象方法

        具体的子类需要实现该方法来写入文档数据和元数据。

        Args:
            docs: 文档数据
            metadatas: 元数据
        """
        raise NotImplementedError
