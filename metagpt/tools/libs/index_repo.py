#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import tiktoken
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field, model_validator

from metagpt.config2 import config
from metagpt.context import Context
from metagpt.logs import logger
from metagpt.rag.engines import SimpleEngine
from metagpt.rag.factories.embedding import RAGEmbeddingFactory
from metagpt.rag.schema import FAISSIndexConfig, FAISSRetrieverConfig, LLMRankerConfig
from metagpt.utils.common import aread, awrite, generate_fingerprint, list_files
from metagpt.utils.file import File
from metagpt.utils.report import EditorReporter

UPLOADS_INDEX_ROOT = "/data/.index/uploads"  # 上传文件的索引存储路径
DEFAULT_INDEX_ROOT = UPLOADS_INDEX_ROOT  # 默认索引存储路径
UPLOAD_ROOT = "/data/uploads"  # 上传文件的存储路径
DEFAULT_ROOT = UPLOAD_ROOT  # 默认存储路径
CHATS_INDEX_ROOT = "/data/.index/chats"  # 聊天记录的索引存储路径
CHATS_ROOT = "/data/chats/"  # 聊天记录存储路径
OTHER_TYPE = "other"  # 其他文件类型标识

DEFAULT_MIN_TOKEN_COUNT = 10000  # 默认最小 token 数量
DEFAULT_MAX_TOKEN_COUNT = 100000000  # 默认最大 token 数量

class IndexRepoMeta(BaseModel):
    min_token_count: int  # 最小 token 计数
    max_token_count: int  # 最大 token 计数

class TextScore(BaseModel):
    filename: str  # 文件名
    text: str  # 文本内容
    score: Optional[float] = None  # 评分，可选

class IndexRepo(BaseModel):
    persist_path: str = DEFAULT_INDEX_ROOT  # 索引库的持久化路径
    root_path: str = DEFAULT_ROOT  # 文件的根路径
    fingerprint_filename: str = "fingerprint.json"  # 指纹文件名
    meta_filename: str = "meta.json"  # 元数据文件名
    model: Optional[str] = None  # 预训练模型，可选
    min_token_count: int = DEFAULT_MIN_TOKEN_COUNT  # 最小 token 计数
    max_token_count: int = DEFAULT_MAX_TOKEN_COUNT  # 最大 token 计数
    recall_count: int = 5  # 召回的最大结果数
    embedding: Optional[BaseEmbedding] = Field(default=None, exclude=True)  # 词向量模型
    fingerprints: Dict[str, str] = Field(default_factory=dict)  # 文件指纹存储

    @model_validator(mode="after")
    def _update_fingerprints(self) -> "IndexRepo":
        """如果指纹数据未加载，则从指纹文件中加载。"""
        if not self.fingerprints:
            filename = Path(self.persist_path) / self.fingerprint_filename
            if not filename.exists():
                return self
            with open(str(filename), "r") as reader:
                self.fingerprints = json.load(reader)
        return self

    async def search(
        self, query: str, filenames: Optional[List[Path]] = None
    ) -> Optional[List[Union[NodeWithScore, TextScore]]]:
        """搜索与查询相关的文档。"""
        encoding = tiktoken.get_encoding("cl100k_base")
        result: List[Union[NodeWithScore, TextScore]] = []
        filenames, excludes = await self._filter(filenames)  # 过滤文件类型
        if not filenames:
            raise ValueError(f"不支持的文件类型: {[str(i) for i in excludes]}")
        resource = EditorReporter()
        for i in filenames:
            await resource.async_report(str(i), "path")
        filter_filenames = set()
        meta = await self._read_meta()
        new_files = {}
        for i in filenames:
            if Path(i).suffix.lower() in {".pdf", ".doc", ".docx"}:  # 处理特定格式的文档
                if str(i) not in self.fingerprints:
                    new_files[i] = ""
                    logger.warning(f'文件 "{i}" 未被索引')
                filter_filenames.add(str(i))
                continue
            content = await File.read_text_file(i)
            token_count = len(encoding.encode(content))
            if not self._is_buildable(token_count, meta.min_token_count, meta.max_token_count):
                result.append(TextScore(filename=str(i), text=content))
                continue
            file_fingerprint = generate_fingerprint(content)
            if str(i) not in self.fingerprints or (self.fingerprints.get(str(i)) != file_fingerprint):
                new_files[i] = content
                logger.warning(f'文件 "{i}" 发生变更但未被索引')
                continue
            filter_filenames.add(str(i))
        if new_files:
            added, others = await self.add(paths=list(new_files.keys()), file_datas=new_files)
            filter_filenames.update([str(i) for i in added])
            for i in others:
                result.append(TextScore(filename=str(i), text=new_files.get(i)))
                filter_filenames.discard(str(i))
        nodes = await self._search(query=query, filters=filter_filenames)
        return result + nodes

    async def merge(
        self, query: str, indices_list: List[List[Union[NodeWithScore, TextScore]]]
    ) -> List[Union[NodeWithScore, TextScore]]:
        """合并多个索引的搜索结果，并根据查询排序。"""
        flat_nodes = [node for indices in indices_list if indices for node in indices if node]
        if len(flat_nodes) <= self.recall_count:
            return flat_nodes

        if not self.embedding:
            if self.model:
                config.embedding.model = self.model
            factory = RAGEmbeddingFactory(config)
            self.embedding = factory.get_rag_embedding()

        scores = []
        query_embedding = await self.embedding.aget_text_embedding(query)
        for i in flat_nodes:
            try:
                text_embedding = await self.embedding.aget_text_embedding(i.text)
            except Exception as e:
                tenth = int(len(i.text) / 10)
                logger.warning(
                    f"{e}, 十分之一长度={tenth}, 前部分长度={len(i.text[: tenth * 6])}, 后部分长度={len(i.text[tenth * 4:])}"
                )
                pre_win_part = await self.embedding.aget_text_embedding(i.text[: tenth * 6])
                post_win_part = await self.embedding.aget_text_embedding(i.text[tenth * 4 :])
                similarity = max(
                    self.embedding.similarity(query_embedding, pre_win_part),
                    self.embedding.similarity(query_embedding, post_win_part),
                )
                scores.append((similarity, i))
                continue
            similarity = self.embedding.similarity(query_embedding, text_embedding)
            scores.append((similarity, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [i[1] for i in scores][: self.recall_count]


    async def add(
        self, paths: List[Path], file_datas: Dict[Union[str, Path], str] = None
    ) -> Tuple[List[str], List[str]]:
        """向索引中添加新文档。

        参数:
            paths (List[Path]): 需要添加的文档路径列表。
            file_datas (Dict[Union[str, Path], str]): 文件内容的字典，可选。

        返回:
            Tuple[List[str], List[str]]: 包含两个列表的元组：
                1. 成功添加到索引的文件名列表。
                2. 由于不可构建而未添加到索引的文件名列表。
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        filenames, _ = await self._filter(paths)  # 过滤无效文件
        filter_filenames = []  # 可构建的文件
        delete_filenames = []  # 需删除的文件
        file_datas = file_datas or {}

        for i in filenames:
            content = file_datas.get(i) or await File.read_text_file(i)  # 读取文件内容
            file_datas[i] = content
            if not self._is_fingerprint_changed(filename=i, content=content):
                continue  # 文件内容未变化，跳过
            token_count = len(encoding.encode(content))  # 计算 token 数
            if self._is_buildable(token_count):  # 判断是否可构建
                filter_filenames.append(i)
                logger.debug(f"{i} 可构建: {token_count}, 允许范围 {self.min_token_count}~{self.max_token_count}")
            else:
                delete_filenames.append(i)
                logger.debug(f"{i} 不可构建: {token_count}, 允许范围 {self.min_token_count}~{self.max_token_count}")

        await self._add_batch(filenames=filter_filenames, delete_filenames=delete_filenames, file_datas=file_datas)
        return filter_filenames, delete_filenames

    async def _add_batch(
        self,
        filenames: List[Union[str, Path]],
        delete_filenames: List[Union[str, Path]],
        file_datas: Dict[Union[str, Path], str],
    ):
        """批量添加和删除文档。

        参数:
            filenames (List[Union[str, Path]]): 需要添加的文件列表。
            delete_filenames (List[Union[str, Path]]): 需要删除的文件列表。
        """
        if not filenames:
            return
        logger.info(f"更新索引库, 添加 {filenames}, 删除 {delete_filenames}")
        engine = None
        Context()
        if Path(self.persist_path).exists():  # 如果索引已存在
            logger.debug(f"从 {self.persist_path} 加载索引")
            engine = SimpleEngine.from_index(
                index_config=FAISSIndexConfig(persist_path=self.persist_path),
                retriever_configs=[FAISSRetrieverConfig()],
            )
            try:
                engine.delete_docs(filenames + delete_filenames)  # 删除旧文档
                logger.info(f"删除文档 {filenames + delete_filenames}")
                engine.add_docs(input_files=filenames)  # 添加新文档
                logger.info(f"添加文档 {filenames}")
            except NotImplementedError as e:  # 如果删除失败，则重建索引
                logger.debug(f"{e}")
                filenames = list(set([str(i) for i in filenames] + list(self.fingerprints.keys())))
                engine = None
                logger.info(f"{e}。重新构建索引。")

        if not engine:  # 如果索引未初始化，则创建新索引
            engine = SimpleEngine.from_docs(
                input_files=[str(i) for i in filenames],
                retriever_configs=[FAISSRetrieverConfig()],
                ranker_configs=[LLMRankerConfig()],
            )
            logger.info(f"添加文档 {filenames}")

        engine.persist(persist_dir=self.persist_path)  # 持久化索引
        for i in filenames:
            content = file_datas.get(i) or await File.read_text_file(i)
            fp = generate_fingerprint(content)
            self.fingerprints[str(i)] = fp

        await awrite(filename=Path(self.persist_path) / self.fingerprint_filename, data=json.dumps(self.fingerprints))
        await self._save_meta()

    def _is_buildable(self, token_count: int, min_token_count: int = -1, max_token_count=-1) -> bool:
        """判断文档是否可构建。

        参数:
            token_count (int): 文档的 token 数量。

        返回:
            bool: 如果文档在允许的 token 范围内，则返回 True，否则返回 False。
        """
        min_token_count = min_token_count if min_token_count >= 0 else self.min_token_count
        max_token_count = max_token_count if max_token_count >= 0 else self.max_token_count
        return min_token_count <= token_count <= max_token_count

    async def _filter(self, filenames: Optional[List[Union[str, Path]]] = None) -> (List[Path], List[Path]):
        """过滤无效文件，仅保留有效的文本文件。

        参数:
            filenames (Optional[List[Union[str, Path]]]): 需要筛选的文件列表。

        返回:
            Tuple[List[Path], List[Path]]: (有效的文件列表, 被排除的文件列表)
        """
        root_path = Path(self.root_path).absolute()
        if not filenames:
            filenames = [root_path]
        pathnames = []
        excludes = []

        for i in filenames:
            path = Path(i).absolute()
            if not path.is_relative_to(root_path):  # 只允许处理 root 目录下的文件
                excludes.append(path)
                logger.debug(f"{path} 不属于 {root_path} 范围内")
                continue

            if not path.is_dir():  # 判断是否是文本文件
                is_text = await File.is_textual_file(path)
                if is_text:
                    pathnames.append(path)
                continue

            subfiles = list_files(path)  # 递归获取文件夹内的文件
            for j in subfiles:
                is_text = await File.is_textual_file(j)
                if is_text:
                    pathnames.append(j)

        logger.debug(f"有效文件: {pathnames}, 排除: {excludes}")
        return pathnames, excludes

    async def _search(self, query: str, filters: Set[str]) -> List[NodeWithScore]:
        """执行查询，并返回匹配的文档。

        参数:
            query (str): 查询内容。
            filters (Set[str]): 需要过滤的文件路径集合。

        返回:
            List[NodeWithScore]: 匹配的文档列表，包含分数。
        """
        if not filters:
            return []
        if not Path(self.persist_path).exists():
            raise ValueError(f"索引库 {Path(self.persist_path).name} 不存在。")
        Context()
        engine = SimpleEngine.from_index(
            index_config=FAISSIndexConfig(persist_path=self.persist_path),
            retriever_configs=[FAISSRetrieverConfig()],
        )
        rsp = await engine.aretrieve(query)  # 执行查询
        return [i for i in rsp if i.metadata.get("file_path") in filters]  # 过滤结果，仅返回符合条件的

    def _is_fingerprint_changed(self, filename: Union[str, Path], content: str) -> bool:
        """检查给定文档内容的指纹是否已更改。

        Args:
            filename (Union[str, Path]): 文档的文件名。
            content (str): 文档的内容。

        Returns:
            bool: 如果指纹已更改，则返回 True；否则返回 False。
        """
        old_fp = self.fingerprints.get(str(filename))
        if not old_fp:
            return True
        fp = generate_fingerprint(content)
        return old_fp != fp

    @staticmethod
    def find_index_repo_path(files: List[Union[str, Path]]) -> Tuple[Dict[str, Set[Path]], Dict[str, str]]:
        """将文件路径映射到对应的索引库路径。

        Args:
            files (List[Union[str, Path]]): 需要分类的文件路径列表，可以是字符串或 Path 对象。

        Returns:
            Tuple[Dict[str, Set[Path]], Dict[str, str]]:
                - 一个字典，将索引库路径映射到文件集合。
                - 一个字典，将索引库路径映射到它们对应的根目录。
        """
        mappings = {
            UPLOADS_INDEX_ROOT: re.compile(r"^/data/uploads($|/.*)"),  # 上传文件索引库
            CHATS_INDEX_ROOT: re.compile(r"^/data/chats/[a-z0-9]+($|/.*)"),  # 聊天记录索引库
        }

        clusters = {}  # 存储不同索引库对应的文件路径集合
        roots = {}  # 存储索引库的根目录
        for i in files:
            path = Path(i).absolute()
            path_type = OTHER_TYPE  # 默认类型

            # 根据正则匹配文件路径，确定文件所属的索引库
            for type_, pattern in mappings.items():
                if re.match(pattern, str(i)):
                    path_type = type_
                    break

            # 处理聊天记录索引库
            if path_type == CHATS_INDEX_ROOT:
                chat_id = path.parts[3]  # 获取聊天 ID
                path_type = str(Path(path_type) / chat_id)
                roots[path_type] = str(Path(CHATS_ROOT) / chat_id)

            # 处理上传文件索引库
            elif path_type == UPLOADS_INDEX_ROOT:
                roots[path_type] = UPLOAD_ROOT

            # 归类文件到对应索引库
            if path_type in clusters:
                clusters[path_type].add(path)
            else:
                clusters[path_type] = {path}

        return clusters, roots

    async def _save_meta(self):
        """保存索引库的元数据信息。"""
        meta = IndexRepoMeta(min_token_count=self.min_token_count, max_token_count=self.max_token_count)
        await awrite(filename=Path(self.persist_path) / self.meta_filename, data=meta.model_dump_json())

    async def _read_meta(self) -> IndexRepoMeta:
        """读取索引库的元数据信息。

        Returns:
            IndexRepoMeta: 解析后的索引库元数据对象。
        """
        default_meta = IndexRepoMeta(min_token_count=self.min_token_count, max_token_count=self.max_token_count)

        filename = Path(self.persist_path) / self.meta_filename
        if not filename.exists():
            return default_meta  # 如果元数据文件不存在，返回默认值
        meta_data = await aread(filename=filename)
        try:
            meta = IndexRepoMeta.model_validate_json(meta_data)
            return meta
        except Exception as e:
            logger.warning(f"加载元数据出错: {e}")
        return default_meta  # 解析失败时返回默认值

    @staticmethod
    async def cross_repo_search(query: str, file_or_path: Union[str, Path]) -> List[str]:
        """在多个索引库中搜索指定查询内容。

        该异步函数会在给定的文件或目录中搜索指定的查询内容。

        Args:
            query (str): 需要搜索的查询字符串。
            file_or_path (Union[str, Path]): 文件或目录路径，可为字符串或 Path 对象。

        Returns:
            List[str]: 包含查询结果的文件路径列表。

        Raises:
            ValueError: 如果查询路径不存在，则抛出异常。
        """
        if not file_or_path or not Path(file_or_path).exists():
            raise ValueError(f'"{str(file_or_path)}" 不存在')

        # 如果输入路径是文件，则直接加入列表；如果是目录，则获取目录下的所有文件
        files = [file_or_path] if not Path(file_or_path).is_dir() else list_files(file_or_path)
        clusters, roots = IndexRepo.find_index_repo_path(files)  # 分类文件路径

        futures = []  # 异步任务列表
        others = set()  # 其他类型文件集合

        # 遍历分类后的文件，执行不同的搜索策略
        for persist_path, filenames in clusters.items():
            if persist_path == OTHER_TYPE:
                others.update(filenames)  # 其他类型的文件，直接存入 `others` 集合
                continue
            root = roots[persist_path]  # 获取索引库根目录
            repo = IndexRepo(persist_path=persist_path, root_path=root)  # 创建索引库对象
            futures.append(repo.search(query=query, filenames=list(filenames)))  # 添加异步搜索任务

        # 对 `others` 集合中的文件进行单独读取
        for i in others:
            futures.append(File.read_text_file(i))

        futures_results = []
        if futures:
            futures_results = await asyncio.gather(*futures)  # 并发执行所有异步任务

        result = []  # 存储普通文本结果
        v_result = []  # 存储索引库查询结果
        for i in futures_results:
            if not i:
                continue
            if isinstance(i, str):
                result.append(i)  # 文本结果存入 `result`
            else:
                v_result.append(i)  # 索引库结果存入 `v_result`

        repo = IndexRepo()  # 创建索引库对象
        merged = await repo.merge(query=query, indices_list=v_result)  # 合并索引库查询结果

        return [i.text for i in merged] + result  # 返回最终的搜索结果