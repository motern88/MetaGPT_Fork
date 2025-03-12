#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/8 14:03
@Author  : alexanderwu
@File    : document.py
@Desc    : Classes and Operations Related to Files in the File System.
"""
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PDFReader
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from metagpt.logs import logger
from metagpt.repo_parser import RepoParser


def validate_cols(content_col: str, df: pd.DataFrame):
    """
    验证 DataFrame 是否包含指定的内容列。

    参数:
        content_col (str): 需要检查的列名。
        df (pd.DataFrame): 需要检查的 Pandas DataFrame。

    异常:
        ValueError: 如果 `content_col` 不在 DataFrame 的列中，则抛出异常。
    """
    if content_col not in df.columns:
        raise ValueError("Content column not found in DataFrame.")


def read_data(data_path: Path) -> Union[pd.DataFrame, list[Document]]:
    """
    读取不同格式的数据文件，并返回 DataFrame 或 Document 列表。

    参数:
        data_path (Path): 数据文件的路径。

    返回:
        Union[pd.DataFrame, list[Document]]: 解析后的数据，可能是 DataFrame（对于结构化数据）或 Document 列表（对于非结构化文本）。

    异常:
        NotImplementedError: 如果文件格式不受支持，则抛出异常。
    """
    suffix = data_path.suffix
    if ".xlsx" == suffix:
        data = pd.read_excel(data_path)
    elif ".csv" == suffix:
        data = pd.read_csv(data_path)
    elif ".json" == suffix:
        data = pd.read_json(data_path)
    elif suffix in (".docx", ".doc"):
        data = SimpleDirectoryReader(input_files=[str(data_path)]).load_data()
    elif ".txt" == suffix:
        data = SimpleDirectoryReader(input_files=[str(data_path)]).load_data()
        node_parser = SimpleNodeParser.from_defaults(separator="\n", chunk_size=256, chunk_overlap=0)
        data = node_parser.get_nodes_from_documents(data)
    elif ".pdf" == suffix:
        data = PDFReader.load_data(str(data_path))
    else:
        raise NotImplementedError("File format not supported.")
    return data


class DocumentStatus(Enum):
    """
    文档状态枚举，类似于 RFC/PEP 中的状态机制。

    可选状态:
        - DRAFT: 草稿状态
        - UNDERREVIEW: 审核中
        - APPROVED: 审核通过
        - DONE: 完成
    """

    DRAFT = "draft"
    UNDERREVIEW = "underreview"
    APPROVED = "approved"
    DONE = "done"


class Document(BaseModel):
    """
    Document 类：用于处理与文档相关的操作。

    属性:
        path (Path): 文档文件的路径。
        name (str): 文档名称。
        content (str): 文档内容。
        author (str): 文档作者。
        status (DocumentStatus): 文档状态，默认值为 DRAFT。
        reviews (list): 记录文档的审核信息。
    """

    path: Path = Field(default=None)
    name: str = Field(default="")
    content: str = Field(default="")

    author: str = Field(default="")
    status: DocumentStatus = Field(default=DocumentStatus.DRAFT)
    reviews: list = Field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path):
        """
        从文件路径创建 Document 实例。

        参数:
            path (Path): 文件路径。

        返回:
            Document: 解析后的 Document 实例。

        异常:
            FileNotFoundError: 如果文件不存在，则抛出异常。
        """
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found.")
        content = path.read_text()
        return cls(content=content, path=path)

    @classmethod
    def from_text(cls, text: str, path: Optional[Path] = None):
        """
        通过文本字符串创建 Document 实例。

        参数:
            text (str): 文档内容。
            path (Optional[Path]): 关联的文件路径（可选）。

        返回:
            Document: 生成的 Document 实例。
        """
        return cls(content=text, path=path)

    def to_path(self, path: Optional[Path] = None):
        """
        将文档内容保存到指定文件路径。

        参数:
            path (Optional[Path]): 目标文件路径（可选）。

        异常:
            ValueError: 如果未提供路径，则抛出异常。
        """
        if path is not None:
            self.path = path

        if self.path is None:
            raise ValueError("File path is not set.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: 未来可扩展支持 Excel、CSV、JSON 等格式的存储
        self.path.write_text(self.content, encoding="utf-8")

    def persist(self):
        """
        持久化文档到磁盘。

        返回:
            None
        """
        return self.to_path()


class IndexableDocument(Document):
    """
    可索引文档类：适用于向量数据库或搜索引擎的高级文档处理。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Union[pd.DataFrame, list]
    content_col: Optional[str] = Field(default="")  # 主要内容所在的列
    meta_col: Optional[str] = Field(default="")  # 元数据所在的列

    @classmethod
    def from_path(cls, data_path: Path, content_col="content", meta_col="metadata"):
        """
        从文件路径创建 IndexableDocument 实例。

        参数：
            data_path (Path): 数据文件路径。
            content_col (str): 主要内容列的名称。
            meta_col (str): 元数据列的名称。

        返回：
            IndexableDocument 实例。
        """
        if not data_path.exists():
            raise FileNotFoundError(f"文件 {data_path} 未找到。")

        data = read_data(data_path)
        if isinstance(data, pd.DataFrame):
            validate_cols(content_col, data)
            return cls(data=data, content=str(data), content_col=content_col, meta_col=meta_col)

        try:
            content = data_path.read_text()
        except Exception as e:
            logger.debug(f"加载 {str(data_path)} 出错: {e}")
            content = ""

        return cls(data=data, content=content, content_col=content_col, meta_col=meta_col)

    def _get_docs_and_metadatas_by_df(self) -> (list, list):
        """
        从 DataFrame 提取文档内容和元数据。
        """
        df = self.data
        docs = []
        metadatas = []
        for i in tqdm(range(len(df))):
            docs.append(df[self.content_col].iloc[i])
            if self.meta_col:
                metadatas.append({self.meta_col: df[self.meta_col].iloc[i]})
            else:
                metadatas.append({})
        return docs, metadatas

    def _get_docs_and_metadatas_by_llamaindex(self) -> (list, list):
        """
        从 LlamaIndex 解析的文档列表提取内容和元数据。
        """
        data = self.data
        docs = [i.text for i in data]
        metadatas = [i.metadata for i in data]
        return docs, metadatas

    def get_docs_and_metadatas(self) -> (list, list):
        """
        根据数据类型获取文档和元数据。
        """
        if isinstance(self.data, pd.DataFrame):
            return self._get_docs_and_metadatas_by_df()
        elif isinstance(self.data, list):
            return self._get_docs_and_metadatas_by_llamaindex()
        else:
            raise NotImplementedError("不支持的元数据提取数据类型。")


class RepoMetadata(BaseModel):
    """
    存储仓库的元信息。
    """
    name: str = Field(default="")  # 仓库名称
    n_docs: int = Field(default=0)  # 文档数量
    n_chars: int = Field(default=0)  # 文档字符总数
    symbols: list = Field(default_factory=list)  # 代码符号列表


class Repo(BaseModel):
    """
    仓库类：管理文档、代码和资源文件。
    """
    name: str = Field(default="")  # 仓库名称
    docs: dict[Path, Document] = Field(default_factory=dict)  # 纯文本文档
    codes: dict[Path, Document] = Field(default_factory=dict)  # 代码文件
    assets: dict[Path, Document] = Field(default_factory=dict)  # 资源文件
    path: Path = Field(default=None)  # 仓库路径

    def _path(self, filename):
        """获取指定文件的完整路径。"""
        return self.path / filename

    @classmethod
    def from_path(cls, path: Path):
        """
        从文件系统加载仓库。
        """
        path.mkdir(parents=True, exist_ok=True)
        repo = Repo(path=path, name=path.name)
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".json", ".txt", ".md", ".py", ".js", ".css", ".html"]:
                repo._set(file_path.read_text(), file_path)
        return repo

    def to_path(self):
        """
        持久化所有文档、代码和资源文件到磁盘。
        """
        for doc in self.docs.values():
            doc.to_path()
        for code in self.codes.values():
            code.to_path()
        for asset in self.assets.values():
            asset.to_path()

    def _set(self, content: str, path: Path):
        """
        将文档添加到适当的分类。
        """
        suffix = path.suffix
        doc = Document(content=content, path=path, name=str(path.relative_to(self.path)))

        if suffix.lower() == ".md":
            self.docs[path] = doc
        elif suffix.lower() in [".py", ".js", ".css", ".html"]:
            self.codes[path] = doc
        else:
            self.assets[path] = doc
        return doc

    def set(self, filename: str, content: str):
        """
        设置文档并持久化到磁盘。
        """
        path = self._path(filename)
        doc = self._set(content, path)
        doc.to_path()

    def get(self, filename: str) -> Optional[Document]:
        """
        根据文件名获取文档。
        """
        path = self._path(filename)
        return self.docs.get(path) or self.codes.get(path) or self.assets.get(path)

    def get_text_documents(self) -> list[Document]:
        """
        获取所有文本和代码文档。
        """
        return list(self.docs.values()) + list(self.codes.values())

    def eda(self) -> RepoMetadata:
        """
        计算仓库的元数据信息。
        """
        n_docs = sum(len(i) for i in [self.docs, self.codes, self.assets])
        n_chars = sum(sum(len(j.content) for j in i.values()) for i in [self.docs, self.codes, self.assets])
        symbols = RepoParser(base_directory=self.path).generate_symbols()
        return RepoMetadata(name=self.name, n_docs=n_docs, n_chars=n_chars, symbols=symbols)
