import asyncio
from fileinput import FileInput
from pathlib import Path
from typing import List, Optional, Union

from llama_index.core import Document
from llama_index.core.async_utils import run_jobs
from llama_index.core.readers.base import BaseReader

from metagpt.logs import logger
from metagpt.rag.schema import OmniParseOptions, OmniParseType, ParseResultType
from metagpt.utils.async_helper import NestAsyncio
from metagpt.utils.omniparse_client import OmniParseClient


class OmniParse(BaseReader):
    """OmniParse 解析器"""

    def __init__(
        self, api_key: str = None, base_url: str = "http://localhost:8000", parse_options: OmniParseOptions = None
    ):
        """
        初始化方法。

        参数:
            api_key: 默认为 None，稍后可以用于身份验证。
            base_url: OmniParse API 的基础 URL，默认为 "http://localhost:8000"。
            parse_options: OmniParse 的可选设置，默认为 OmniParseOptions 的默认值。
        """
        self.parse_options = parse_options or OmniParseOptions()
        self.omniparse_client = OmniParseClient(api_key, base_url, max_timeout=self.parse_options.max_timeout)

    @property
    def parse_type(self):
        """返回解析类型"""
        return self.parse_options.parse_type

    @property
    def result_type(self):
        """返回结果类型"""
        return self.parse_options.result_type

    @parse_type.setter
    def parse_type(self, parse_type: Union[str, OmniParseType]):
        """
        设置解析类型

        参数:
            parse_type: 解析类型，可以是字符串或 OmniParseType 枚举值。
        """
        if isinstance(parse_type, str):
            parse_type = OmniParseType(parse_type)
        self.parse_options.parse_type = parse_type

    @result_type.setter
    def result_type(self, result_type: Union[str, ParseResultType]):
        """
        设置结果类型

        参数:
            result_type: 结果类型，可以是字符串或 ParseResultType 枚举值。
        """
        if isinstance(result_type, str):
            result_type = ParseResultType(result_type)
        self.parse_options.result_type = result_type

    async def _aload_data(
        self,
        file_path: Union[str, bytes, Path],
        extra_info: Optional[dict] = None,
    ) -> List[Document]:
        """
        从输入的文件路径加载数据。

        参数:
            file_path: 文件路径或文件字节数据。
            extra_info: 可选的包含额外信息的字典。

        返回:
            List[Document]: 解析后的文档列表
        """
        try:
            if self.parse_type == OmniParseType.PDF:
                # 解析 PDF 文件
                parsed_result = await self.omniparse_client.parse_pdf(file_path)
            else:
                # 解析其他文档类型
                # 对于兼容字节数据，需要额外的文件名
                extra_info = extra_info or {}
                filename = extra_info.get("filename")
                parsed_result = await self.omniparse_client.parse_document(file_path, bytes_filename=filename)

            # 获取指定类型的结构化数据
            content = getattr(parsed_result, self.result_type)
            docs = [
                Document(
                    text=content,
                    metadata=extra_info or {},
                )
            ]
        except Exception as e:
            logger.error(f"OMNI 解析错误: {e}")
            docs = []

        return docs

    async def aload_data(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
    ) -> List[Document]:
        """
        加载数据，支持单个文件或多个文件的处理。

        参数:
            file_path: 文件路径或文件字节数据。
            extra_info: 可选的包含额外信息的字典。

        注意:
            此方法最终调用 _aload_data 进行数据处理。

        返回:
            List[Document]: 解析后的文档列表
        """
        docs = []
        if isinstance(file_path, (str, bytes, Path)):
            # 处理单个文件
            docs = await self._aload_data(file_path, extra_info)
        elif isinstance(file_path, list):
            # 同时处理多个文件
            parse_jobs = [self._aload_data(file_item, extra_info) for file_item in file_path]
            doc_ret_list = await run_jobs(jobs=parse_jobs, workers=self.parse_options.num_workers)
            docs = [doc for docs in doc_ret_list for doc in docs]
        return docs

    def load_data(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
    ) -> List[Document]:
        """
        从输入的文件路径加载数据。

        参数:
            file_path: 文件路径或文件字节数据。
            extra_info: 可选的包含额外信息的字典。

        注意:
            此方法最终调用 aload_data 进行数据处理。

        返回:
            List[Document]: 解析后的文档列表
        """
        NestAsyncio.apply_once()  # 确保兼容嵌套的异步调用
        return asyncio.run(self.aload_data(file_path, extra_info))
