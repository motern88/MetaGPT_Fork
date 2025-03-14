#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : file.py
@Describe : General file operations.
"""
import base64
from pathlib import Path
from typing import Optional, Tuple, Union

import aiofiles
from fsspec.implementations.memory import MemoryFileSystem as _MemoryFileSystem

from metagpt.config2 import config
from metagpt.logs import logger
from metagpt.utils import read_docx
from metagpt.utils.common import aread, aread_bin, awrite_bin, check_http_endpoint
from metagpt.utils.exceptions import handle_exception
from metagpt.utils.repo_to_markdown import is_text_file


class File:
    """文件操作的通用工具类。"""

    CHUNK_SIZE = 64 * 1024  # 默认文件读取块大小（64KB）

    @classmethod
    @handle_exception
    async def write(cls, root_path: Path, filename: str, content: bytes) -> Path:
        """将文件内容写入本地指定路径。

        参数:
            root_path: 文件的根路径，例如 "/data"。
            filename: 文件名，例如 "test.txt"。
            content: 文件的二进制内容。

        返回:
            返回文件的完整路径，例如 "/data/test.txt"。

        异常:
            Exception: 如果在文件写入过程中发生意外错误。
        """
        root_path.mkdir(parents=True, exist_ok=True)  # 确保根路径存在
        full_path = root_path / filename  # 拼接文件完整路径
        async with aiofiles.open(full_path, mode="wb") as writer:
            await writer.write(content)  # 异步写入文件内容
            logger.debug(f"Successfully write file: {full_path}")  # 写入成功日志
            return full_path

    @classmethod
    @handle_exception
    async def read(cls, file_path: Path, chunk_size: int = None) -> bytes:
        """按块读取文件内容。

        参数:
            file_path: 文件的完整路径，例如 "/data/test.txt"。
            chunk_size: 每块读取的大小（默认是64KB）。

        返回:
            文件的二进制内容。

        异常:
            Exception: 如果在文件读取过程中发生意外错误。
        """
        chunk_size = chunk_size or cls.CHUNK_SIZE  # 默认使用64KB的块大小
        async with aiofiles.open(file_path, mode="rb") as reader:
            chunks = list()
            while True:
                chunk = await reader.read(chunk_size)  # 读取指定大小的块
                if not chunk:
                    break  # 如果没有更多数据，则退出循环
                chunks.append(chunk)
            content = b"".join(chunks)  # 合并所有块
            logger.debug(f"Successfully read file, the path of file: {file_path}")  # 读取成功日志
            return content

    @staticmethod
    async def is_textual_file(filename: Union[str, Path]) -> bool:
        """判断给定的文件是否为文本文件。

        文件被认为是文本文件，如果它是纯文本文件或具有特定MIME类型的文件，
        例如PDF和Microsoft Word文档。

        参数:
            filename (Union[str, Path]): 要检查的文件路径。

        返回:
            bool: 如果文件是文本文件则返回True，否则返回False。
        """
        is_text, mime_type = await is_text_file(filename)  # 检查文件是否为文本文件
        if is_text:
            return True
        if mime_type == "application/pdf":
            return True
        if mime_type in {
            "application/msword",  # Word文档类型
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Word文档类型
            "application/vnd.ms-word.document.macroEnabled.12",  # 启用了宏的Word文档类型
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template",  # Word模板类型
            "application/vnd.ms-word.template.macroEnabled.12",  # 启用了宏的Word模板类型
        }:
            return True
        return False

    @staticmethod
    async def read_text_file(filename: Union[str, Path]) -> Optional[str]:
        """读取文件的全部内容，支持文本、PDF和Word文件。

        参数:
            filename (Union[str, Path]): 文件路径。

        返回:
            str: 文件的文本内容，如果文件无法读取则返回None。
        """
        is_text, mime_type = await is_text_file(filename)  # 检查文件类型
        if is_text:
            return await File._read_text(filename)  # 读取纯文本文件
        if mime_type == "application/pdf":
            return await File._read_pdf(filename)  # 读取PDF文件
        if mime_type in {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-word.document.macroEnabled.12",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
            "application/vnd.ms-word.template.macroEnabled.12",
        }:
            return await File._read_docx(filename)  # 读取Word文档
        return None

    @staticmethod
    async def _read_text(path: Union[str, Path]) -> str:
        return await aread(path)  # 使用异步读取方法读取文本文件内容

    @staticmethod
    async def _read_pdf(path: Union[str, Path]) -> str:
        result = await File._omniparse_read_file(path)  # 尝试使用OmniParse解析文件
        if result:
            return result

        from llama_index.readers.file import PDFReader  # PDF读取器

        reader = PDFReader()
        lines = reader.load_data(file=Path(path))  # 读取PDF文件数据
        return "\n".join([i.text for i in lines])  # 返回合并后的PDF内容

    @staticmethod
    async def _read_docx(path: Union[str, Path]) -> str:
        result = await File._omniparse_read_file(path)  # 尝试使用OmniParse解析文件
        if result:
            return result
        return "\n".join(read_docx(str(path)))  # 使用docx读取器读取Word文档

    @staticmethod
    async def _omniparse_read_file(path: Union[str, Path], auto_save_image: bool = False) -> Optional[str]:
        from metagpt.tools.libs import get_env_default
        from metagpt.utils.omniparse_client import OmniParseClient

        env_base_url = await get_env_default(key="base_url", app_name="OmniParse", default_value="")
        env_timeout = await get_env_default(key="timeout", app_name="OmniParse", default_value="")
        conf_base_url, conf_timeout = await File._read_omniparse_config()

        base_url = env_base_url or conf_base_url
        if not base_url:
            return None
        api_key = await get_env_default(key="api_key", app_name="OmniParse", default_value="")
        timeout = env_timeout or conf_timeout or 600
        try:
            timeout = int(timeout)
        except ValueError:
            timeout = 600

        try:
            if not await check_http_endpoint(url=base_url):  # 检查API端点是否可用
                logger.warning(f"{base_url}: NOT AVAILABLE")
                return None
            client = OmniParseClient(api_key=api_key, base_url=base_url, max_timeout=timeout)
            file_data = await aread_bin(filename=path)  # 读取二进制文件内容
            ret = await client.parse_document(file_input=file_data, bytes_filename=str(path))  # 使用OmniParse解析文件
        except (ValueError, Exception) as e:
            logger.exception(f"{path}: {e}")
            return None
        if not ret.images or not auto_save_image:  # 如果没有图片或不保存图片，则返回文本内容
            return ret.text

        result = [ret.text]
        img_dir = Path(path).parent / (Path(path).name.replace(".", "_") + "_images")  # 图片存储目录
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in ret.images:
            byte_data = base64.b64decode(i.image)  # 解码图片
            filename = img_dir / i.image_name
            await awrite_bin(filename=filename, data=byte_data)  # 保存图片
            result.append(f"![{i.image_name}]({str(filename)})")
        return "\n".join(result)  # 返回文本和图片的合并结果

    @staticmethod
    async def _read_omniparse_config() -> Tuple[str, int]:
        if config.omniparse and config.omniparse.base_url:
            return config.omniparse.base_url, config.omniparse.timeout
        return "", 0


class MemoryFileSystem(_MemoryFileSystem):
    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(str(path))  # 去除协议部分
