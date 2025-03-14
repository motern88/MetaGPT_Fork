import mimetypes
from pathlib import Path
from typing import Union

import httpx

from metagpt.rag.schema import OmniParsedResult
from metagpt.utils.common import aread_bin


class OmniParseClient:
    """
    OmniParse 服务器客户端
    该客户端与 OmniParse 服务器交互，用于解析不同类型的媒体文件和文档。

    OmniParse API 文档: https://docs.cognitivelab.in/api

    属性:
        ALLOWED_DOCUMENT_EXTENSIONS (set): 支持的文档文件扩展名集合。
        ALLOWED_AUDIO_EXTENSIONS (set): 支持的音频文件扩展名集合。
        ALLOWED_VIDEO_EXTENSIONS (set): 支持的视频文件扩展名集合。
    """

    ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".ppt", ".pptx", ".doc", ".docx"}
    ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac"}
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}

    def __init__(self, api_key: str = None, base_url: str = "http://localhost:8000", max_timeout: int = 120):
        """
        初始化 OmniParse 客户端。

        参数:
            api_key: 默认为 None，可以用于身份验证。
            base_url: API 的基础 URL。
            max_timeout: 最大请求超时时间（秒）。
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_timeout = max_timeout

        self.parse_media_endpoint = "/parse_media"
        self.parse_website_endpoint = "/parse_website"
        self.parse_document_endpoint = "/parse_document"

    async def _request_parse(
        self,
        endpoint: str,
        method: str = "POST",
        files: dict = None,
        params: dict = None,
        data: dict = None,
        json: dict = None,
        headers: dict = None,
        **kwargs,
    ) -> dict:
        """
        请求 OmniParse API 解析文档。

        参数:
            endpoint (str): API 的 endpoint。
            method (str, 可选): 使用的 HTTP 方法，默认为 "POST"。
            files (dict, 可选): 请求中包含的文件。
            params (dict, 可选): 查询字符串参数。
            data (dict, 可选): 请求体中的表单数据。
            json (dict, 可选): 请求体中的 JSON 数据。
            headers (dict, 可选): 请求中包含的 HTTP 头。
            **kwargs: 其他 httpx.AsyncClient.request() 的额外关键字参数。

        返回:
            dict: JSON 格式的响应数据。
        """
        url = f"{self.base_url}{endpoint}"
        method = method.upper()
        headers = headers or {}
        _headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update(**_headers)
        async with httpx.AsyncClient() as client:
            response = await client.request(
                url=url,
                method=method,
                files=files,
                params=params,
                json=json,
                data=data,
                headers=headers,
                timeout=self.max_timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()

    async def parse_document(self, file_input: Union[str, bytes, Path], bytes_filename: str = None) -> OmniParsedResult:
        """
        解析文档类型的数据（支持 ".pdf", ".ppt", ".pptx", ".doc", ".docx"）。

        参数:
            file_input: 文件路径或文件字节数据。
            bytes_filename: 字节数据的文件名，有助于确定 HTTP 请求的 MIME 类型。

        异常:
            ValueError: 如果文件扩展名不被允许。

        返回:
            OmniParsedResult: 文档解析的结果。
        """
        self.verify_file_ext(file_input, self.ALLOWED_DOCUMENT_EXTENSIONS, bytes_filename)
        file_info = await self.get_file_info(file_input, bytes_filename)
        resp = await self._request_parse(self.parse_document_endpoint, files={"file": file_info})
        data = OmniParsedResult(**resp)
        return data

    async def parse_pdf(self, file_input: Union[str, bytes, Path]) -> OmniParsedResult:
        """
        解析 PDF 文档。

        参数:
            file_input: 文件路径或文件字节数据。

        异常:
            ValueError: 如果文件扩展名不被允许。

        返回:
            OmniParsedResult: PDF 解析的结果。
        """
        self.verify_file_ext(file_input, {".pdf"})
        # parse_pdf 仅支持接受文件的字节数据进行解析。
        file_info = await self.get_file_info(file_input, only_bytes=True)
        endpoint = f"{self.parse_document_endpoint}/pdf"
        resp = await self._request_parse(endpoint=endpoint, files={"file": file_info})
        data = OmniParsedResult(**resp)
        return data

    async def parse_video(self, file_input: Union[str, bytes, Path], bytes_filename: str = None) -> dict:
        """
        解析视频类型的数据（支持 ".mp4", ".mkv", ".avi", ".mov"）。

        参数:
            file_input: 文件路径或文件字节数据。
            bytes_filename: 字节数据的文件名，有助于确定 HTTP 请求的 MIME 类型。

        异常:
            ValueError: 如果文件扩展名不被允许。

        返回:
            dict: JSON 格式的响应数据。
        """
        self.verify_file_ext(file_input, self.ALLOWED_VIDEO_EXTENSIONS, bytes_filename)
        file_info = await self.get_file_info(file_input, bytes_filename)
        return await self._request_parse(f"{self.parse_media_endpoint}/video", files={"file": file_info})

    async def parse_audio(self, file_input: Union[str, bytes, Path], bytes_filename: str = None) -> dict:
        """
        解析音频类型的数据（支持 ".mp3", ".wav", ".aac"）。

        参数:
            file_input: 文件路径或文件字节数据。
            bytes_filename: 字节数据的文件名，有助于确定 HTTP 请求的 MIME 类型。

        异常:
            ValueError: 如果文件扩展名不被允许。

        返回:
            dict: JSON 格式的响应数据。
        """
        self.verify_file_ext(file_input, self.ALLOWED_AUDIO_EXTENSIONS, bytes_filename)
        file_info = await self.get_file_info(file_input, bytes_filename)
        return await self._request_parse(f"{self.parse_media_endpoint}/audio", files={"file": file_info})

    @staticmethod
    def verify_file_ext(file_input: Union[str, bytes, Path], allowed_file_extensions: set, bytes_filename: str = None):
        """
        验证文件扩展名。

        参数:
            file_input: 文件路径或文件字节数据。
            allowed_file_extensions: 允许的文件扩展名集合。
            bytes_filename: 用于字节数据验证的文件名。

        异常:
            ValueError: 如果文件扩展名不被允许。

        返回:
        """
        verify_file_path = None
        if isinstance(file_input, (str, Path)):
            verify_file_path = str(file_input)
        elif isinstance(file_input, bytes) and bytes_filename:
            verify_file_path = bytes_filename

        if not verify_file_path:
            # 如果仅提供字节数据，则不进行验证
            return

        file_ext = Path(verify_file_path).suffix.lower()
        if file_ext not in allowed_file_extensions:
            raise ValueError(f"不允许的 {file_ext} 文件扩展名，必须是 {allowed_file_extensions} 中的一种")

    @staticmethod
    async def get_file_info(
        file_input: Union[str, bytes, Path],
        bytes_filename: str = None,
        only_bytes: bool = False,
    ) -> Union[bytes, tuple]:
        """
        获取文件信息。

        参数:
            file_input: 文件路径或文件字节数据。
            bytes_filename: 用于上传字节数据时的文件名，有助于确定 MIME 类型。
            only_bytes: 是否仅返回字节数据。默认为 False，返回元组。

        异常:
            ValueError: 如果在传递字节数据时未提供 bytes_filename，或者文件输入类型无效。

        返回:
            [bytes, tuple] 返回字节数据（如果 only_bytes 为 True），否则返回元组（文件名、文件字节数据、MIME 类型）。
        """
        if isinstance(file_input, (str, Path)):
            filename = Path(file_input).name
            file_bytes = await aread_bin(file_input)

            if only_bytes:
                return file_bytes

            mime_type = mimetypes.guess_type(file_input)[0]
            return filename, file_bytes, mime_type
        elif isinstance(file_input, bytes):
            if only_bytes:
                return file_input
            if not bytes_filename:
                raise ValueError("传递字节数据时必须设置 bytes_filename")

            mime_type = mimetypes.guess_type(bytes_filename)[0]
            return bytes_filename, file_input, mime_type
        else:
            raise ValueError("file_input 必须是字符串（文件路径）或字节数据。")
