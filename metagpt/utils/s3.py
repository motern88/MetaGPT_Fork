import base64
import os.path
import traceback
import uuid
from pathlib import Path
from typing import Optional

import aioboto3
import aiofiles

from metagpt.config2 import S3Config
from metagpt.const import BASE64_FORMAT
from metagpt.logs import logger


class S3:
    """一个用于与Amazon S3存储交互的类。"""

    def __init__(self, config: S3Config):
        """初始化S3类的实例，配置S3连接信息。"""
        self.session = aioboto3.Session()  # 创建异步会话
        self.config = config
        self.auth_config = {
            "service_name": "s3",  # 服务名称
            "aws_access_key_id": config.access_key,  # AWS访问密钥
            "aws_secret_access_key": config.secret_key,  # AWS秘密访问密钥
            "endpoint_url": config.endpoint,  # S3的端点URL
        }

    async def upload_file(
        self,
        bucket: str,
        local_path: str,
        object_name: str,
    ) -> None:
        """将本地文件上传到指定的S3存储桶。

        参数:
            bucket: S3存储桶的名称。
            local_path: 本地文件路径（包括文件名）。
            object_name: 上传到S3的文件路径（包括文件名）。

        异常:
            如果上传过程中发生错误，抛出异常。
        """
        try:
            async with self.session.client(**self.auth_config) as client:
                async with aiofiles.open(local_path, mode="rb") as reader:
                    body = await reader.read()
                    await client.put_object(Body=body, Bucket=bucket, Key=object_name)
                    logger.info(f"成功将文件上传到 S3 存储桶 {bucket} 的路径 {object_name}.")
        except Exception as e:
            logger.error(f"上传文件失败: {e}")
            raise e

    async def get_object_url(
        self,
        bucket: str,
        object_name: str,
    ) -> str:
        """获取S3存储桶中指定文件的可下载或预览URL。

        参数:
            bucket: S3存储桶的名称。
            object_name: 文件在S3中的完整路径（包括文件名）。

        返回:
            文件的URL。

        异常:
            如果获取URL时发生错误，抛出异常。
        """
        try:
            async with self.session.client(**self.auth_config) as client:
                file = await client.get_object(Bucket=bucket, Key=object_name)
                return str(file["Body"].url)
        except Exception as e:
            logger.error(f"获取文件URL失败: {e}")
            raise e

    async def get_object(
        self,
        bucket: str,
        object_name: str,
    ) -> bytes:
        """获取存储在S3中的文件的二进制数据。

        参数:
            bucket: S3存储桶的名称。
            object_name: 文件在S3中的完整路径（包括文件名）。

        返回:
            文件的二进制数据。

        异常:
            如果获取文件数据时发生错误，抛出异常。
        """
        try:
            async with self.session.client(**self.auth_config) as client:
                s3_object = await client.get_object(Bucket=bucket, Key=object_name)
                return await s3_object["Body"].read()
        except Exception as e:
            logger.error(f"获取文件数据失败: {e}")
            raise e

    async def download_file(
        self, bucket: str, object_name: str, local_path: str, chunk_size: Optional[int] = 128 * 1024
    ) -> None:
        """从S3下载文件到本地。

        参数:
            bucket: S3存储桶的名称。
            object_name: 文件在S3中的完整路径（包括文件名）。
            local_path: 下载到本地的文件路径。
            chunk_size: 每次读取和写入的数据块大小，默认128 KB。

        异常:
            如果下载过程中发生错误，抛出异常。
        """
        try:
            async with self.session.client(**self.auth_config) as client:
                s3_object = await client.get_object(Bucket=bucket, Key=object_name)
                stream = s3_object["Body"]
                async with aiofiles.open(local_path, mode="wb") as writer:
                    while True:
                        file_data = await stream.read(chunk_size)
                        if not file_data:
                            break
                        await writer.write(file_data)
        except Exception as e:
            logger.error(f"从S3下载文件失败: {e}")
            raise e

    async def cache(self, data: str, file_ext: str, format: str = "") -> str:
        """将数据保存到远程S3并返回URL。

        参数:
            data: 要保存的数据。
            file_ext: 文件扩展名，如".txt"。
            format: 数据的格式（例如BASE64格式）。

        返回:
            S3上文件的URL。

        异常:
            如果保存数据到S3时发生错误，抛出异常。
        """
        object_name = uuid.uuid4().hex + file_ext  # 使用UUID生成唯一文件名
        path = Path(__file__).parent
        pathname = path / object_name
        try:
            # 将数据保存到本地文件
            async with aiofiles.open(str(pathname), mode="wb") as file:
                data = base64.b64decode(data) if format == BASE64_FORMAT else data.encode(encoding="utf-8")
                await file.write(data)

            # 上传到S3
            bucket = self.config.bucket
            object_pathname = self.config.bucket or "system"
            object_pathname += f"/{object_name}"
            object_pathname = os.path.normpath(object_pathname)
            await self.upload_file(bucket=bucket, local_path=str(pathname), object_name=object_pathname)
            pathname.unlink(missing_ok=True)

            # 返回文件URL
            return await self.get_object_url(bucket=bucket, object_name=object_pathname)
        except Exception as e:
            logger.exception(f"错误发生: {e}, 堆栈信息: {traceback.format_exc()}")
            pathname.unlink(missing_ok=True)
            return None
