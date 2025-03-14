from dataclasses import dataclass
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, VectorParams

from metagpt.document_store.base_store import BaseStore


@dataclass
class QdrantConnection:
    """
    Qdrant 连接配置类

    Args:
        url: Qdrant 服务的 URL
        host: Qdrant 服务的主机地址
        port: Qdrant 服务的端口
        memory: 是否使用内存模式，默认是 False
        api_key: Qdrant 云服务的 API 密钥
    """

    url: str = None  # Qdrant 服务的 URL
    host: str = None  # Qdrant 服务的主机地址
    port: int = None  # Qdrant 服务的端口
    memory: bool = False  # 是否使用内存模式
    api_key: str = None  # Qdrant 云服务的 API 密钥


class QdrantStore(BaseStore):
    """
    Qdrant 存储类，用于与 Qdrant 数据库进行交互，支持创建集合、插入数据、查询等操作
    """

    def __init__(self, connect: QdrantConnection):
        """初始化 QdrantStore 对象

        根据连接配置（QdrantConnection），创建不同的 Qdrant 客户端实例。

        Args:
            connect (QdrantConnection): 包含 Qdrant 连接信息的对象
        """
        if connect.memory:
            self.client = QdrantClient(":memory:")  # 如果使用内存模式
        elif connect.url:
            self.client = QdrantClient(url=connect.url, api_key=connect.api_key)  # 通过 URL 连接 Qdrant
        elif connect.host and connect.port:
            self.client = QdrantClient(host=connect.host, port=connect.port, api_key=connect.api_key)  # 通过 host 和 port 连接 Qdrant
        else:
            raise Exception("please check QdrantConnection.")  # 如果连接信息不完整，抛出异常

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams,
        force_recreate=False,
        **kwargs,
    ):
        """创建 Qdrant 集合

        Args:
            collection_name (str): 集合名称
            vectors_config (VectorParams): 向量配置对象，详情请参见 https://github.com/qdrant/qdrant-client
            force_recreate (bool, optional): 是否强制重新创建集合，默认为 False。如果为 True，将会删除已存在的集合并重新创建。
            **kwargs: 其他参数

        Returns:
            bool or dict: 返回创建结果，若创建成功则返回 True，否则返回异常信息
        """
        try:
            # 检查集合是否存在
            self.client.get_collection(collection_name)
            if force_recreate:  # 如果需要强制重新创建集合
                res = self.client.recreate_collection(collection_name, vectors_config=vectors_config, **kwargs)
                return res
            return True
        except:  # noqa: E722 捕获异常
            # 如果集合不存在或发生异常，重新创建集合
            return self.client.recreate_collection(collection_name, vectors_config=vectors_config, **kwargs)

    def has_collection(self, collection_name: str):
        """检查集合是否存在

        Args:
            collection_name (str): 集合名称

        Returns:
            bool: 如果集合存在，返回 True，否则返回 False
        """
        try:
            self.client.get_collection(collection_name)
            return True
        except:  # noqa: E722 捕获异常
            return False

    def delete_collection(self, collection_name: str, timeout=60):
        """删除指定名称的 Qdrant 集合

        Args:
            collection_name (str): 集合名称
            timeout (int, optional): 删除操作的超时时间，默认为 60 秒

        Raises:
            Exception: 如果删除集合失败，抛出异常
        """
        res = self.client.delete_collection(collection_name, timeout=timeout)
        if not res:
            raise Exception(f"Delete collection {collection_name} failed.")  # 如果删除失败，抛出异常

    def add(self, collection_name: str, points: List[PointStruct]):
        """向 Qdrant 集合中添加向量数据

        Args:
            collection_name (str): 集合名称
            points (List[PointStruct]): 向量数据列表，PointStruct 对象的详细信息请参见 https://github.com/qdrant/qdrant-client

        Returns:
            None
        """
        # 向集合中插入数据
        self.client.upsert(
            collection_name,
            points,
        )

    def search(
        self,
        collection_name: str,
        query: List[float],
        query_filter: Filter = None,
        k=10,
        return_vector=False,
    ):
        """进行向量搜索

        Args:
            collection_name (str): Qdrant 集合名称
            query (List[float]): 输入的查询向量
            query_filter (Filter, optional): 查询过滤条件，Filter 对象的详细信息请参见 https://github.com/qdrant/qdrant-client
            k (int, optional): 返回最相似的 k 条数据，默认为 10
            return_vector (bool, optional): 是否返回向量，默认为 False

        Returns:
            List[dict]: 搜索结果的列表
        """
        # 执行搜索操作
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query,
            query_filter=query_filter,
            limit=k,
            with_vectors=return_vector,
        )
        # 返回搜索结果的字典列表
        return [hit.__dict__ for hit in hits]

    def write(self, *args, **kwargs):
        """写入数据

        该方法目前未实现。
        """
        pass  # 目前无实现
