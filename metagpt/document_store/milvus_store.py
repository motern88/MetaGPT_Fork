from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from metagpt.document_store.base_store import BaseStore


@dataclass
class MilvusConnection:
    """
    Milvus 连接配置类

    Args:
        uri: Milvus 服务器的 URL
        token: Milvus 认证令牌
    """
    uri: str = None  # Milvus 连接的 URL
    token: str = None  # Milvus 认证令牌


class MilvusStore(BaseStore):
    """
    Milvus 存储类，用于与 Milvus 数据库进行交互，支持创建集合、插入数据、查询等操作
    """

    def __init__(self, connect: MilvusConnection):
        """初始化 MilvusStore 对象

        Args:
            connect (MilvusConnection): 包含 Milvus 连接信息的对象
        """
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise Exception("Please install pymilvus first.")  # 如果未安装 pymilvus，抛出异常
        if not connect.uri:
            raise Exception("please check MilvusConnection, uri must be set.")  # 检查 uri 是否设置
        self.client = MilvusClient(uri=connect.uri, token=connect.token)  # 创建 Milvus 客户端实例

    def create_collection(self, collection_name: str, dim: int, enable_dynamic_schema: bool = True):
        """创建 Milvus 集合

        Args:
            collection_name (str): 集合名称
            dim (int): 向量维度
            enable_dynamic_schema (bool, optional): 是否启用动态模式，默认为 True
        """
        from pymilvus import DataType

        # 如果集合已存在，则删除该集合
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)

        # 创建集合模式（schema），定义集合中字段的类型
        schema = self.client.create_schema(
            auto_id=False,  # 禁用自动 ID
            enable_dynamic_field=False,  # 禁用动态字段
        )
        # 添加 id 字段，并设置为主键
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=36)
        # 添加 vector 字段，用于存储向量数据
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

        # 创建索引参数
        index_params = self.client.prepare_index_params()
        # 为 vector 字段添加索引
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

        # 创建集合并应用索引参数
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_schema=enable_dynamic_schema,  # 是否启用动态模式
        )

    @staticmethod
    def build_filter(key, value) -> str:
        """构建过滤条件

        根据不同的数据类型构建适当的过滤条件。

        Args:
            key (str): 字段名称
            value (str, list, or int): 字段值，可以是字符串、列表或整数

        Returns:
            str: 构建的过滤条件表达式
        """
        if isinstance(value, str):
            filter_expression = f'{key} == "{value}"'  # 如果值是字符串，构建相等条件
        else:
            if isinstance(value, list):
                filter_expression = f"{key} in {value}"  # 如果值是列表，构建 in 条件
            else:
                filter_expression = f"{key} == {value}"  # 否则，构建相等条件

        return filter_expression

    def search(
        self,
        collection_name: str,
        query: List[float],
        filter: Dict = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
    ) -> List[dict]:
        """在 Milvus 集合中进行相似度搜索

        Args:
            collection_name (str): 集合名称
            query (List[float]): 查询向量
            filter (Dict, optional): 过滤条件，默认为 None
            limit (int, optional): 返回的最大结果数量，默认为 10
            output_fields (Optional[List[str]], optional): 输出的字段列表，默认为 None

        Returns:
            List[dict]: 搜索结果
        """
        # 构建过滤表达式
        filter_expression = " and ".join([self.build_filter(key, value) for key, value in filter.items()])
        print(filter_expression)

        # 执行搜索
        res = self.client.search(
            collection_name=collection_name,
            data=[query],  # 查询向量
            filter=filter_expression,  # 过滤条件
            limit=limit,  # 最大返回结果数量
            output_fields=output_fields,  # 输出字段
        )[0]

        return res

    def add(self, collection_name: str, _ids: List[str], vector: List[List[float]], metadata: List[Dict[str, Any]]):
        """向 Milvus 集合中添加数据

        Args:
            collection_name (str): 集合名称
            _ids (List[str]): 文档的唯一标识符列表
            vector (List[List[float]]): 向量数据列表
            metadata (List[Dict[str, Any]]): 元数据列表
        """
        data = dict()

        for i, id in enumerate(_ids):
            data["id"] = id
            data["vector"] = vector[i]
            data["metadata"] = metadata[i]

        # 向集合中插入数据
        self.client.upsert(collection_name=collection_name, data=data)

    def delete(self, collection_name: str, _ids: List[str]):
        """删除 Milvus 集合中的数据

        Args:
            collection_name (str): 集合名称
            _ids (List[str]): 要删除的文档的唯一标识符列表
        """
        self.client.delete(collection_name=collection_name, ids=_ids)

    def write(self, *args, **kwargs):
        """写入数据

        该方法目前未实现。
        """
        pass  # 目前无实现
