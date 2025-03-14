"""Simple Serializer."""

from typing import Any

from metagpt.exp_pool.serializers.base import BaseSerializer


# SimpleSerializer 类继承自 BaseSerializer，用于简单的序列化和反序列化操作
class SimpleSerializer(BaseSerializer):
    # 序列化请求的方法，将请求对象转换为字符串
    def serialize_req(self, **kwargs) -> str:
        """直接使用 `str` 将请求对象转换为字符串。

        参数:
            req (Any): 要序列化的请求对象。

        返回:
            str: 序列化后的请求字符串。
        """

        # 获取请求数据并转换为字符串
        return str(kwargs.get("req", ""))

    # 序列化响应的方法，将响应对象转换为字符串
    def serialize_resp(self, resp: Any) -> str:
        """直接使用 `str` 将响应对象转换为字符串。

        参数:
            resp (Any): 要序列化的响应对象。

        返回:
            str: 序列化后的响应字符串。
        """

        # 将响应对象转换为字符串
        return str(resp)

    # 反序列化响应的方法，直接返回字符串响应
    def deserialize_resp(self, resp: str) -> Any:
        """直接返回字符串响应。

        参数:
            resp (str): 序列化后的响应字符串。

        返回:
            Any: 直接返回原始的字符串响应。
        """

        # 直接返回字符串响应
        return resp
