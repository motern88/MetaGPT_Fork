"""RoleZero Serializer."""

import copy
import json

from metagpt.exp_pool.serializers.simple import SimpleSerializer


# RoleZeroSerializer 类继承自 SimpleSerializer，用于序列化请求数据
class RoleZeroSerializer(SimpleSerializer):
    # 序列化请求的方法，将请求数据转换为 JSON 字符串
    def serialize_req(self, **kwargs) -> str:
        """序列化请求以便存储到数据库，确保返回为字符串格式。

        由于 `req` 可能非常长，容易导致嵌入错误，因此只提取必要的内容。

        参数:
            req (list[dict]): 要序列化的请求。例如:
                [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "user", "content": "context"},
                ]

        返回:
            str: 序列化后的请求 JSON 字符串。
        """
        req = kwargs.get("req", [])

        if not req:
            return ""

        # 过滤请求数据，保留必要内容
        filtered_req = self._filter_req(req)

        # 如果有额外的 state_data，加入到过滤后的请求中
        if state_data := kwargs.get("state_data"):
            filtered_req.append({"role": "user", "content": state_data})

        # 将过滤后的请求转化为 JSON 字符串
        return json.dumps(filtered_req)

    # 过滤请求内容，只保留有用的数据
    def _filter_req(self, req: list[dict]) -> list[dict]:
        """过滤 `req`，只保留必要的项。

        参数:
            req (list[dict]): 原始请求。

        返回:
            list[dict]: 过滤后的请求。
        """

        # 深拷贝每一项，只保留有用内容
        filtered_req = [copy.deepcopy(item) for item in req if self._is_useful_content(item["content"])]

        return filtered_req

    # 判断请求中的内容是否有用
    def _is_useful_content(self, content: str) -> bool:
        """目前只考虑文件的内容，后续可以增加更多判断规则。

        参数:
            content (str): 请求中的内容。

        返回:
            bool: 如果内容有用则返回 True，否则返回 False。
        """

        # 判断内容中是否包含特定的字符串，表示是有用的内容
        if "Command Editor.read executed: file_path" in content:
            return True

        return False
