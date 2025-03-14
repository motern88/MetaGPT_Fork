"""ActionNode Serializer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

# 如果是类型检查，则导入 ActionNode 类
if TYPE_CHECKING:
    from metagpt.actions.action_node import ActionNode

# 导入 SimpleSerializer 类
from metagpt.exp_pool.serializers.simple import SimpleSerializer

# ActionNode 的序列化器类，继承自 SimpleSerializer
class ActionNodeSerializer(SimpleSerializer):
    # 序列化 ActionNode 响应的方法
    def serialize_resp(self, resp: ActionNode) -> str:
        """将 ActionNode 响应序列化为 JSON 字符串。"""
        return resp.instruct_content.model_dump_json()

    # 反序列化 ActionNode 响应的方法
    def deserialize_resp(self, resp: str) -> ActionNode:
        """自定义反序列化，当发现完美经验时会触发。

        ActionNode 无法直接序列化，它会抛出 'cannot pickle 'SSLContext' object' 错误。
        """

        # 定义一个内部类 InstructContent，用于存储和处理 JSON 数据
        class InstructContent:
            def __init__(self, json_data):
                self.json_data = json_data

            def model_dump_json(self):
                """返回存储的 JSON 数据。"""
                return self.json_data

        # 从 ActionNode 类导入
        from metagpt.actions.action_node import ActionNode

        # 创建一个 ActionNode 实例
        action_node = ActionNode(key="", expected_type=Type[str], instruction="", example="")

        # 将反序列化的 JSON 数据赋值给 ActionNode 的 instruct_content 属性
        action_node.instruct_content = InstructContent(resp)

        # 返回反序列化后的 ActionNode 实例
        return action_node
