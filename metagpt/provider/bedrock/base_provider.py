import json
from abc import ABC, abstractmethod
from typing import Union


class BaseBedrockProvider(ABC):
    """基础 Bedrock 提供者类，用于处理不同的生成参数。"""

    # 用于指定最大 token 数的字段名称
    max_tokens_field_name = "max_tokens"

    def __init__(self, reasoning: bool = False, reasoning_max_token: int = 4000):
        """
        初始化 BaseBedrockProvider。

        :param reasoning: 是否启用推理模式
        :param reasoning_max_token: 推理模式下的最大 token 数
        """
        self.reasoning = reasoning
        self.reasoning_max_token = reasoning_max_token

    @abstractmethod
    def _get_completion_from_dict(self, rsp_dict: dict) -> str:
        """
        抽象方法，子类需要实现该方法，以从响应字典中提取补全文本。

        :param rsp_dict: 响应数据字典
        :return: 补全的文本
        """
        ...

    def get_request_body(self, messages: list[dict], const_kwargs, *args, **kwargs) -> str:
        """
        生成请求体，将消息列表转换为 JSON 格式的请求数据。

        :param messages: 消息列表，格式为 [{"role": "user", "content": "消息内容"}]
        :param const_kwargs: 额外的请求参数
        :return: JSON 格式的请求体字符串
        """
        body = json.dumps({"prompt": self.messages_to_prompt(messages), **const_kwargs})
        return body

    def get_choice_text(self, response_body: dict) -> Union[str, dict[str, str]]:
        """
        从响应数据中提取补全文本。

        :param response_body: 响应数据字典
        :return: 提取的补全文本
        """
        completions = self._get_completion_from_dict(response_body)
        return completions

    def get_choice_text_from_stream(self, event) -> Union[bool, str]:
        """
        从流式响应事件中提取补全文本。

        :param event: 流式响应事件，包含 JSON 数据
        :return: (布尔值, 补全文本) ，布尔值用于指示是否有更多数据
        """
        rsp_dict = json.loads(event["chunk"]["bytes"])
        completions = self._get_completion_from_dict(rsp_dict)
        return False, completions

    def messages_to_prompt(self, messages: list[dict]) -> str:
        """
        将消息列表转换为字符串格式的提示文本。

        :param messages: 消息列表，格式为 [{"role": "user", "content": "消息内容"}]
        :return: 转换后的提示文本，例如：
                 "user: 你好\nassistant: 你好！有什么可以帮助你的吗？"
        """
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
