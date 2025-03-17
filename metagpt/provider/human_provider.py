"""
Filename: MetaGPT/metagpt/provider/human_provider.py
Created Date: Wednesday, November 8th 2023, 11:55:46 pm
Author: garylin2099
"""
from typing import Optional

from metagpt.configs.llm_config import LLMConfig
from metagpt.const import LLM_API_TIMEOUT, USE_CONFIG_TIMEOUT
from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM


class HumanProvider(BaseLLM):
    """人类提供者作为一个 '模型'，实际上接受人类输入作为其响应。
    这使得可以在框架中的任何地方用人类替换 LLM，从而引入人类交互。
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def ask(self, msg: str, timeout=USE_CONFIG_TIMEOUT) -> str:
        """提示用户输入并返回其响应。如果用户输入 'exit' 或 'quit'，程序将退出。

        参数:
            msg (str): 提示信息。
            timeout (int): 请求超时时间，默认为配置中的超时时间。

        返回:
            str: 用户的输入响应。
        """
        logger.info("轮到你了，请输入你的回答。你也可以参考下面的上下文")
        rsp = input(msg)
        if rsp in ["exit", "quit"]:
            exit()
        return rsp

    async def aask(
            self,
            msg: str,
            system_msgs: Optional[list[str]] = None,
            format_msgs: Optional[list[dict[str, str]]] = None,
            generator: bool = False,
            timeout=USE_CONFIG_TIMEOUT,
            **kwargs
    ) -> str:
        """异步版本的 `ask` 方法，实际实现是调用同步的 `ask` 方法。

        参数:
            msg (str): 提示信息。
            system_msgs (list[str], 可选): 系统消息列表（默认为 None）。
            format_msgs (list[dict[str, str]], 可选): 格式化消息（默认为 None）。
            generator (bool, 可选): 是否启用生成器（默认为 False）。
            timeout (int): 请求超时时间，默认为配置中的超时时间。

        返回:
            str: 用户的输入响应。
        """
        return self.ask(msg, timeout=self.get_timeout(timeout))

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """异步补全方法，当前不做任何操作"""
        pass

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """空实现基类中抽象方法的异步版本"""
        return []

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """异步流式补全方法，当前不做任何操作"""
        pass

    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT) -> str:
        """空实现基类中抽象方法的异步版本，用于流式文本补全"""
        return ""

    def get_timeout(self, timeout: int) -> int:
        """获取超时时间，优先使用传入的 `timeout` 参数，否则使用默认的 LLM_API_TIMEOUT 配置。

        参数:
            timeout (int): 请求超时时间。

        返回:
            int: 最终的超时时间。
        """
        return timeout or LLM_API_TIMEOUT
