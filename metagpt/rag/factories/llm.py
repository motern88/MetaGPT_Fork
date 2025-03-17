"""RAG LLM."""
import asyncio
from typing import Any

from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

from metagpt.config2 import config
from metagpt.provider.base_llm import BaseLLM
from metagpt.utils.async_helper import NestAsyncio
from metagpt.utils.token_counter import TOKEN_MAX


class RAGLLM(CustomLLM):
    """LlamaIndex的LLM与MetaGPT的LLM有所不同。

    继承自LlamaIndex的CustomLLM，使得MetaGPT的LLM可以被LlamaIndex使用。

    如果遇到“Calculated available context size -xxx was not non-negative”的错误，可以在config.yaml中设置LLM的context_length或max_token。
    """

    model_infer: BaseLLM = Field(..., description="MetaGPT的LLM模型。")
    context_window: int = -1  # 上下文窗口大小
    num_output: int = -1  # 输出的最大token数
    model_name: str = ""  # 模型名称

    def __init__(
            self,
            model_infer: BaseLLM,
            context_window: int = -1,
            num_output: int = -1,
            model_name: str = "",
            *args,
            **kwargs
    ):
        """初始化RAGLLM实例。

        参数:
        - model_infer: MetaGPT的LLM模型实例
        - context_window: 上下文窗口大小
        - num_output: 输出的最大token数
        - model_name: 模型名称
        """
        super().__init__(*args, **kwargs)

        # 如果没有提供context_window，则从配置中获取默认值
        if context_window < 0:
            context_window = TOKEN_MAX.get(config.llm.model, DEFAULT_CONTEXT_WINDOW)

        # 如果没有提供num_output，则从配置中获取默认值
        if num_output < 0:
            num_output = config.llm.max_token

        # 如果没有提供model_name，则从配置中获取默认值
        if not model_name:
            model_name = config.llm.model

        self.model_infer = model_infer
        self.context_window = context_window
        self.num_output = num_output
        self.model_name = model_name

    @property
    def metadata(self) -> LLMMetadata:
        """获取LLM的元数据，包括上下文窗口大小、输出token数和模型名称。"""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name or "unknown"
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """同步的文本生成完成方法，返回生成的文本结果。"""
        NestAsyncio.apply_once()  # 确保只应用一次异步IO
        return asyncio.get_event_loop().run_until_complete(self.acomplete(prompt, **kwargs))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """异步的文本生成完成方法，返回生成的文本结果。"""
        text = await self.model_infer.aask(msg=prompt, stream=False)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """流式文本生成方法（待实现）。"""
        ...


def get_rag_llm(model_infer: BaseLLM = None) -> RAGLLM:
    """获取可以被LlamaIndex使用的LLM实例。

    参数:
    - model_infer: 可选的MetaGPT的LLM模型实例，如果没有提供则使用默认的LLM实例。
    """
    from metagpt.llm import LLM

    return RAGLLM(model_infer=model_infer or LLM())
