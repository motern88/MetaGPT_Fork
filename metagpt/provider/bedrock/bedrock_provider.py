import json
from typing import Literal, Tuple, Union

from metagpt.provider.bedrock.base_provider import BaseBedrockProvider
from metagpt.provider.bedrock.utils import (
    messages_to_prompt_llama2,
    messages_to_prompt_llama3,
)


# 定义 MistralProvider 类，继承自 BaseBedrockProvider
class MistralProvider(BaseBedrockProvider):
    # 参考 AWS Bedrock 官方文档：
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html

    def messages_to_prompt(self, messages: list[dict]):
        # 使用 LLaMA2 兼容的消息转换方法
        return messages_to_prompt_llama2(messages)

    def _get_completion_from_dict(self, rsp_dict: dict) -> str:
        # 提取 API 响应中的文本内容
        return rsp_dict["outputs"][0]["text"]


# 定义 AnthropicProvider 类，继承自 BaseBedrockProvider
class AnthropicProvider(BaseBedrockProvider):
    # 参考 AWS Bedrock 官方文档：
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-37.html
    # https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html#anthropic_claude

    def _split_system_user_messages(self, messages: list[dict]) -> Tuple[str, list[dict]]:
        # 将系统消息和用户消息分开
        system_messages = []
        user_messages = []
        for message in messages:
            if message["role"] == "system":
                system_messages.append(message)
            else:
                user_messages.append(message)
        return self.messages_to_prompt(system_messages), user_messages

    def get_request_body(self, messages: list[dict], generate_kwargs, *args, **kwargs) -> str:
        # 如果开启了 reasoning（推理），设置 temperature 和思考参数
        if self.reasoning:
            generate_kwargs["temperature"] = 1  # 需要固定为 1
            generate_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.reasoning_max_token}

        system_message, user_messages = self._split_system_user_messages(messages)
        # 组织 API 请求的 JSON 结构
        body = json.dumps(
            {
                "messages": user_messages,
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_message,
                **generate_kwargs,
            }
        )
        return body

    def _get_completion_from_dict(self, rsp_dict: dict) -> dict[str, Tuple[str, str]]:
        # 解析 API 响应中的文本内容
        if self.reasoning:
            return {"reasoning_content": rsp_dict["content"][0]["thinking"], "content": rsp_dict["content"][1]["text"]}
        return rsp_dict["content"][0]["text"]

    def get_choice_text_from_stream(self, event) -> Union[bool, str]:
        # 解析流式输出
        rsp_dict = json.loads(event["chunk"]["bytes"])
        if rsp_dict["type"] == "content_block_delta":
            reasoning = False
            delta_type = rsp_dict["delta"]["type"]
            if delta_type == "text_delta":
                completions = rsp_dict["delta"]["text"]
            elif delta_type == "thinking_delta":
                completions = rsp_dict["delta"]["thinking"]
                reasoning = True
            elif delta_type == "signature_delta":
                completions = ""
            return reasoning, completions
        else:
            return False, ""


# 定义 CohereProvider 类，适用于 AWS Bedrock 的 Cohere 模型
class CohereProvider(BaseBedrockProvider):
    # 参考 AWS Bedrock 官方文档：
    # Command: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
    # Command R/R+: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _get_completion_from_dict(self, rsp_dict: dict) -> str:
        return rsp_dict["generations"][0]["text"]

    def messages_to_prompt(self, messages: list[dict]) -> str:
        if "command-r" in self.model_name:
            # 适配 Command-R 模型
            role_map = {"user": "USER", "assistant": "CHATBOT", "system": "USER"}
            messages = list(
                map(lambda message: {"role": role_map[message["role"]], "message": message["content"]}, messages)
            )
            return messages
        else:
            # 适配普通 Cohere 模型
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    def get_request_body(self, messages: list[dict], generate_kwargs, *args, **kwargs):
        prompt = self.messages_to_prompt(messages)
        if "command-r" in self.model_name:
            # 适用于 Command-R/R+ 模型
            chat_history, message = prompt[:-1], prompt[-1]["message"]
            body = json.dumps({"message": message, "chat_history": chat_history, **generate_kwargs})
        else:
            body = json.dumps({"prompt": prompt, "stream": kwargs.get("stream", False), **generate_kwargs})
        return body

    def get_choice_text_from_stream(self, event) -> Union[bool, str]:
        rsp_dict = json.loads(event["chunk"]["bytes"])
        completions = rsp_dict.get("text", "")
        return False, completions


# 定义 MetaProvider 类，适用于 LLaMA2 和 LLaMA3
class MetaProvider(BaseBedrockProvider):
    # 参考 AWS Bedrock 官方文档：
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

    max_tokens_field_name = "max_gen_len"

    def __init__(self, llama_version: Literal["llama2", "llama3"]) -> None:
        self.llama_version = llama_version

    def messages_to_prompt(self, messages: list[dict]):
        if self.llama_version == "llama2":
            return messages_to_prompt_llama2(messages)
        else:
            return messages_to_prompt_llama3(messages)

    def _get_completion_from_dict(self, rsp_dict: dict) -> str:
        return rsp_dict["generation"]


# 定义 AI21Provider 类，适用于 AI21 Jurassic-2 和 Jamba 模型
class Ai21Provider(BaseBedrockProvider):
    # 参考 AWS Bedrock 官方文档：
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jurassic2.html

    def __init__(self, model_type: Literal["j2", "jamba"]) -> None:
        self.model_type = model_type
        self.max_tokens_field_name = "maxTokens" if model_type == "j2" else "max_tokens"

    def _get_completion_from_dict(self, rsp_dict: dict) -> str:
        return rsp_dict["completions"][0]["data"]["text"] if self.model_type == "j2" else rsp_dict["choices"][0]["message"]["content"]


# 定义 AmazonProvider 类，适用于 AWS Titan 模型
class AmazonProvider(BaseBedrockProvider):
    # 参考 AWS Bedrock 官方文档：
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html

    max_tokens_field_name = "maxTokenCount"

    def get_request_body(self, messages: list[dict], generate_kwargs, *args, **kwargs):
        body = json.dumps({"inputText": self.messages_to_prompt(messages), "textGenerationConfig": generate_kwargs})
        return body

    def _get_completion_from_dict(self, rsp_dict: dict) -> str:
        return rsp_dict["results"][0]["outputText"]


# 模型提供商映射表
PROVIDERS = {
    "mistral": MistralProvider,
    "meta": MetaProvider,
    "ai21": Ai21Provider,
    "cohere": CohereProvider,
    "anthropic": AnthropicProvider,
    "amazon": AmazonProvider,
}


def get_provider(model_id: str, reasoning: bool = False, reasoning_max_token: int = 4000):
    """
    根据 `model_id` 获取相应的模型提供者类实例。

    参数：
        model_id (str): 模型标识符，例如 "meta.llama2" 或 "mistral.some_model"。
        reasoning (bool, 可选): 是否启用推理模式，默认为 False。
        reasoning_max_token (int, 可选): 推理模式下的最大 token 预算，默认为 4000。

    处理逻辑：
    1. 解析 `model_id`，通过 `.` 分割获取提供者 (`provider`) 和模型名称 (`model_name`)。
    2. 若 `model_id` 由两个部分组成（如 `"meta.llama2"`），直接拆分。
    3. 若 `model_id` 由三个部分组成（如 `"us.meta.llama2"`，可能包含国家/区域信息），忽略第一个部分。
    4. 检查 `provider` 是否存在于 `PROVIDERS` 字典中，若不存在则抛出 `KeyError`。
    5. 处理特定提供者的模型类型：
       - `"meta"`: 区分 `llama2` 和 `llama3`，只取 `model_name` 前 6 个字符。
       - `"ai21"`: 区分 `j2` 和 `jamba`，取 `model_name` 以 `"-"` 分割后的第一个部分。
       - `"cohere"`: 处理 `command-r`、`command-r+` 等特殊模型。
    6. 其他提供者默认使用 `reasoning` 和 `reasoning_max_token` 进行实例化。

    返回：
        BaseBedrockProvider 的子类实例，对应 `model_id` 指定的模型提供者。

    异常：
        KeyError: 若 `provider` 不在 `PROVIDERS` 中，则抛出异常。

    """
    arr = model_id.split(".")
    if len(arr) == 2:
        provider, model_name = arr  # 解析提供者名称和模型名称，例如 "meta.llama2"
    elif len(arr) == 3:
        # 处理类似 "us.meta.llama2" 形式的 model_id，忽略国家代码部分
        _, provider, model_name = arr

    if provider not in PROVIDERS:
        raise KeyError(f"{provider} is not supported!")  # 若提供者不支持，则抛出异常

    if provider == "meta":
        # 区分 Llama2 和 Llama3，取 `model_name` 前 6 个字符
        return PROVIDERS[provider](model_name[:6])
    elif provider == "ai21":
        # 区分 j2 和 jamba，取 `model_name` 以 `-` 分割后的第一个部分
        return PROVIDERS[provider](model_name.split("-")[0])
    elif provider == "cohere":
        # 处理 `command-r`、`command-r+` 等 Cohere 模型
        return PROVIDERS[provider](model_name)

    # 其他提供者默认传入 `reasoning` 和 `reasoning_max_token`
    return PROVIDERS[provider](reasoning=reasoning, reasoning_max_token=reasoning_max_token)