"""Set of default prompts."""

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = """
你是一个高效的助手，任务是根据给定的问题评估一系列文档。

我将提供一个问题和一系列文档。你的任务是根据问题的相关性，按顺序回应你需要查阅的文档编号，并给出相关性分数。

## 问题
{query_str}

## 文档
{context_str}

## 格式示例
Doc: 9, Relevance: 7

## 指令
- 理解问题。
- 评估问题与文档之间的相关性。
- 相关性得分为1-10，表示文档与问题的相关程度。
- 不要包括任何与问题不相关的文档。
- 如果没有任何文档能直接回答问题，回答“没有相关文档”。

## 约束
格式：仅按照示例格式打印结果，像**格式示例**一样。

## 操作
遵循指令，生成输出，并确保符合 **约束**。
"""

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT)
