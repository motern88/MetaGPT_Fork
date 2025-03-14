from __future__ import annotations

import json
import traceback
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator
from rank_bm25 import BM25Okapi

from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.prompts.di.role_zero import JSON_REPAIR_PROMPT
from metagpt.schema import Plan
from metagpt.tools import TOOL_REGISTRY
from metagpt.tools.tool_data_type import Tool
from metagpt.tools.tool_registry import validate_tool_names
from metagpt.utils.common import CodeParser
from metagpt.utils.repair_llm_raw_output import RepairType, repair_llm_raw_output

TOOL_INFO_PROMPT = """
## 功能
- 你可以在任何代码行中使用“可用工具”中预定义的工具，形式为 Python 类或函数。
- 你可以自由组合使用其他公共包，如 sklearn、numpy、pandas 等。

## 可用工具：
每个工具的描述为 JSON 格式。当你调用某个工具时，首先从其路径导入该工具。
{tool_schemas}
"""


TOOL_RECOMMENDATION_PROMPT = """
## 用户需求：
{current_task}

## 任务
推荐最多 {topk} 个能帮助解决“用户需求”的工具。

## 可用工具：
{available_tools}

## 工具选择和说明：
- 选择与完成“用户需求”最相关的工具。
- 如果认为没有合适的工具，返回一个空列表。
- 只列出工具名称，不需要列出每个工具的完整模式。
- 确保选中的工具存在于“可用工具”中。
- 输出一个 JSON 格式的工具名称列表：
```json
["tool_name1", "tool_name2", ...]
```
"""


class ToolRecommender(BaseModel):
    """
     默认的 ToolRecommender 类:
     1. Recall: 由子类实现。根据给定的上下文和计划召回工具。
     2. Rank: 使用 LLM 从召回的工具集中选择最终候选工具。
    """

    tools: dict[str, Tool] = {}  # 工具字典，存储所有可用的工具
    force: bool = False  # 是否强制推荐指定的工具

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tools(cls, v: list[str]) -> dict[str, Tool]:
        """
        校验工具列表，并根据需要转换成工具字典
        """
        if isinstance(v, dict):
            return v

        if v == ["<all>"]:
            return TOOL_REGISTRY.get_all_tools()
        else:
            return validate_tool_names(v)

    async def recommend_tools(
            self, context: str = "", plan: Plan = None, recall_topk: int = 20, topk: int = 5
    ) -> list[Tool]:
        """
        根据给定的上下文和计划推荐一组工具。推荐过程包括两个阶段：从大池中召回工具并对召回的工具进行排序，选择最终的工具集合。

        参数:
            context (str): 工具推荐的上下文。
            plan (Plan): 工具推荐的计划。
            recall_topk (int): 在召回阶段从工具池中选择的工具数量。
            topk (int): 排序后返回的最终推荐工具数量。

        返回:
            list[Tool]: 推荐的工具列表。
        """

        if not self.tools:
            return []

        if self.force or (not context and not plan):
            # 如果是强制推荐或者没有有效的上下文和计划，直接返回用户指定的工具
            return list(self.tools.values())

        recalled_tools = await self.recall_tools(context=context, plan=plan, topk=recall_topk)
        if not recalled_tools:
            return []

        ranked_tools = await self.rank_tools(recalled_tools=recalled_tools, context=context, plan=plan, topk=topk)

        logger.info(f"推荐的工具： \n{[tool.name for tool in ranked_tools]}")

        return ranked_tools

    async def get_recommended_tool_info(self, fixed: list[str] = None, **kwargs) -> str:
        """
        将推荐的工具信息以字符串的形式包装起来，适合在提示语中直接使用。
        """
        recommended_tools = await self.recommend_tools(**kwargs)
        if fixed:
            recommended_tools.extend([self.tools[tool_name] for tool_name in fixed if tool_name in self.tools])
        if not recommended_tools:
            return ""
        tool_schemas = {tool.name: tool.schemas for tool in recommended_tools}
        return TOOL_INFO_PROMPT.format(tool_schemas=tool_schemas)

    async def recall_tools(self, context: str = "", plan: Plan = None, topk: int = 20) -> list[Tool]:
        """
        从大池中根据上下文和计划召回相关工具。

        需要在子类中实现。
        """
        raise NotImplementedError

    async def rank_tools(
            self, recalled_tools: list[Tool], context: str = "", plan: Plan = None, topk: int = 5
    ) -> list[Tool]:
        """
        默认的工具排序方法。使用 LLM 根据上下文和计划对召回的工具进行排序，并返回最终的推荐工具。

        参数:
            recalled_tools (list[Tool]): 召回的工具列表。
            context (str): 工具推荐的上下文。
            plan (Plan): 工具推荐的计划。
            topk (int): 排序后返回的最终推荐工具数量。

        返回:
            list[Tool]: 排序后的推荐工具列表。
        """
        current_task = plan.current_task.instruction if plan else context

        available_tools = {tool.name: tool.schemas["description"] for tool in recalled_tools}
        prompt = TOOL_RECOMMENDATION_PROMPT.format(
            current_task=current_task,
            available_tools=available_tools,
            topk=topk,
        )
        rsp = await LLM().aask(prompt, stream=False)

        # 临时方案，待 role zero 的版本完成可将本注释内的代码直接替换掉
        # -------------开始---------------
        try:
            ranked_tools = CodeParser.parse_code(block=None, lang="json", text=rsp)
            ranked_tools = json.loads(
                repair_llm_raw_output(output=ranked_tools, req_keys=[None], repair_type=RepairType.JSON)
            )
        except json.JSONDecodeError:
            ranked_tools = await LLM().aask(msg=JSON_REPAIR_PROMPT.format(json_data=rsp))
            ranked_tools = json.loads(CodeParser.parse_code(block=None, lang="json", text=ranked_tools))
        except Exception:
            tb = traceback.format_exc()
            print(tb)

        # 为了对 LLM 不按格式生成进行容错
        if isinstance(ranked_tools, dict):
            ranked_tools = list(ranked_tools.values())[0]
        # -------------结束---------------

        if not isinstance(ranked_tools, list):
            logger.warning(f"无效的排序结果：{ranked_tools}，将使用召回的工具代替。")
            ranked_tools = list(available_tools.keys())

        valid_tools = validate_tool_names(ranked_tools)

        return list(valid_tools.values())[:topk]


class TypeMatchToolRecommender(ToolRecommender):
    """
    传统的工具推荐器，使用任务类型匹配在召回阶段：
    1. 召回：根据任务类型与工具标签的精确匹配来找到工具；
    2. 排序：使用 LLM 对召回的工具进行排序，排序方式与默认的 ToolRecommender 相同。
    """

    async def recall_tools(self, context: str = "", plan: Plan = None, topk: int = 20) -> list[Tool]:
        if not plan:
            # 如果没有计划，直接返回用户指定的前 `topk` 个工具
            return list(self.tools.values())[:topk]

        # 根据任务类型与工具标签的精确匹配来查找工具
        task_type = plan.current_task.task_type
        # 从工具注册表中获取符合任务类型标签的工具
        candidate_tools = TOOL_REGISTRY.get_tools_by_tag(task_type)
        candidate_tool_names = set(self.tools.keys()) & candidate_tools.keys()
        # 选择与任务类型匹配的工具
        recalled_tools = [candidate_tools[tool_name] for tool_name in candidate_tool_names][:topk]

        logger.info(f"召回的工具: \n{[tool.name for tool in recalled_tools]}")

        return recalled_tools


class BM25ToolRecommender(ToolRecommender):
    """
    使用 BM25 算法在召回阶段推荐工具：
    1. 召回：通过任务指令查询工具描述，如果计划存在；如果没有计划，则返回用户指定的所有工具；
    2. 排序：使用 LLM 对召回的工具进行排序，排序方式与默认的 ToolRecommender 相同。
    """

    bm25: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化语料库
        self._init_corpus()

    def _init_corpus(self):
        # 通过将工具名称、标签和描述结合在一起生成语料库
        corpus = [f"{tool.name} {tool.tags}: {tool.schemas['description']}" for tool in self.tools.values()]
        # 将语料库中的每篇文档进行分词
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text):
        # 目前的分词方式较简单，返回按空格分开的单词，可能需要改进
        return text.split()

    async def recall_tools(self, context: str = "", plan: Plan = None, topk: int = 20) -> list[Tool]:
        # 如果有计划，则使用计划中的任务指令；否则，使用上下文
        query = plan.current_task.instruction if plan else context

        # 对查询进行分词
        query_tokens = self._tokenize(query)
        # 获取与查询最相关的文档的分数
        doc_scores = self.bm25.get_scores(query_tokens)
        # 获取与查询最相关的前 `topk` 个工具
        top_indexes = np.argsort(doc_scores)[::-1][:topk]
        recalled_tools = [list(self.tools.values())[index] for index in top_indexes]

        logger.info(
            f"召回的工具: \n{[tool.name for tool in recalled_tools]}; 分数: {[np.round(doc_scores[index], 4) for index in top_indexes]}"
        )

        return recalled_tools


class EmbeddingToolRecommender(ToolRecommender):
    """
    说明：待实现。
    使用嵌入（Embedding）在召回阶段推荐工具：
    1. 召回：使用嵌入计算查询与工具信息的相似度；
    2. 排序：使用 LLM 对召回的工具进行排序，排序方式与默认的 ToolRecommender 相同。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def recall_tools(self, context: str = "", plan: Plan = None, topk: int = 20) -> list[Tool]:
        # 尚未实现
        pass
