"""
This module implements a memory system combining short-term and long-term storage for AI role memory management.
It utilizes a RAG (Retrieval-Augmented Generation) engine for long-term memory storage and retrieval.
"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

from metagpt.actions import UserRequirement
from metagpt.const import TEAMLEADER_NAME
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.schema import LongTermMemoryItem, Message
from metagpt.utils.common import any_to_str
from metagpt.utils.exceptions import handle_exception

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore
    from metagpt.rag.engines import SimpleEngine


class RoleZeroLongTermMemory(Memory):
    """
    实现了一个结合短期和长期存储的记忆系统，使用 RAG 引擎。
    当短期存储容量达到上限时，会将旧记忆转移到长期存储。
    需要时可以检索短期和长期记忆的结合。
    """

    persist_path: str = Field(default=".role_memory_data", description="保存数据的目录。")
    collection_name: str = Field(default="role_zero", description="集合名称，例如角色名称。")
    memory_k: int = Field(default=200, description="短期记忆的容量。")
    similarity_top_k: int = Field(default=5, description="检索的长期记忆数量。")
    use_llm_ranker: bool = Field(default=False, description="是否使用 LLM 排序器以获得更好的结果。")

    _rag_engine: Any = None  # RAG 引擎实例

    @property
    def rag_engine(self) -> "SimpleEngine":
        """
        获取 RAG 引擎。如果未初始化，则调用 _resolve_rag_engine() 方法进行初始化。
        """
        if self._rag_engine is None:
            self._rag_engine = self._resolve_rag_engine()

        return self._rag_engine

    def _resolve_rag_engine(self) -> "SimpleEngine":
        """
        延迟加载 RAG 引擎组件，确保只有在需要时才加载。

        它使用 `Chroma` 进行检索，使用 `LLMRanker` 进行排名。
        """

        try:
            from metagpt.rag.engines import SimpleEngine
            from metagpt.rag.schema import ChromaRetrieverConfig, LLMRankerConfig
        except ImportError:
            raise ImportError("要使用 RoleZeroMemory，需要安装 rag 模块。")

        # 设置检索器配置
        retriever_configs = [
            ChromaRetrieverConfig(
                persist_path=self.persist_path,
                collection_name=self.collection_name,
                similarity_top_k=self.similarity_top_k,
            )
        ]
        # 根据是否使用 LLM 排序器来配置排名器
        ranker_configs = [LLMRankerConfig()] if self.use_llm_ranker else []

        # 初始化 RAG 引擎
        rag_engine = SimpleEngine.from_objs(retriever_configs=retriever_configs, ranker_configs=ranker_configs)

        return rag_engine

    def add(self, message: Message):
        """
        添加新消息，并在必要时将其转移到长期记忆中。
        """

        super().add(message)

        # 判断是否需要使用长期记忆，如果需要则转移到长期记忆
        if not self._should_use_longterm_memory_for_add():
            return

        self._transfer_to_longterm_memory()

    def get(self, k=0) -> list[Message]:
        """
        返回最近的记忆，并在必要时将其与相关的长期记忆合并。
        """

        memories = super().get(k)

        # 判断是否需要合并长期记忆
        if not self._should_use_longterm_memory_for_get(k=k):
            return memories

        query = self._build_longterm_memory_query()
        related_memories = self._fetch_longterm_memories(query)
        logger.info(f"已获取 {len(related_memories)} 条长期记忆。")

        # 将长期记忆和短期记忆合并
        final_memories = related_memories + memories

        return final_memories

    def _should_use_longterm_memory_for_add(self) -> bool:
        """
        判断是否应该使用长期记忆。
        当短期记忆数量超过 memory_k 时使用长期记忆。
        """
        return self.count() > self.memory_k

    def _should_use_longterm_memory_for_get(self, k: int) -> bool:
        """
        判断是否应该在获取时使用长期记忆。

        使用长期记忆的条件：
        - k 不等于 0
        - 最后一条消息来自用户要求
        - 最近的记忆数量大于 memory_k
        """
        conds = [
            k != 0,
            self._is_last_message_from_user_requirement(),
            self.count() > self.memory_k,
        ]

        return all(conds)

    def _transfer_to_longterm_memory(self):
        """
        将短期记忆中的一条消息转移到长期记忆中。
        """
        item = self._get_longterm_memory_item()
        self._add_to_longterm_memory(item)

    def _get_longterm_memory_item(self) -> Optional[LongTermMemoryItem]:
        """
        获取最近的一条短期记忆，通常是倒数第 k+1 条消息。
        """

        index = -(self.memory_k + 1)
        message = self.get_by_position(index)

        return LongTermMemoryItem(message=message) if message else None

    @handle_exception
    def _add_to_longterm_memory(self, item: LongTermMemoryItem):
        """
        将长期记忆项添加到 RAG 引擎中。

        如果添加长期记忆失败，记录错误但不中断程序执行。
        """

        if not item or not item.message.content:
            return

        self.rag_engine.add_objs([item])

    @handle_exception(default_return=[])
    def _fetch_longterm_memories(self, query: str) -> list[Message]:
        """
        根据查询获取长期记忆。

        如果获取长期记忆失败，将返回默认值（空列表），而不中断程序执行。

        Args:
            query (str): 用于搜索相关记忆的查询字符串。

        Returns:
            list[Message]: 与查询相关的用户和 AI 消息列表。
        """
        if not query:
            return []

        nodes = self.rag_engine.retrieve(query)
        items = self._get_items_from_nodes(nodes)
        memories = [item.message for item in items]

        return memories

    def _get_items_from_nodes(self, nodes: list["NodeWithScore"]) -> list[LongTermMemoryItem]:
        """
        从节点中获取记忆项，并根据其 `created_at` 时间戳进行排序。
        """

        items: list[LongTermMemoryItem] = [node.metadata["obj"] for node in nodes]
        items.sort(key=lambda item: item.created_at)

        return items

    def _build_longterm_memory_query(self) -> str:
        """
        构建用于查询相关长期记忆的内容。

        默认查询是获取最近的用户消息，如果没有找到则返回空字符串。
        """

        message = self._get_the_last_message()

        return message.content if message else ""

    def _get_the_last_message(self) -> Optional[Message]:
        """
        获取最后一条消息。
        如果没有消息，返回 None。
        """

        if not self.count():
            return None

        return self.get_by_position(-1)

    def _is_last_message_from_user_requirement(self) -> bool:
        """
        检查最后一条消息是否来自用户要求或由团队负责人发送。
        """

        message = self._get_the_last_message()

        if not message:
            return False

        is_user_message = message.is_user_message()
        cause_by_user_requirement = message.cause_by == any_to_str(UserRequirement)
        sent_from_team_leader = message.sent_from == TEAMLEADER_NAME

        return is_user_message and (cause_by_user_requirement or sent_from_team_leader)