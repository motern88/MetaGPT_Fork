"""Enhancing question-answering capabilities through search engine augmentation."""

from __future__ import annotations

import json

from pydantic import Field, PrivateAttr, model_validator

from metagpt.actions import Action
from metagpt.actions.research import CollectLinks, WebBrowseAndSummarize
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.tools.web_browser_engine import WebBrowserEngine
from metagpt.utils.common import CodeParser
from metagpt.utils.parse_html import WebPage
from metagpt.utils.report import ThoughtReporter

REWRITE_QUERY_PROMPT = """
Role: 你是一个高效的助手，负责为给定问题提供更好的搜索查询，以便搜索引擎回答该问题。

我将提供一个问题。你的任务是为搜索引擎提供更好的搜索查询。

## Context
### Question
{q}

## Format Example
```json
{{
    "query": "为搜索引擎提供更好的查询。",
}}
```

## Instructions
- 理解用户给定的问题。
- 理解用户给定的问题。
- 在重写时，如果不确定具体时间，请不要包含时间信息。

## Constraint
格式：只需以 JSON 格式输出结果，格式必须与**示例格式**相同

## Action
遵循 **指令**, 生成输出并确保它符合 **约束**.
"""

SEARCH_ENHANCED_QA_SYSTEM_PROMPT = """
你是由 MGX 构建的大型语言 AI 助手。你将收到一个用户问题，请根据给定的相关上下文，清晰、简洁、准确地回答该问题。每个相关上下文以引用编号开头，例如[[citation:x]]，请使用这些上下文。

你的回答必须是正确的、准确的，并且以专家的身份给出，语气要公正、专业。请限制在 1024 个 tokens 以内。不要给出与问题无关的信息，若给定的上下文没有提供足够的信息，请回答“信息缺失，关于”后跟相关主题。

在回答中不要包括 [citation:x]，除非是代码或特定名称和引用，除此之外，其他部分必须与问题使用相同的语言。

以下是相关的上下文：

{context}

记住，不要机械地重复上下文内容。用户的问题是:
"""


@register_tool(include_functions=["run"])
@register_tool(include_functions=["run"])
class SearchEnhancedQA(Action):
    """通过搜索引擎结果回答问题并提供信息。"""

    name: str = "SearchEnhancedQA"  # 工具的名称
    desc: str = "通过集成搜索引擎结果来回答问题。"  # 工具的描述

    # 搜索引擎链接收集动作，用于从搜索引擎收集相关链接
    collect_links_action: CollectLinks = Field(
        default_factory=CollectLinks, description="从搜索引擎收集相关链接的动作。"
    )

    # 浏览网页并总结文章的动作，用于浏览并提取网页内容
    web_browse_and_summarize_action: WebBrowseAndSummarize = Field(
        default=None,
        description="浏览网页并提供文章和网页摘要的动作。",
    )

    # 每个页面的最大抓取时间（秒）
    per_page_timeout: float = Field(
        default=20, description="每个页面的最大抓取时间（秒），默认值为20秒。"
    )

    # 是否启用JavaScript
    java_script_enabled: bool = Field(
        default=False, description="是否启用JavaScript。默认为False。"
    )

    # 使用的用户代理字符串
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81",
        description="浏览器使用的用户代理。",
    )

    # 额外的HTTP请求头
    extra_http_headers: dict = Field(
        default={"sec-ch-ua": 'Chromium";v="125", "Not.A/Brand";v="24'},
        description="发送的额外HTTP请求头。",
    )

    # 每个网页摘要的最大字符数
    max_chars_per_webpage_summary: int = Field(
        default=4000, description="每个网页内容摘要的最大字符数。"
    )

    # 最大搜索结果数量
    max_search_results: int = Field(
        default=10,
        description="使用collect_links_action收集的最大搜索结果数量。这决定了用于回答问题的潜在来源的数量。",
    )

    _reporter: ThoughtReporter = PrivateAttr(ThoughtReporter())  # 用于记录思考过程的报告器

    @model_validator(mode="after")
    def initialize(self):
        """初始化动作，如果未提供浏览和总结网页的动作，则自动配置。"""
        if self.web_browse_and_summarize_action is None:
            web_browser_engine = WebBrowserEngine.from_browser_config(
                self.config.browser,
                proxy=self.config.proxy,
                java_script_enabled=self.java_script_enabled,
                extra_http_headers=self.extra_http_headers,
                user_agent=self.user_agent,
            )

            self.web_browse_and_summarize_action = WebBrowseAndSummarize(web_browser_engine=web_browser_engine)

        return self

    async def run(self, query: str, rewrite_query: bool = True) -> str:
        """通过网络搜索结果回答用户问题。

        Args:
            query (str): 用户的原始问题。
            rewrite_query (bool): 是否重新编写查询以获得更好的搜索结果。默认为True。

        Returns:
            str: 基于网页搜索结果的详细回答。

        Raises:
            ValueError: 如果查询无效。
        """
        async with self._reporter:
            await self._reporter.async_report({"type": "search", "stage": "init"})
            self._validate_query(query)

            processed_query = await self._process_query(query, rewrite_query)
            context = await self._build_context(processed_query)

            return await self._generate_answer(processed_query, context)

    def _validate_query(self, query: str) -> None:
        """验证输入的查询是否有效。

        Args:
            query (str): 要验证的查询。

        Raises:
            ValueError: 如果查询无效。
        """
        if not query.strip():
            raise ValueError("查询不能为空或仅包含空格。")

    async def _process_query(self, query: str, should_rewrite: bool) -> str:
        """处理查询，并根据需要重写查询。

        Args:
            query (str): 原始查询。
            should_rewrite (bool): 是否重新编写查询。

        Returns:
            str: 处理后的查询。
        """
        if should_rewrite:
            return await self._rewrite_query(query)

        return query

    async def _rewrite_query(self, query: str) -> str:
        """重写查询以适应更好的搜索引擎结果。

        如果重写失败，则返回原始查询。

        Args:
            query (str): 原始查询。

        Returns:
            str: 如果成功，返回重写后的查询，否则返回原始查询。
        """
        prompt = REWRITE_QUERY_PROMPT.format(q=query)

        try:
            resp = await self._aask(prompt)
            rewritten_query = self._extract_rewritten_query(resp)

            logger.info(f"查询重写: '{query}' -> '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.warning(f"查询重写失败，返回原查询。错误: {e}")
            return query

    def _extract_rewritten_query(self, response: str) -> str:
        """从LLM的JSON响应中提取重写后的查询。

        Args:
            response (str): LLM返回的响应。

        Returns:
            str: 重写后的查询。
        """
        resp_json = json.loads(CodeParser.parse_code(response, lang="json"))
        return resp_json["query"]

    async def _build_context(self, query: str) -> str:
        """基于网页搜索引用构建上下文字符串。

        Args:
            query (str): 搜索查询。

        Returns:
            str: 格式化的上下文，带有编号引用。
        """
        citations = await self._search_citations(query)
        context = "\n\n".join([f"[[citation:{i + 1}]] {c}" for i, c in enumerate(citations)])

        return context

    async def _search_citations(self, query: str) -> list[str]:
        """执行网络搜索并总结相关内容。

        Args:
            query (str): 搜索查询。

        Returns:
            list[str]: 相关网页内容的摘要。
        """
        relevant_urls = await self._collect_relevant_links(query)
        await self._reporter.async_report({"type": "search", "stage": "searching", "urls": relevant_urls})
        if not relevant_urls:
            logger.warning(f"未找到相关URL: {query}")
            return []

        logger.info(f"相关链接: {relevant_urls}")

        web_summaries = await self._summarize_web_content(relevant_urls)
        if not web_summaries:
            logger.warning(f"未生成摘要: {query}")
            return []

        citations = list(web_summaries.values())

        return citations

    async def _collect_relevant_links(self, query: str) -> list[str]:
        """搜索并排名相关的URL。

        Args:
            query (str): 搜索查询。

        Returns:
            list[str]: 排名的相关URL列表。
        """
        return await self.collect_links_action._search_and_rank_urls(
            topic=query, query=query, max_num_results=self.max_search_results
        )

    async def _summarize_web_content(self, urls: list[str]) -> dict[str, str]:
        """从给定的URL获取并总结内容。

        Args:
            urls (list[str]): 要总结的URL列表。

        Returns:
            dict[str, str]: URL到摘要的映射。
        """
        contents = await self._fetch_web_contents(urls)

        summaries = {}
        await self._reporter.async_report(
            {"type": "search", "stage": "browsing", "pages": [i.model_dump() for i in contents]}
        )
        for content in contents:
            url = content.url
            inner_text = content.inner_text.replace("\n", "")
            if self.web_browse_and_summarize_action._is_content_invalid(inner_text):
                logger.warning(f"URL {url} 内容无效: {inner_text[:10]}...")
                continue

            summary = inner_text[: self.max_chars_per_webpage_summary]
            summaries[url] = summary

        return summaries

    async def _fetch_web_contents(self, urls: list[str]) -> list[WebPage]:
        return await self.web_browse_and_summarize_action._fetch_web_contents(
            *urls, per_page_timeout=self.per_page_timeout
        )

    async def _generate_answer(self, query: str, context: str) -> str:
        """使用查询和上下文生成回答。

        Args:
            query (str): 用户的问题。
            context (str): 来自网络搜索的相关信息。

        Returns:
            str: 基于上下文生成的回答。
        """
        system_prompt = SEARCH_ENHANCED_QA_SYSTEM_PROMPT.format(context=context)

        async with ThoughtReporter(uuid=self._reporter.uuid, enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "search", "stage": "answer"})
            rsp = await self._aask(query, [system_prompt])
        return rsp
