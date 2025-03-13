#!/usr/bin/env python

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional, Union

from pydantic import TypeAdapter, model_validator

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.search_engine import SearchEngine
from metagpt.tools.web_browser_engine import WebBrowserEngine
from metagpt.utils.common import OutputParser
from metagpt.utils.parse_html import WebPage
from metagpt.utils.text import generate_prompt_chunk, reduce_message_length

# 语言提示
LANG_PROMPT = "Please respond in {language}."  # 用于设置回答语言的提示

# 研究助手系统提示：用于引导AI生成批判性思维的研究报告
RESEARCH_BASE_SYSTEM = """You are an AI critical thinker research assistant. Your sole purpose is to write well \
written, critically acclaimed, objective and structured reports on the given text."""  # 该系统的任务是撰写结构化、客观且有深度的研究报告

# 研究主题助手系统提示：基于特定研究主题提供帮助
RESEARCH_TOPIC_SYSTEM = "You are an AI researcher assistant, and your research topic is:\n#TOPIC#\n{topic}"  # 研究助手系统提示，要求明确研究主题

# 搜索关键词提示：引导用户提供与研究主题相关的关键字以便进行搜索
SEARCH_TOPIC_PROMPT = """Please provide up to 2 necessary keywords related to your research topic for Google search. \
Your response must be in JSON format, for example: ["keyword1", "keyword2"]."""  # 提示用户提供2个关键字，格式为JSON

# 汇总搜索结果的提示：引导AI根据搜索结果生成相关查询，确保查询与研究主题相关
SUMMARIZE_SEARCH_PROMPT = """### Requirements
1. The keywords related to your research topic and the search results are shown in the "Search Result Information" section.
2. Provide up to {decomposition_nums} queries related to your research topic base on the search results.
3. Please respond in the following JSON format: ["query1", "query2", "query3", ...].

### Search Result Information
{search_results}
"""  # 汇总并分析搜索结果，以生成相关查询的要求

# 搜索结果的收集与排序提示：要求删除与查询无关的结果，并根据可信度对相关结果进行排序
COLLECT_AND_RANKURLS_PROMPT = """### Topic
{topic}
### Query
{query}

### The online search results
{results}

### Requirements
Please remove irrelevant search results that are not related to the query or topic.
If the query is time-sensitive or specifies a certain time frame, please also remove search results that are outdated or outside the specified time frame. Notice that the current time is {time_stamp}.
Then, sort the remaining search results based on the link credibility. If two results have equal credibility, prioritize them based on the relevance.
Provide the ranked results' indices in JSON format, like [0, 1, 3, 4, ...], without including other words.
"""  # 根据查询和时间要求，删除不相关和过时的搜索结果，并按链接可信度和相关性排序

# 在线浏览并总结结果的提示：要求根据参考信息回答问题或总结相关文本
WEB_BROWSE_AND_SUMMARIZE_PROMPT = """### Requirements
1. Utilize the text in the "Reference Information" section to respond to the question "{query}".
2. If the question cannot be directly answered using the text, but the text is related to the research topic, please provide \
a comprehensive summary of the text.
3. If the text is entirely unrelated to the research topic, please reply with a simple text "Not relevant."
4. Include all relevant factual information, numbers, statistics, etc., if available.

### Reference Information
{content}
"""  # 根据参考信息回答问题或提供摘要，确保回答准确并包含相关数据

# 进行研究的提示：基于提供的信息撰写详细的研究报告
CONDUCT_RESEARCH_PROMPT = """### Reference Information
{content}

### Requirements
Please provide a detailed research report in response to the following topic: "{topic}", using the information provided \
above. The report must meet the following requirements:

- Focus on directly addressing the chosen topic.
- Ensure a well-structured and in-depth presentation, incorporating relevant facts and figures where available.
- Present data and findings in an intuitive manner, utilizing feature comparative tables, if applicable.
- The report should have a minimum word count of 2,000 and be formatted with Markdown syntax following APA style guidelines.
- Include all source URLs in APA format at the end of the report.
"""  # 编写详尽的研究报告，符合APA格式并包含相关的事实和数据


class CollectLinks(Action):
    """用于从搜索引擎收集链接的动作类。"""

    name: str = "CollectLinks"  # 动作的名称
    i_context: Optional[str] = None  # 可选的上下文
    desc: str = "从搜索引擎收集链接。"  # 动作描述
    search_func: Optional[Any] = None  # 搜索功能函数
    search_engine: Optional[SearchEngine] = None  # 搜索引擎对象
    rank_func: Optional[Callable[[list[str]], None]] = None  # 可选的排名函数

    @model_validator(mode="after")
    def validate_engine_and_run_func(self):
        """验证搜索引擎是否存在，若不存在则初始化一个默认搜索引擎。"""
        if self.search_engine is None:
            self.search_engine = SearchEngine.from_search_config(self.config.search, proxy=self.config.proxy)
        return self

    async def run(
            self,
            topic: str,
            decomposition_nums: int = 4,
            url_per_query: int = 4,
            system_text: str | None = None,
    ) -> dict[str, list[str]]:
        """运行动作，收集链接并返回相关搜索结果。

        参数:
            topic: 研究主题
            decomposition_nums: 用于生成的查询数量
            url_per_query: 每个查询返回的链接数量
            system_text: 系统文本

        返回:
            包含查询问题及其对应链接的字典
        """
        # 生成关键词
        system_text = system_text if system_text else RESEARCH_TOPIC_SYSTEM.format(topic=topic)
        keywords = await self._aask(SEARCH_TOPIC_PROMPT, [system_text])

        # 处理关键词
        try:
            keywords = OutputParser.extract_struct(keywords, list)
            keywords = TypeAdapter(list[str]).validate_python(keywords)
        except Exception as e:
            logger.exception(f"获取与研究主题 '{topic}' 相关的关键词失败: {e}")
            keywords = [topic]

        # 异步执行搜索
        results = await asyncio.gather(*(self.search_engine.run(i, as_string=False) for i in keywords))

        def gen_msg():
            while True:
                search_results = "\n".join(
                    f"#### 关键词: {i}\n 搜索结果: {j}\n" for (i, j) in zip(keywords, results)
                )
                prompt = SUMMARIZE_SEARCH_PROMPT.format(
                    decomposition_nums=decomposition_nums, search_results=search_results
                )
                yield prompt
                remove = max(results, key=len)
                remove.pop()
                if len(remove) == 0:
                    break

        # 生成查询并获取排名
        model_name = self.config.llm.model
        prompt = reduce_message_length(gen_msg(), model_name, system_text, self.config.llm.max_token)
        logger.debug(prompt)
        queries = await self._aask(prompt, [system_text])

        # 处理查询
        try:
            queries = OutputParser.extract_struct(queries, list)
            queries = TypeAdapter(list[str]).validate_python(queries)
        except Exception as e:
            logger.exception(f"拆解研究问题失败: {e}")
            queries = keywords

        ret = {}
        for query in queries:
            ret[query] = await self._search_and_rank_urls(topic, query, url_per_query)
        return ret

    async def _search_and_rank_urls(
            self, topic: str, query: str, num_results: int = 4, max_num_results: int = None
    ) -> list[str]:
        """根据查询对结果进行搜索和排序。

        参数:
            topic: 研究主题
            query: 搜索查询
            num_results: 返回的链接数量
            max_num_results: 最大链接数量

        返回:
            按排名排序的链接列表
        """
        max_results = max_num_results or max(num_results * 2, 6)
        results = await self._search_urls(query, max_results=max_results)

        if len(results) == 0:
            return []

        _results = "\n".join(f"{i}: {j}" for i, j in zip(range(max_results), results))
        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = COLLECT_AND_RANKURLS_PROMPT.format(topic=topic, query=query, results=_results, time_stamp=time_stamp)
        logger.debug(prompt)

        # 排名链接
        indices = await self._aask(prompt)
        try:
            indices = OutputParser.extract_struct(indices, list)
            assert all(isinstance(i, int) for i in indices)
        except Exception as e:
            logger.exception(f"链接排序失败: {e}")
            indices = list(range(max_results))

        results = [results[i] for i in indices]
        if self.rank_func:
            results = self.rank_func(results)

        return [i["link"] for i in results[:num_results]]

    async def _search_urls(self, query: str, max_results: int) -> list[dict[str, str]]:
        """使用搜索引擎获取链接。

        返回:
            e.g. [{"title": "...", "link": "...", "snippet", "..."}]
        """
        return await self.search_engine.run(query, max_results=max_results, as_string=False)

class WebBrowseAndSummarize(Action):
    """用于浏览网页并提供网页内容摘要的动作类。"""

    name: str = "WebBrowseAndSummarize"  # 动作的名称
    i_context: Optional[str] = None  # 可选的上下文
    desc: str = "浏览网页并提供网页内容的摘要。"  # 动作描述
    browse_func: Union[Callable[[list[str]], None], None] = None  # 可选的浏览功能
    web_browser_engine: Optional[WebBrowserEngine] = None  # 网页浏览引擎

    @model_validator(mode="after")
    def validate_engine_and_run_func(self):
        """验证浏览器引擎是否存在，若不存在则初始化默认的浏览器引擎。"""
        if self.web_browser_engine is None:
            self.web_browser_engine = WebBrowserEngine.from_browser_config(
                self.config.browser,
                browse_func=self.browse_func,
                proxy=self.config.proxy,
            )
        return self

    async def run(
        self,
        url: str,
        *urls: str,
        query: str,
        system_text: str = RESEARCH_BASE_SYSTEM,
        use_concurrent_summarization: bool = False,
        per_page_timeout: Optional[float] = None,
    ) -> dict[str, str]:
        """运行动作，浏览网页并提供摘要。

        参数:
            url: 主网页URL
            urls: 其他网页URL
            query: 研究问题
            system_text: 系统文本
            use_concurrent_summarization: 是否并发摘要网页内容
            per_page_timeout: 获取每个页面的最大时间

        返回:
            包含URL及其摘要的字典
        """
        # 获取网页内容
        contents = await self._fetch_web_contents(url, *urls, per_page_timeout=per_page_timeout)

        all_urls = [url] + list(urls)
        summarize_tasks = [self._summarize_content(content, query, system_text) for content in contents]
        summaries = await self._execute_summarize_tasks(summarize_tasks, use_concurrent_summarization)
        result = {url: summary for url, summary in zip(all_urls, summaries) if summary}

        return result

    async def _fetch_web_contents(
        self, url: str, *urls: str, per_page_timeout: Optional[float] = None
    ) -> list[WebPage]:
        """从指定URL获取网页内容。"""
        contents = await self.web_browser_engine.run(url, *urls, per_page_timeout=per_page_timeout)
        return [contents] if not urls else contents

    async def _summarize_content(self, page: WebPage, query: str, system_text: str) -> str:
        """对网页内容进行摘要。"""
        try:
            prompt_template = WEB_BROWSE_AND_SUMMARIZE_PROMPT.format(query=query, content="{}")
            content = page.inner_text

            if self._is_content_invalid(content):
                logger.warning(f"检测到无效内容: {content[:10]}...")
                return None

            chunk_summaries = []
            for prompt in generate_prompt_chunk(content, prompt_template, self.llm.model, system_text, 4096):
                logger.debug(prompt)
                summary = await self._aask(prompt, [system_text])
                if summary == "Not relevant.":
                    continue
                chunk_summaries.append(summary)

            if not chunk_summaries:
                return None

            if len(chunk_summaries) == 1:
                return chunk_summaries[0]

            content = "\n".join(chunk_summaries)
            prompt = WEB_BROWSE_AND_SUMMARIZE_PROMPT.format(query=query, content=content)
            summary = await self._aask(prompt, [system_text])
            return summary
        except Exception as e:
            logger.error(f"摘要失败: {e}")
            return None

    def _is_content_invalid(self, content: str) -> bool:
        """检查内容是否无效，基于特定的起始短语。

        Args:
            content: 网页内容。

        Returns:
            bool: 如果内容以无效短语开始，则返回True，否则返回False。
        """
        invalid_starts = ["Fail to load page", "Access Denied"]

        # 检查内容是否以无效短语开始
        return any(content.strip().startswith(phrase) for phrase in invalid_starts)

    async def _execute_summarize_tasks(self, tasks: list[Coroutine[Any, Any, str]], use_concurrent: bool) -> list[str]:
        """执行摘要任务，可以选择并发执行或顺序执行。

        Args:
            tasks: 摘要任务的协程列表。
            use_concurrent: 是否使用并发执行任务。

        Returns:
            list[str]: 包含每个任务摘要的列表。
        """
        if use_concurrent:
            # 并发执行任务
            return await asyncio.gather(*tasks)

        # 顺序执行任务
        return [await task for task in tasks]


class ConductResearch(Action):
    """用于进行研究并生成研究报告的 Action 类。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run(
            self,
            topic: str,
            content: str,
            system_text: str = RESEARCH_BASE_SYSTEM,
    ) -> str:
        """执行研究操作并生成研究报告。

        Args:
            topic: 研究的主题。
            content: 进行研究的内容。
            system_text: 系统文本，默认是 RESEARCH_BASE_SYSTEM。

        Returns:
            str: 生成的研究报告。
        """
        # 格式化提示语，将主题和内容加入其中
        prompt = CONDUCT_RESEARCH_PROMPT.format(topic=topic, content=content)
        logger.debug(prompt)

        # 设置 LLM 自动最大化 tokens 数量
        self.llm.auto_max_tokens = True

        # 使用 _aask 异步请求并返回研究报告
        return await self._aask(prompt, [system_text])


def get_research_system_text(topic: str, language: str):
    """获取进行研究时的系统文本。

    Args:
        topic: 研究的主题。
        language: 系统文本的语言。

    Returns:
        str: 生成的研究系统文本。
    """
    # 将主题和语言的相关信息格式化为系统文本
    return " ".join((RESEARCH_TOPIC_SYSTEM.format(topic=topic), LANG_PROMPT.format(language=language)))