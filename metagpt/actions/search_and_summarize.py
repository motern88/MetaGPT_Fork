#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 17:26
@Author  : alexanderwu
@File    : search_google.py
"""
from typing import Optional

import pydantic
from pydantic import model_validator

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.tools.search_engine import SearchEngine

# 定义一个搜索和总结的系统，目的是根据历史对话和参考信息来总结当前的对话请求
SEARCH_AND_SUMMARIZE_SYSTEM = """### Requirements
1. 请根据参考信息（次要）和对话历史（主要）总结最新的对话内容。不要包含与对话无关的文本。
- 参考信息仅供参考。如果与用户的搜索请求历史无关，请减少参考信息的使用。
2. 如果参考信息中包含可引用的链接，请在主要文本中按照格式 [主要文本](引用链接) 注释。如果参考信息中没有链接，则不写链接。
3. 回复应该优雅、清晰、简洁、流畅，字数适中，使用 {LANG} 语言。

### Dialogue History (For example)
A: MLOps competitors

### Current Question (For example)
A: MLOps competitors

### Current Reply (For example)
1. Alteryx Designer: <desc> 等，如有的话
2. Matlab: 同上
3. IBM SPSS Statistics
4. RapidMiner Studio
5. DataRobot AI Platform
6. Databricks Lakehouse Platform
7. Amazon SageMaker
8. Dataiku
"""

# 搜索和总结系统的英文版本，格式化以适应美国英语（en-us）
SEARCH_AND_SUMMARIZE_SYSTEM_EN_US = SEARCH_AND_SUMMARIZE_SYSTEM.format(LANG="en-us")

# 搜索和总结的提示模板，要求根据参考信息和对话历史生成回复
SEARCH_AND_SUMMARIZE_PROMPT = """
### Reference Information
{CONTEXT}

### Dialogue History
{QUERY_HISTORY}
{QUERY}

### Current Question
{QUERY}

### Current Reply: 基于以上信息，请写出对该问题的回答
"""

# 销售系统版本的搜索和总结要求和模板，适用于简体中文
SEARCH_AND_SUMMARIZE_SALES_SYSTEM = """## Requirements
1. 请根据参考信息（次要）和对话历史（主要）总结最新的对话内容。不要包含与对话无关的文本。
- 参考信息仅供参考。如果与用户的搜索请求历史无关，请减少参考信息的使用。
2. 如果参考信息中包含可引用的链接，请在主要文本中按照格式 [主要文本](引用链接) 注释。如果参考信息中没有链接，则不写链接。
3. 回复应该优雅、清晰、简洁、流畅，字数适中，使用简体中文。

# 示例
## Reference Information
...

## Dialogue History
user: 哪款洁面乳适合油性皮肤？
Salesperson: 你好，针对油性皮肤，建议选择深层清洁、控油、温和不刺激的产品。根据客户反馈和市场口碑，推荐以下洁面乳：...
user: 有L'Oreal的产品吗？
> Salesperson: ...

## 理想答案
是的，我为您精选了以下几款：
1. L'Oreal男士洁面乳：控油、抗痘、水油平衡、毛孔净化、有效去黑头，深层清洁，洗后不紧绷，泡沫丰富。
2. L'Oreal Age Perfect保湿洁面乳：加入了椰油酰甘氨酸钠和积雪草两种有效成分，深层清洁、紧致皮肤，温和不紧绷。
"""

# 销售版本的搜索和总结提示模板，适用于简体中文
SEARCH_AND_SUMMARIZE_SALES_PROMPT = """
## Reference Information
{CONTEXT}

## Dialogue History
{QUERY_HISTORY}
{QUERY}
> {ROLE}: 
"""

# 用户查询请求示例：厦门有哪些美味的食物？
SEARCH_FOOD = """
# 用户搜索请求
厦门有哪些美味的食物？

# 要求
您是专业管家团队的一员，提供有帮助的建议：
1. 请根据上下文总结用户的搜索请求，避免包含无关的文本。
2. 使用 [主要文本](引用链接) 的 markdown 格式，在主要文本中**自然地注释** 3-5个文本元素（如产品词汇或类似的文本段落），以便用户轻松导航。
3. 回复应优雅、清晰，**避免重复文本**，并且顺畅流畅，字数适中。
"""


class SearchAndSummarize(Action):
    # 类的属性定义
    name: str = ""  # 存储动作名称
    content: Optional[str] = None  # 可选的内容属性
    search_engine: SearchEngine = None  # 搜索引擎实例
    result: str = ""  # 存储最终的结果

    # 使用 Pydantic 模型验证器，在运行时检查搜索引擎的配置
    @model_validator(mode="after")
    def validate_search_engine(self):
        # 如果 search_engine 为空，则尝试从配置中加载搜索引擎
        if self.search_engine is None:
            try:
                config = self.config  # 获取配置
                # 尝试使用配置中的搜索引擎信息初始化 SearchEngine 实例
                search_engine = SearchEngine.from_search_config(config.search, proxy=config.proxy)
            except pydantic.ValidationError:
                search_engine = None  # 如果初始化失败，则将搜索引擎设为 None

            self.search_engine = search_engine  # 更新搜索引擎属性
        return self  # 返回当前对象，确保链式调用

    # 异步运行方法，根据上下文查询并生成回复
    async def run(self, context: list[Message], system_text=SEARCH_AND_SUMMARIZE_SYSTEM) -> str:
        # 如果没有配置搜索引擎，输出警告并返回空字符串
        if self.search_engine is None:
            logger.warning("Configure one of SERPAPI_API_KEY, SERPER_API_KEY, GOOGLE_API_KEY to unlock full feature")
            return ""

        query = context[-1].content  # 获取对话历史中的最后一条消息作为查询
        # 调用搜索引擎的 run 方法执行查询，获取搜索结果
        rsp = await self.search_engine.run(query)
        self.result = rsp  # 保存搜索结果
        if not rsp:  # 如果结果为空，输出错误并返回空字符串
            logger.error("empty rsp...")
            return ""

        # 准备系统提示语，用于给定的上下文信息生成对话
        system_prompt = [system_text]

        # 格式化搜索和总结的提示，包含角色、上下文、查询历史和当前查询
        prompt = SEARCH_AND_SUMMARIZE_PROMPT.format(
            ROLE=self.prefix,  # 获取角色前缀
            CONTEXT=rsp,  # 将搜索结果作为上下文传入
            QUERY_HISTORY="\n".join([str(i) for i in context[:-1]]),  # 格式化历史对话
            QUERY=str(context[-1]),  # 获取当前查询
        )
        # 异步调用 _aask 方法，生成最终的对话结果
        result = await self._aask(prompt, system_prompt)
        # 输出调试信息，查看生成的提示和最终结果
        logger.debug(prompt)
        logger.debug(result)
        return result  # 返回生成的结果
