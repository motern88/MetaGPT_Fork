#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : tutorial_assistant.py
@Describe : Tutorial Assistant's prompt templates.
"""

COMMON_PROMPT = """
你现在是一位资深的互联网技术专家。
我们需要你写一篇关于"{topic}"的技术教程。
"""

DIRECTORY_PROMPT = (
    COMMON_PROMPT
    + """
请为本教程提供具体的目录结构，严格按照以下要求：
1. 输出必须严格使用指定的语言，{language}。
2. 严格按照字典格式输出，类似这样：{{"title": "xxx", "directory": [{{"dir 1": ["sub dir 1", "sub dir 2"]}}, {{"dir 2": ["sub dir 3", "sub dir 4"]}}]}}。
3. 目录应尽可能具体且完整，包含主目录和子目录。子目录应为数组形式。
4. 不要有多余的空格和换行符。
5. 每个目录标题应具有实际意义。
"""
)

CONTENT_PROMPT = (
    COMMON_PROMPT
    + """
现在我将给你关于该主题的模块目录标题。
请详细输出每个标题的原理内容。
如果有代码示例，请按照标准代码规范提供。
如果没有代码示例，则不需要提供。

该主题的模块目录标题如下：
{directory}

严格按照以下要求输出：
1. 使用 Markdown 语法格式布局。
2. 如果有代码示例，必须遵循标准语法规范，并且有文档注释，且显示在代码块中。
3. 输出必须严格使用指定的语言，{language}。
4. 不要有多余的输出，包括结尾语句。
5. 严格要求不要输出主题"{topic}"。
"""
)
