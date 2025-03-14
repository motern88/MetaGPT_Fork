#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 16:03
@Author  : alexanderwu
@File    : prompt_writer.py
"""
from typing import Union


class GPTPromptGenerator:
    """使用LLM，给定输出，要求LLM提供输入（支持指令、聊天机器人和查询风格）"""

    def __init__(self):
        # 创建一个字典，将风格映射到相应的生成方法
        self._generators = {i: getattr(self, f"gen_{i}_style") for i in ["instruction", "chatbot", "query"]}

    def gen_instruction_style(self, example):
        """指令风格：给定输出，要求LLM提供输入"""
        return f"""Instruction: X
Output: {example}
What kind of instruction might this output come from?
X:"""

    def gen_chatbot_style(self, example):
        """聊天机器人风格：给定输出，要求LLM提供输入"""
        return f"""You are a chatbot. A user sent you an informal message, and you replied as follows.
Message: X
Reply: {example}
What could the informal message X be?
X:"""

    def gen_query_style(self, example):
        """查询风格：给定输出，要求LLM提供输入"""
        return f"""You are a search engine. Someone made a detailed query, and the most relevant document to this query is as follows.
Query: X
Document: {example} What is the detailed query X?
X:"""

    def gen(self, example: str, style: str = "all") -> Union[List[str], str]:
        """
        使用示例生成一个或多个输出，允许LLM回复相应的输入

        :param example: 预期的LLM输出示例
        :param style: (all|instruction|chatbot|query)
        :return: 预期的LLM输入示例（一个或多个）
        """
        if style != "all":
            return self._generators[style](example)
        # 如果风格是“all”，则返回所有风格的生成结果
        return [f(example) for f in self._generators.values()]


class WikiHowTemplate:
    """生成有关问题的步骤模板"""

    def __init__(self):
        self._prompts = """给我{step}步来{question}。
How to {question}?
Do you know how can I {question}?
List {step} instructions to {question}.
What are some tips to {question}?
What are some steps to {question}?
Can you provide {step} clear and concise instructions on how to {question}?
I'm interested in learning how to {question}. Could you break it down into {step} easy-to-follow steps?
For someone who is new to {question}, what would be {step} key steps to get started?
What is the most efficient way to {question}? Could you provide a list of {step} steps?
Do you have any advice on how to {question} successfully? Maybe a step-by-step guide with {step} steps?
I'm trying to accomplish {question}. Could you walk me through the process with {step} detailed instructions?
What are the essential {step} steps to {question}?
I need to {question}, but I'm not sure where to start. Can you give me {step} actionable steps?
As a beginner in {question}, what are the {step} basic steps I should take?
I'm looking for a comprehensive guide on how to {question}. Can you provide {step} detailed steps?
Could you outline {step} practical steps to achieve {question}?
What are the {step} fundamental steps to consider when attempting to {question}?"""

    def gen(self, question: str, step: str) -> List[str]:
        """生成与问题相关的步骤模板"""
        return self._prompts.format(question=question, step=step).splitlines()


class EnronTemplate:
    """生成有关电子邮件主题的模板"""

    def __init__(self):
        self._prompts = """写一封主题为"{subj}"的电子邮件。
Can you craft an email with the subject {subj}?
Would you be able to compose an email and use {subj} as the subject?
Create an email about {subj}.
Draft an email and include the subject "{subj}".
Generate an email about {subj}.
Hey, can you shoot me an email about {subj}?
Do you mind crafting an email for me with {subj} as the subject?
Can you whip up an email with the subject of "{subj}"?
Hey, can you write an email and use "{subj}" as the subject?
Can you send me an email about {subj}?"""

    def gen(self, subj):
        """生成关于电子邮件主题的模板"""
        return self._prompts.format(subj=subj).splitlines()


class BEAGECTemplate:
    """生成与文档编辑相关的模板"""

    def __init__(self):
        self._prompts = """编辑和修订此文档以提高其语法、词汇、拼写和风格。
Revise this document to correct all the errors related to grammar, spelling, and style.
Refine this document by eliminating all grammatical, lexical, and orthographic errors and improving its writing style.
Polish this document by rectifying all errors related to grammar, vocabulary, and writing style.
Enhance this document by correcting all the grammar errors and style issues, and improving its overall quality.
Rewrite this document by fixing all grammatical, lexical and orthographic errors.
Fix all grammar errors and style issues and rewrite this document.
Take a stab at fixing all the mistakes in this document and make it sound better.
Give this document a once-over and clean up any grammar or spelling errors.
Tweak this document to make it read smoother and fix any mistakes you see.
Make this document sound better by fixing all the grammar, spelling, and style issues.
Proofread this document and fix any errors that make it sound weird or confusing."""

    def gen(self):
        """生成文档编辑相关的模板"""
        return self._prompts.splitlines()
