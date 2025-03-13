#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/27
@Author  : mashenquan
@File    : write_teaching_plan.py
"""
from typing import Optional

from metagpt.actions import Action
from metagpt.context import Context
from metagpt.logs import logger


# 定义 WriteTeachingPlanPart 类，表示写教学计划的某一部分
class WriteTeachingPlanPart(Action):
    """编写教学计划部分"""

    i_context: Optional[str] = None  # 可选的上下文信息
    topic: str = ""  # 教学计划部分的主题
    language: str = "Chinese"  # 教学语言，默认为中文
    rsp: Optional[str] = None  # 最终的响应结果

    # 异步运行方法，生成教学计划的相关内容
    async def run(self, with_message=None, **kwargs):
        # 获取该主题下的声明模式（语句模板）
        statement_patterns = TeachingPlanBlock.TOPIC_STATEMENTS.get(self.topic, [])
        statements = []  # 用于存储生成的语句

        # 将每个声明模式格式化为具体的语句
        for p in statement_patterns:
            s = self.format_value(p, context=self.context)
            statements.append(s)

        # 根据主题选择合适的提示模板
        formatter = (
            TeachingPlanBlock.PROMPT_TITLE_TEMPLATE
            if self.topic == TeachingPlanBlock.COURSE_TITLE
            else TeachingPlanBlock.PROMPT_TEMPLATE
        )

        # 格式化提示，生成最终的提示文本
        prompt = formatter.format(
            formation=TeachingPlanBlock.FORMATION,
            role=self.prefix,
            statements="\n".join(statements),
            lesson=self.i_context,
            topic=self.topic,
            language=self.language,
        )

        # 输出调试信息
        logger.debug(prompt)

        # 通过模型生成响应结果
        rsp = await self._aask(prompt=prompt)

        # 输出生成的响应内容
        logger.debug(rsp)

        # 设置结果并返回
        self._set_result(rsp)
        return self.rsp

    def _set_result(self, rsp):
        """处理生成的响应结果并格式化"""
        # 如果响应包含教学计划开始标签，截取标签后的内容
        if TeachingPlanBlock.DATA_BEGIN_TAG in rsp:
            ix = rsp.index(TeachingPlanBlock.DATA_BEGIN_TAG)
            rsp = rsp[ix + len(TeachingPlanBlock.DATA_BEGIN_TAG):]
        # 如果响应包含教学计划结束标签，截取标签前的内容
        if TeachingPlanBlock.DATA_END_TAG in rsp:
            ix = rsp.index(TeachingPlanBlock.DATA_END_TAG)
            rsp = rsp[0:ix]

        # 去除响应中的多余空白并设置为最终的响应
        self.rsp = rsp.strip()

        # 如果主题不是教学标题，则跳过以下逻辑
        if self.topic != TeachingPlanBlock.COURSE_TITLE:
            return

        # 如果响应结果没有以 "#" 开头，则在前面加上 "#" 以表示一级标题
        if "#" not in self.rsp or self.rsp.index("#") != 0:
            self.rsp = "# " + self.rsp

    def __str__(self):
        """返回主题的字符串表示"""
        return self.topic

    def __repr__(self):
        """在调试时显示主题的字符串表示"""
        return self.topic

    @staticmethod
    def format_value(value, context: Context):
        """填充 `value` 中的参数，使用 `context` 中的选项"""
        # 如果值不是字符串，则直接返回
        if not isinstance(value, str):
            return value

        # 如果值中没有需要格式化的占位符，直接返回
        if "{" not in value:
            return value

        # 获取模型的配置选项
        options = context.config.model_dump()

        # 用提供的参数（kwargs）覆盖模型中的默认选项
        for k, v in context.kwargs:
            options[k] = v  # 允许 None 值覆盖模型中的值

        # 只保留值不为 None 的选项
        opts = {k: v for k, v in options.items() if v is not None}

        try:
            # 尝试格式化字符串
            return value.format(**opts)
        except KeyError as e:
            # 如果缺少参数，则记录警告
            logger.warning(f"参数缺失: {e}")

        # 替换占位符为实际值
        for k, v in opts.items():
            value = value.replace("{" + f"{k}" + "}", str(v))
        return value


# 定义教学计划块类，包含教学计划的常量和模板
class TeachingPlanBlock:
    # 定义教学计划的结构
    FORMATION = (
        '"Capacity and role" 定义你当前的角色；\n'
        '\t"[LESSON_BEGIN]" 和 "[LESSON_END]" 标签包含教材内容；\n'
        '\t"Statement" 定义你在此阶段需要完成的工作细节；\n'
        '\t"Answer options" 定义你响应的格式要求；\n'
        '\t"Constraint" 定义你的响应必须遵守的条件。'
    )

    COURSE_TITLE = "Title"  # 教学计划的标题
    # 定义所有教学计划的主题
    TOPICS = [
        COURSE_TITLE,
        "Teaching Hours",
        "Teaching Objectives",
        "Teaching Content",
        "Teaching Methods and Strategies",
        "Learning Activities",
        "Teaching Time Allocation",
        "Assessment and Feedback",
        "Teaching Summary and Improvement",
        "Vocabulary Cloze",
        "Choice Questions",
        "Grammar Questions",
        "Translation Questions",
    ]

    # 定义每个主题的声明模板
    TOPIC_STATEMENTS = {
        COURSE_TITLE: [
            "Statement: 仅返回教学计划的标题，使用 markdown 的一级标题格式，不需要其他内容。"
        ],
        "Teaching Content": [
            'Statement: "教学内容" 必须包括教材中出现的词汇、语法结构分析与例子，'
            "以及听力材料和重点内容。",
            'Statement: "教学内容" 必须包含更多的例子。',
        ],
        "Teaching Time Allocation": [
            'Statement: "教学时间分配" 必须包括教材内容的各个部分的时间分配。'
        ],
        "Teaching Methods and Strategies": [
            'Statement: "教学方法与策略" 必须详细包括教学重点、难点、材料、程序等内容。'
        ],
        "Vocabulary Cloze": [
            'Statement: 根据教材内容，创建词汇填空题。题目应包括10个{language}的问题和{teaching_language}的答案，'
            "同时也应包括10个{teaching_language}的问题和{language}的答案。教材内容中的关键词汇和短语必须包含在练习中。",
        ],
        "Grammar Questions": [
            'Statement: 根据教材内容，创建语法题。10道题目。'
        ],
        "Choice Questions": [
            'Statement: 根据教材内容，创建选择题。10道题目。'
        ],
        "Translation Questions": [
            'Statement: 根据教材内容，创建翻译题。题目应包括10个{language}的问题和{teaching_language}的答案，'
            "同时也应包括10个{teaching_language}的问题和{language}的答案。",
        ],
    }

    # 教学计划标题模板
    PROMPT_TITLE_TEMPLATE = (
        "不要参考之前对话记录的上下文，重新开始对话。\n\n"
        "Formation: {formation}\n\n"
        "{statements}\n"
        "Constraint: 使用{language}编写。\n"
        'Answer options: 使用"[TEACHING_PLAN_BEGIN]"和"[TEACHING_PLAN_END]"标签来包含教学计划标题。\n'
        "[LESSON_BEGIN]\n"
        "{lesson}\n"
        "[LESSON_END]"
    )

    # 教学计划部分模板
    PROMPT_TEMPLATE = (
        "不要参考之前对话记录的上下文，重新开始对话。\n\n"
        "Formation: {formation}\n\n"
        "Capacity and role: {role}\n"
        'Statement: 写 "{topic}" 部分的教学计划，'
        '仅包含与 "{topic}" 相关的内容，无其他内容！！\n'
        "{statements}\n"
        'Answer options: 使用"[TEACHING_PLAN_BEGIN]"和"[TEACHING_PLAN_END]"标签来包含教学计划内容。\n'
        "Answer options: 使用第二级标题格式的markdown。\n"
        "Constraint: 使用{language}编写。\n"
        "[LESSON_BEGIN]\n"
        "{lesson}\n"
        "[LESSON_END]"
    )

    # 教学计划数据标签
    DATA_BEGIN_TAG = "[TEACHING_PLAN_BEGIN]"
    DATA_END_TAG = "[TEACHING_PLAN_END]"
