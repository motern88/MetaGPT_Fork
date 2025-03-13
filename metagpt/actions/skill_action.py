#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/28
@Author  : mashenquan
@File    : skill_action.py
@Desc    : Call learned skill
"""
from __future__ import annotations

import ast
import importlib
import traceback
from copy import deepcopy
from typing import Dict, Optional

from metagpt.actions import Action
from metagpt.learn.skill_loader import Skill
from metagpt.logs import logger
from metagpt.schema import Message


# TOTEST
class ArgumentsParingAction(Action):
    skill: Skill
    ask: str
    rsp: Optional[Message] = None
    args: Optional[Dict] = None

    @property
    def prompt(self):
        # 构建提示信息，描述技能的参数和示例
        prompt = f"{self.skill.name} 功能参数描述:\n"
        for k, v in self.skill.arguments.items():
            prompt += f"参数 `{k}`: {v}\n"
        prompt += "\n---\n"
        prompt += "示例:\n"
        for e in self.skill.examples:
            prompt += f"如果你想让我做 `{e.ask}`，返回 `{e.answer}` 简洁明了。\n"
        prompt += "\n---\n"
        prompt += (
            f"\n参照 `{self.skill.name}` 函数描述，并根据示例中 '我想让你做xx' 填写函数参数。\n"
            f"现在我想让你做 `{self.ask}`，请返回像示例中一样的函数参数，简洁明了。"
        )
        return prompt

    async def run(self, with_message=None, **kwargs) -> Message:
        # 根据提示生成函数参数
        prompt = self.prompt
        rsp = await self.llm.aask(
            msg=prompt,
            system_msgs=["你是一个函数解析器。", "你可以将口语转换为函数参数。"],
            stream=False,
        )
        logger.debug(f"SKILL:{prompt}\n, RESULT:{rsp}")
        self.args = ArgumentsParingAction.parse_arguments(skill_name=self.skill.name, txt=rsp)
        self.rsp = Message(content=rsp, role="assistant", instruct_content=self.args, cause_by=self)
        return self.rsp

    @staticmethod
    def parse_arguments(skill_name, txt) -> dict:
        # 解析返回的文本，将其转换为函数参数字典
        prefix = skill_name + "("
        if prefix not in txt:
            logger.error(f"{skill_name} 不在 {txt} 中")
            return None
        if ")" not in txt:
            logger.error(f"')' 不在 {txt} 中")
            return None
        begin_ix = txt.find(prefix)
        end_ix = txt.rfind(")")
        args_txt = txt[begin_ix + len(prefix) : end_ix]
        logger.info(args_txt)
        fake_expression = f"dict({args_txt})"
        parsed_expression = ast.parse(fake_expression, mode="eval")
        args = {}
        for keyword in parsed_expression.body.keywords:
            key = keyword.arg
            value = ast.literal_eval(keyword.value)
            args[key] = value
        return args


class SkillAction(Action):
    skill: Skill
    args: Dict
    rsp: Optional[Message] = None

    async def run(self, with_message=None, **kwargs) -> Message:
        """执行技能操作"""
        options = deepcopy(kwargs)
        if self.args:
            for k in self.args.keys():
                if k in options:
                    options.pop(k)
        try:
            rsp = await self.find_and_call_function(self.skill.name, args=self.args, **options)
            self.rsp = Message(content=rsp, role="assistant", cause_by=self)
        except Exception as e:
            logger.exception(f"{e}, traceback:{traceback.format_exc()}")
            self.rsp = Message(content=f"错误: {e}", role="assistant", cause_by=self)
        return self.rsp

    @staticmethod
    async def find_and_call_function(function_name, args, **kwargs) -> str:
        try:
            # 导入模块并调用指定函数
            module = importlib.import_module("metagpt.learn")
            function = getattr(module, function_name)
            # 调用函数并返回结果
            result = await function(**args, **kwargs)
            return result
        except (ModuleNotFoundError, AttributeError):
            logger.error(f"{function_name} 未找到")
            raise ValueError(f"{function_name} 未找到")
