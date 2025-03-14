#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : human interaction to get required type text

import json
from typing import Any, Tuple, Type

from pydantic import BaseModel

from metagpt.logs import logger
from metagpt.utils.common import import_class


class HumanInteraction(object):
    stop_list = ("q", "quit", "exit")  # 定义停止输入的关键词，用户输入这些词时，结束交互

    def multilines_input(self, prompt: str = "Enter: ") -> str:
        # 提示用户输入多行文本，直到输入结束（Ctrl-D 或 Ctrl-Z）
        logger.warning("请输入内容，使用 Ctrl-D 或 Ctrl-Z ( Windows ) 来保存输入。")
        logger.info(f"{prompt}\n")
        lines = []  # 用于保存输入的多行内容
        while True:
            try:
                line = input()  # 获取每一行输入
                lines.append(line)  # 将输入内容追加到列表中
            except EOFError:  # 捕捉输入结束的异常（如 Ctrl-D 或 Ctrl-Z）
                break
        return "".join(lines)  # 将所有行连接为一个字符串并返回

    def check_input_type(self, input_str: str, req_type: Type) -> Tuple[bool, Any]:
        # 检查输入的内容是否符合所需的类型
        check_ret = True  # 默认为符合要求
        if req_type == str:
            # 如果需要的是字符串类型，直接返回
            return check_ret, input_str
        try:
            input_str = input_str.strip()  # 去除输入内容的首尾空格
            data = json.loads(input_str)  # 尝试将输入的字符串转换为 JSON 格式
        except Exception:
            return False, None  # 如果转换失败，返回 False 和 None

        # 动态导入 ActionNode 类以避免循环导入
        actionnode_class = import_class("ActionNode", "metagpt.actions.action_node")
        tmp_key = "tmp"
        # 动态创建一个临时的类，用于验证输入数据是否符合 req_type
        tmp_cls = actionnode_class.create_model_class(class_name=tmp_key.upper(), mapping={tmp_key: (req_type, ...)})
        try:
            _ = tmp_cls(**{tmp_key: data})  # 尝试用输入数据实例化这个类
        except Exception:
            check_ret = False  # 如果实例化失败，说明输入数据不符合要求
        return check_ret, data

    def input_until_valid(self, prompt: str, req_type: Type) -> Any:
        # 循环检查输入直到其符合要求
        while True:
            input_content = self.multilines_input(prompt)  # 获取用户输入
            check_ret, structure_content = self.check_input_type(input_content, req_type)  # 检查输入内容类型
            if check_ret:
                break  # 如果输入符合要求，退出循环
            else:
                logger.error(f"输入内容不符合要求的类型: {req_type}，请重新输入。")
        return structure_content  # 返回符合要求的内容

    def input_num_until_valid(self, num_max: int) -> int:
        # 循环输入直到输入的数字有效
        while True:
            input_num = input("请输入交互键的编号: ")  # 提示用户输入编号
            input_num = input_num.strip()  # 去除空格
            if input_num in self.stop_list:
                return input_num  # 如果输入了停止词，返回停止标志
            try:
                input_num = int(input_num)  # 尝试将输入转为整数
                if 0 <= input_num < num_max:  # 如果输入的数字在有效范围内，返回该数字
                    return input_num
            except Exception:
                pass  # 如果转换失败，继续循环

    def interact_with_instruct_content(
        self, instruct_content: BaseModel, mapping: dict = dict(), interact_type: str = "review"
    ) -> dict[str, Any]:
        # 与指令内容进行交互，用户可以选择查看或修改字段内容
        assert interact_type in ["review", "revise"]  # 确保交互类型有效
        assert instruct_content  # 确保指令内容不为空
        instruct_content_dict = instruct_content.model_dump()  # 将指令内容转换为字典
        num_fields_map = dict(zip(range(0, len(instruct_content_dict)), instruct_content_dict.keys()))  # 将字段映射为数字编号
        logger.info(
            f"\n{interact_type.upper()} 交互\n"
            f"交互数据: {num_fields_map}\n"
            f"输入编号选择要交互的字段，或者输入 `q`/`quit`/`exit` 停止交互。\n"
            f"请输入字段内容，直到符合字段要求的类型。\n"
        )

        interact_contents = {}  # 用于保存交互内容
        while True:
            input_num = self.input_num_until_valid(len(instruct_content_dict))  # 获取有效的字段编号
            if input_num in self.stop_list:  # 如果输入了停止词，结束交互
                logger.warning("停止人工交互")
                break

            field = num_fields_map.get(input_num)  # 获取对应字段名称
            logger.info(f"你选择交互的字段是: {field}，进行 `{interact_type}` 操作。")

            if interact_type == "review":
                prompt = "请输入你的评论: "
                req_type = str  # 查看操作需要输入字符串类型的评论
            else:
                prompt = "请输入你的修改内容: "
                req_type = mapping.get(field)[0]  # 修改操作需要输入符合字段要求的内容

            field_content = self.input_until_valid(prompt=prompt, req_type=req_type)  # 获取有效的输入内容
            interact_contents[field] = field_content  # 将内容保存到交互内容中

        return interact_contents  # 返回所有交互内容
