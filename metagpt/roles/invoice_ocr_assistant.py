#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 14:10:05
@Author  : Stitch-z
@File    : invoice_ocr_assistant.py
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from metagpt.actions.invoice_ocr import GenerateTable, InvoiceOCR, ReplyQuestion
from metagpt.prompts.invoice_ocr import INVOICE_OCR_SUCCESS
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message


class InvoicePath(BaseModel):
    # 定义文件路径，类型为 Path，默认值为空字符串
    file_path: Path = ""


class OCRResults(BaseModel):
    # 定义 OCR 结果，类型为字符串，默认值为空的 JSON 数组字符串
    ocr_result: str = "[]"


class InvoiceData(BaseModel):
    # 定义发票数据，类型为包含字典的列表，默认值为空列表
    invoice_data: list[dict] = []


class ReplyData(BaseModel):
    # 定义回复数据，类型为字符串，默认值为空字符串
    content: str = ""


class InvoiceOCRAssistant(Role):
    """Invoice OCR 助手，支持对发票的 PDF、png、jpg 和 zip 文件进行 OCR 文字识别，
    生成包含收款人、城市、总金额和开票日期的发票信息表格，并根据 OCR 识别结果对单个文件进行提问。

    参数：
        name: 角色名称
        profile: 角色简介
        goal: 角色目标
        constraints: 角色约束或要求
        language: 生成发票表格的语言
    """

    name: str = "Stitch"  # 角色名称
    profile: str = "Invoice OCR Assistant"  # 角色简介
    goal: str = "OCR 识别发票文件并生成发票主要信息表格"  # 角色目标
    constraints: str = ""  # 角色约束（为空）
    language: str = "ch"  # 语言设置（中文）
    filename: str = ""  # 文件名
    origin_query: str = ""  # 初始查询内容
    orc_data: Optional[list] = None  # OCR 数据（可选）

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([InvoiceOCR])  # 设置初始的 OCR 行动
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)  # 设置反应模式为按顺序反应

    async def _act(self) -> Message:
        """执行角色确定的操作。

        返回：
            包含操作结果的消息。
        """
        msg = self.rc.memory.get(k=1)[0]  # 从记忆中获取最近的一条消息
        todo = self.rc.todo  # 获取待办事项
        if isinstance(todo, InvoiceOCR):
            # 如果当前待办事项是 OCR 识别
            self.origin_query = msg.content  # 保存原始查询内容
            invoice_path: InvoicePath = msg.instruct_content  # 获取指令内容中的文件路径
            file_path = invoice_path.file_path
            self.filename = file_path.name  # 获取文件名
            if not file_path:
                raise Exception("未上传发票文件")  # 如果文件路径为空，抛出异常

            # 执行 OCR 识别操作
            resp = await todo.run(file_path)
            actions = list(self.actions)  # 获取当前的操作列表
            if len(resp) == 1:
                # 如果只有一个 OCR 结果，支持基于 OCR 识别结果的提问
                actions.extend([GenerateTable, ReplyQuestion])
                self.orc_data = resp[0]
            else:
                actions.append(GenerateTable)  # 如果有多个 OCR 结果，追加生成表格操作
            self.set_actions(actions)  # 更新操作列表
            self.rc.max_react_loop = len(self.actions)  # 设置最大反应循环次数
            content = INVOICE_OCR_SUCCESS  # 设置 OCR 识别成功的消息内容
            resp = OCRResults(ocr_result=json.dumps(resp))  # 将 OCR 结果转换为 OCRResults 类型
        elif isinstance(todo, GenerateTable):
            # 如果当前待办事项是生成表格
            ocr_results: OCRResults = msg.instruct_content  # 获取 OCR 结果
            resp = await todo.run(json.loads(ocr_results.ocr_result), self.filename)  # 执行生成表格操作

            # 将生成的列表转换为 Markdown 格式的表格字符串
            df = pd.DataFrame(resp)
            markdown_table = df.to_markdown(index=False)
            content = f"{markdown_table}\n\n\n"  # 设置表格内容
            resp = InvoiceData(invoice_data=resp)  # 将表格结果转换为 InvoiceData 类型
        else:
            # 如果是其他操作，执行原始查询
            resp = await todo.run(self.origin_query, self.orc_data)
            content = resp
            resp = ReplyData(content=resp)  # 将回复内容转换为 ReplyData 类型

        # 创建消息并将其添加到记忆中
        msg = Message(content=content, instruct_content=resp)
        self.rc.memory.add(msg)  # 将消息添加到记忆中
        return msg
