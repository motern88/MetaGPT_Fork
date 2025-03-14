#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:45
@Author  : alexanderwu
@File    : read_document.py
"""

import docx


def read_docx(file_path: str) -> list:
    """打开一个 docx 文件并读取其中的内容"""

    # 打开指定路径的 docx 文件
    doc = docx.Document(file_path)

    # 创建一个空列表来存储段落的内容
    paragraphs_list = []

    # 遍历文档中的每个段落，并将其内容添加到列表中
    for paragraph in doc.paragraphs:
        paragraphs_list.append(paragraph.text)

    # 返回包含所有段落内容的列表
    return paragraphs_list
