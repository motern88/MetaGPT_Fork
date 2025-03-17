#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 16:30:25
@Author  : Stitch-z
@File    : invoice_ocr.py
@Describe : Prompts of the invoice ocr assistant.
"""

# 定义一个通用的提示信息，说明即将提供OCR识别结果的发票数据。
COMMON_PROMPT = "现在我将提供发票的OCR文本识别结果。"

# 提取发票的主要信息，包括收款人、城市、总费用和开票日期的提示模板
EXTRACT_OCR_MAIN_INFO_PROMPT = (
    COMMON_PROMPT
    + """
请提取发票的收款人、城市、总费用和开票日期。

发票的OCR数据如下：
{ocr_result}

强制性限制要求如下：
1. 总费用指的是总价格和税费。不要包括 `¥` 符号。
2. 城市必须是收款人所在的城市。
3. 返回的JSON字典必须使用 {language} 语言。
4. 强制要求以JSON格式输出: {{"收款人":"x","城市":"x","总费用/元":"","开票日期":""}}.
"""
)

# 回答OCR相关问题的提示模板
REPLY_OCR_QUESTION_PROMPT = (
    COMMON_PROMPT
    + """
请回答以下问题：{query}

发票的OCR数据如下：
{ocr_result}

强制性限制要求如下：
1. 用 {language} 语言回答。
2. 不允许返回收到的OCR数据。
3. 返回时使用Markdown语法格式。
"""
)

# OCR成功完成发票文本识别的提示信息
INVOICE_OCR_SUCCESS = "成功完成发票OCR文本识别。"
