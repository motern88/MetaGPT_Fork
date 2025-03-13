#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 18:10:20
@Author  : Stitch-z
@File    : invoice_ocr.py
@Describe : Actions of the invoice ocr assistant.
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from paddleocr import PaddleOCR

from metagpt.actions import Action
from metagpt.const import INVOICE_OCR_TABLE_PATH
from metagpt.logs import logger
from metagpt.prompts.invoice_ocr import (
    EXTRACT_OCR_MAIN_INFO_PROMPT,
    REPLY_OCR_QUESTION_PROMPT,
)
from metagpt.utils.common import OutputParser
from metagpt.utils.file import File


class InvoiceOCR(Action):
    """处理发票文件的OCR操作类，包括zip、PDF、png和jpg文件格式。

    参数：
        name: 操作的名称，默认为空字符串。
        language: OCR输出的语言，默认为"ch"（中文）。

    """

    name: str = "InvoiceOCR"  # 定义操作名称
    i_context: Optional[str] = None  # 可选的上下文，用于存储一些额外信息

    @staticmethod
    async def _check_file_type(file_path: Path) -> str:
        """检查给定文件路径的文件类型。

        参数：
            file_path: 文件的路径。

        返回：
            文件类型，根据FileExtensionType枚举返回。

        异常：
            Exception: 如果文件格式不是zip、pdf、png或jpg，抛出异常。
        """
        ext = file_path.suffix  # 获取文件扩展名
        if ext not in [".zip", ".pdf", ".png", ".jpg"]:  # 检查文件扩展名是否有效
            raise Exception("The invoice format is not zip, pdf, png, or jpg")  # 如果无效，抛出异常

        return ext  # 返回文件类型

    @staticmethod
    async def _unzip(file_path: Path) -> Path:
        """解压文件并返回解压后的文件夹路径。

        参数：
            file_path: zip文件的路径。

        返回：
            解压后的文件夹路径。
        """
        # 生成解压后的目标文件夹路径
        file_directory = file_path.parent / "unzip_invoices" / datetime.now().strftime("%Y%m%d%H%M%S")
        with zipfile.ZipFile(file_path, "r") as zip_ref:  # 打开zip文件
            for zip_info in zip_ref.infolist():  # 遍历zip文件中的所有内容
                # 使用CP437编码文件名，再用GBK解码以防中文乱码
                relative_name = Path(zip_info.filename.encode("cp437").decode("gbk"))
                if relative_name.suffix:  # 如果是有效的文件名
                    full_filename = file_directory / relative_name  # 生成完整路径
                    await File.write(full_filename.parent, relative_name.name, zip_ref.read(zip_info.filename))  # 解压文件内容

        logger.info(f"unzip_path: {file_directory}")  # 输出解压后的路径
        return file_directory  # 返回解压后的文件夹路径

    @staticmethod
    async def _ocr(invoice_file_path: Path):
        """执行OCR识别发票文件。

        参数：
            invoice_file_path: 发票文件路径。

        返回：
            OCR结果。
        """
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=1)  # 创建OCR对象，支持角度分类器，语言设置为中文
        ocr_result = ocr.ocr(str(invoice_file_path), cls=True)  # 执行OCR识别
        for result in ocr_result[0]:  # 遍历OCR识别结果
            result[1] = (result[1][0], round(result[1][1], 2))  # 将置信度分数四舍五入，减少token消耗
        return ocr_result  # 返回OCR结果

    async def run(self, file_path: Path, *args, **kwargs) -> list:
        """执行OCR操作以识别发票文件。

        参数：
            file_path: 输入文件路径。

        返回：
            OCR结果列表。
        """
        file_ext = await self._check_file_type(file_path)  # 检查文件类型

        if file_ext == ".zip":  # 如果是zip文件
            # 解压zip批量文件
            unzip_path = await self._unzip(file_path)
            ocr_list = []  # 存储OCR识别结果
            for root, _, files in os.walk(unzip_path):  # 遍历解压后的文件夹
                for filename in files:
                    invoice_file_path = Path(root) / Path(filename)  # 获取每个文件的完整路径
                    # 识别符合条件的文件
                    if Path(filename).suffix in [".zip", ".pdf", ".png", ".jpg"]:
                        ocr_result = await self._ocr(str(invoice_file_path))  # 执行OCR
                        ocr_list.append(ocr_result)  # 保存OCR结果
            return ocr_list  # 返回所有OCR结果

        else:
            # OCR识别单个文件
            ocr_result = await self._ocr(file_path)  # 执行OCR
            return [ocr_result]  # 返回OCR结果


class GenerateTable(Action):
    """根据OCR结果生成表格的操作类。

    参数：
        name: 操作的名称，默认为空字符串。
        language: 用于生成表格的语言，默认为"ch"（中文）。

    """

    name: str = "GenerateTable"  # 定义操作名称
    i_context: Optional[str] = None  # 可选的上下文
    language: str = "ch"  # 语言设置为中文

    async def run(self, ocr_results: list, filename: str, *args, **kwargs) -> dict[str, str]:
        """处理OCR结果，提取发票信息，生成表格，并将其保存为Excel文件。

        参数：
            ocr_results: 从发票处理获得的OCR结果列表。
            filename: 输出Excel文件的名称。

        返回：
            包含发票信息的字典。
        """
        table_data = []  # 用于存储表格数据
        pathname = INVOICE_OCR_TABLE_PATH  # 表格保存路径
        pathname.mkdir(parents=True, exist_ok=True)  # 确保保存路径存在

        for ocr_result in ocr_results:  # 遍历OCR结果
            # 提取发票的主要信息
            prompt = EXTRACT_OCR_MAIN_INFO_PROMPT.format(ocr_result=ocr_result, language=self.language)
            ocr_info = await self._aask(prompt=prompt)  # 使用预设的提示进行提问，提取信息
            invoice_data = OutputParser.extract_struct(ocr_info, dict)  # 解析结果为字典格式
            if invoice_data:  # 如果提取到信息
                table_data.append(invoice_data)  # 将数据添加到表格数据中

        # 生成Excel文件
        filename = f"{filename.split('.')[0]}.xlsx"  # 设置文件名为xlsx格式
        full_filename = f"{pathname}/{filename}"  # 生成完整文件路径
        df = pd.DataFrame(table_data)  # 使用pandas生成数据框
        df.to_excel(full_filename, index=False)  # 保存为Excel文件，不包含索引
        return table_data  # 返回表格数据


class ReplyQuestion(Action):
    """根据OCR结果生成回复的操作类。

    参数：
        name: 操作的名称，默认为空字符串。
        language: 用于生成回复的语言，默认为"ch"（中文）。

    """

    language: str = "ch"  # 语言设置为中文

    async def run(self, query: str, ocr_result: list, *args, **kwargs) -> str:
        """根据OCR结果回复问题。

        参数：
            query: 需要回复的问题。
            ocr_result: OCR结果的列表。

        返回：
            生成的回复。
        """
        prompt = REPLY_OCR_QUESTION_PROMPT.format(query=query, ocr_result=ocr_result, language=self.language)
        resp = await self._aask(prompt=prompt)  # 使用预设的提示生成回复
        return resp  # 返回回复结果