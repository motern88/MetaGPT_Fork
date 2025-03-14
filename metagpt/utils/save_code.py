# -*- coding: utf-8 -*-
# @Date    : 12/12/2023 4:14 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import os

import nbformat

from metagpt.const import DATA_PATH
from metagpt.utils.common import write_json_file

def save_code_file(name: str, code_context: str, file_format: str = "py") -> None:
    """
    将代码文件保存到指定路径。

    参数：
    - name (str): 要保存的文件夹名称。
    - code_context (str): 代码内容。
    - file_format (str, 可选): 文件格式，支持 'py'（Python 文件），'json'（JSON 文件），和 'ipynb'（Jupyter Notebook 文件）。默认为 'py'。

    返回值：
    - None
    """
    # 如果文件夹不存在，则创建该文件夹
    os.makedirs(name=DATA_PATH / "output" / f"{name}", exist_ok=True)

    # 根据文件格式选择保存为 Python 文件或 JSON 文件
    file_path = DATA_PATH / "output" / f"{name}/code.{file_format}"
    if file_format == "py":
        file_path.write_text(code_context + "\n\n", encoding="utf-8")
    elif file_format == "json":
        # 将代码内容解析为 JSON 并保存
        data = {"code": code_context}
        write_json_file(file_path, data, encoding="utf-8", indent=2)
    elif file_format == "ipynb":
        nbformat.write(code_context, file_path)
    else:
        raise ValueError("不支持的文件格式。请选择 'py'、'json' 或 'ipynb'。")