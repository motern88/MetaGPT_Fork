#!/usr/bin/env python3
from __future__ import annotations

import sys


def print_flake8_output(input_string, show_line_numbers=False):
    """打印 Flake8 的输出结果。

    参数:
        input_string (str): Flake8 输出的字符串，包含警告或错误信息。
        show_line_numbers (bool): 是否显示行号。如果为True，则显示每个警告或错误的行号，默认为False。

    返回:
        None
    """
    # 遍历输入的每一行
    for value in input_string.split("\n"):
        parts = value.split()  # 将每行按空格分割成多个部分
        if not show_line_numbers:
            # 如果不显示行号，打印警告或错误信息（不包含行号）
            print(f"- {' '.join(parts[1:])}")
        else:
            # 如果需要显示行号，提取并格式化行号，打印行号和警告或错误信息
            line_nums = ":".join(parts[0].split(":")[1:])  # 提取行号部分并格式化
            print(f"- {line_nums} {' '.join(parts[1:])}")  # 打印行号和警告信息

if __name__ == "__main__":
    # 从命令行参数获取 Flake8 输出
    lint_output = sys.argv[1]
    print_flake8_output(lint_output)
