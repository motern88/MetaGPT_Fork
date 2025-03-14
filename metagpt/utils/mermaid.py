#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/4 10:53
@Author  : alexanderwu alitrack
@File    : mermaid.py
"""
import asyncio
import os
import re
from pathlib import Path
from typing import List, Optional

from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.utils.common import awrite, check_cmd_exists


async def mermaid_to_file(
        engine,
        mermaid_code,
        output_file_without_suffix,
        width=2048,
        height=2048,
        config=None,
        suffixes: Optional[List[str]] = None,
) -> int:
    """
    将 Mermaid 代码转换为各种文件格式。

    参数:
        engine (str): 用于转换的引擎。支持的引擎有 "nodejs"、"playwright"、"pyppeteer"、"ink" 和 "none"。
        mermaid_code (str): 要转换的 Mermaid 代码。
        output_file_without_suffix (str): 输出文件名（不包括后缀）。
        width (int, 可选): 输出图像的宽度。默认值为 2048。
        height (int, 可选): 输出图像的高度。默认值为 2048。
        config (Optional[Config], 可选): 用于转换的配置。如果为 None，则使用默认配置。
        suffixes (Optional[List[str]], 可选): 生成的文件后缀。支持 "png"、"pdf" 和 "svg"。默认值为 ["svg"]。

    返回:
        int: 如果转换成功，返回 0；如果转换失败，返回 -1。
    """
    file_head = "%%{init: {'theme': 'default', 'themeVariables': { 'fontFamily': 'Inter' }}}%%\n"
    if not re.match(r"^%%\{.+", mermaid_code):
        mermaid_code = file_head + mermaid_code  # 如果 Mermaid 代码没有初始化部分，则添加

    suffixes = suffixes or ["svg"]  # 默认使用 svg 后缀
    # 将 Mermaid 代码写入临时文件
    config = config if config else Config.default()
    dir_name = os.path.dirname(output_file_without_suffix)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)  # 创建输出文件所在的目录（如果不存在）
    tmp = Path(f"{output_file_without_suffix}.mmd")
    await awrite(filename=tmp, data=mermaid_code)  # 异步写入临时文件

    # 如果引擎是 nodejs
    if engine == "nodejs":
        if check_cmd_exists(config.mermaid.path) != 0:
            logger.warning(
                "运行 `npm install -g @mermaid-js/mermaid-cli` 来安装 mmdc，"
                "或者考虑更改引擎为 `playwright`、`pyppeteer` 或 `ink`。"
            )
            return -1  # 如果没有安装 mermaid-cli，则返回失败

        for suffix in suffixes:
            output_file = f"{output_file_without_suffix}.{suffix}"
            # 调用 `mmdc` 命令将 Mermaid 代码转换为 PNG
            logger.info(f"生成 {output_file}..")

            if config.mermaid.puppeteer_config:
                commands = [
                    config.mermaid.path,
                    "-p",
                    config.mermaid.puppeteer_config,
                    "-i",
                    str(tmp),
                    "-o",
                    output_file,
                    "-w",
                    str(width),
                    "-H",
                    str(height),
                ]
            else:
                commands = [config.mermaid.path, "-i", str(tmp), "-o", output_file, "-w", str(width), "-H", str(height)]

            # 异步执行命令并获取输出
            process = await asyncio.create_subprocess_shell(
                " ".join(commands), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            if stdout:
                logger.info(stdout.decode())
            if stderr:
                logger.warning(stderr.decode())

    else:
        # 对于其他引擎，调用相应的转换方法
        if engine == "playwright":
            from metagpt.utils.mmdc_playwright import mermaid_to_file

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, width, height, suffixes=suffixes)
        elif engine == "pyppeteer":
            from metagpt.utils.mmdc_pyppeteer import mermaid_to_file

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, width, height, suffixes=suffixes)
        elif engine == "ink":
            from metagpt.utils.mmdc_ink import mermaid_to_file

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, suffixes=suffixes)
        elif engine == "none":
            return 0  # 如果引擎为 "none"，不进行任何转换
        else:
            logger.warning(f"不支持的 Mermaid 引擎: {engine}")
    return 0  # 默认成功返回 0


# 示例 Mermaid 代码：类图
MMC1 = """
classDiagram
    class Main {
        -SearchEngine search_engine
        +main() str
    }
    class SearchEngine {
        -Index index
        -Ranking ranking
        -Summary summary
        +search(query: str) str
    }
    class Index {
        -KnowledgeBase knowledge_base
        +create_index(data: dict)
        +query_index(query: str) list
    }
    class Ranking {
        +rank_results(results: list) list
    }
    class Summary {
        +summarize_results(results: list) str
    }
    class KnowledgeBase {
        +update(data: dict)
        +fetch_data(query: str) dict
    }
    Main --> SearchEngine
    SearchEngine --> Index
    SearchEngine --> Ranking
    SearchEngine --> Summary
    Index --> KnowledgeBase
"""

# 示例 Mermaid 代码：时序图
MMC2 = """
sequenceDiagram
    participant M as Main
    participant SE as SearchEngine
    participant I as Index
    participant R as Ranking
    participant S as Summary
    participant KB as KnowledgeBase
    M->>SE: search(query)
    SE->>I: query_index(query)
    I->>KB: fetch_data(query)
    KB-->>I: return data
    I-->>SE: return results
    SE->>R: rank_results(results)
    R-->>SE: return ranked_results
    SE->>S: summarize_results(ranked_results)
    S-->>SE: return summary
    SE-->>M: return summary
"""
