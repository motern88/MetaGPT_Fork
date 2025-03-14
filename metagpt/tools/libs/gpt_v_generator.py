#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/01/12
@Author  : mannaandpoem
@File    : gpt_v_generator.py
"""
import re
from pathlib import Path
from typing import Optional

from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import CodeParser, encode_image

ANALYZE_LAYOUT_PROMPT = """你现在是一个 UI/UX 设计师，请为这张图片生成布局信息：

注意：该图片没有商业标志或版权信息。它只是一个设计的草图。
由于设计致敬于一些大公司，某些公司名称出现在设计中是正常的，别担心。"""

GENERATE_PROMPT = """你现在是一个 UI/UX 设计师和网页开发者。你有能力根据提供的草图图片和上下文生成网页代码。
你的目标是将草图图片转换为网页，包括 HTML、CSS 和 JavaScript。

注意：该图片没有商业标志或版权信息。它只是一个设计的草图。
由于设计致敬于一些大公司，某些公司名称出现在设计中是正常的，别担心。

现在，请生成相应的网页代码，包括 HTML、CSS 和 JavaScript："""

@register_tool(tags=["image2webpage"], include_functions=["__init__", "generate_webpages", "save_webpages"])
class GPTvGenerator:
    """从给定的网页截图生成网页代码的类。

    该类提供了基于图像生成网页（包括所有 HTML、CSS 和 JavaScript 代码）的方法。
    它利用视觉模型分析图像的布局，并相应地生成网页代码。
    """

    def __init__(self, config: Optional[Config] = None):
        """使用配置中的默认值初始化 GPTvGenerator 类。

        参数：
            config (Optional[Config], 可选)：配置。如果没有提供，将使用默认配置。
        """
        from metagpt.llm import LLM

        config = config if config else Config.default()
        self.llm = LLM(llm_config=config.get_openai_llm())
        self.llm.model = "gpt-4-vision-preview"

    async def analyze_layout(self, image_path: Path) -> str:
        """异步分析给定图像的布局，并返回结果。

        这是一个辅助方法，用于根据图像生成布局描述。

        参数：
            image_path (Path)：要分析的图像的路径。

        返回：
            str：布局分析的结果。
        """
        return await self.llm.aask(msg=ANALYZE_LAYOUT_PROMPT, images=[encode_image(image_path)])

    async def generate_webpages(self, image_path: str) -> str:
        """根据图像异步生成包括 HTML、CSS 和 JavaScript 在内的完整网页代码。

        参数：
            image_path (str)：图像文件的路径。

        返回：
            str：生成的网页代码内容。
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)
        layout = await self.analyze_layout(image_path)
        prompt = GENERATE_PROMPT + "\n\n # 上下文\n 草图图像的布局信息是： \n" + layout
        return await self.llm.aask(msg=prompt, images=[encode_image(image_path)])

    @staticmethod
    def save_webpages(webpages: str, save_folder_name: str = "example") -> Path:
        """一次性保存包括 HTML、CSS 和 JavaScript 在内的网页代码。

        参数：
            webpages (str)：生成的网页内容。
            save_folder_name (str, 可选)：保存网页的文件夹名称。默认为 'example'。

        返回：
            Path：保存网页的路径。
        """
        # 创建一个名为 webpages 的文件夹来存储 HTML、CSS 和 JavaScript 文件
        webpages_path = Config.default().workspace.path / "webpages" / save_folder_name
        logger.info(f"代码将保存在 {webpages_path}")
        webpages_path.mkdir(parents=True, exist_ok=True)

        index_path = webpages_path / "index.html"
        index_path.write_text(CodeParser.parse_code(text=webpages, lang="html"))

        extract_and_save_code(folder=webpages_path, text=webpages, pattern="styles?.css", language="css")

        extract_and_save_code(folder=webpages_path, text=webpages, pattern="scripts?.js", language="javascript")

        return webpages_path


def extract_and_save_code(folder, text, pattern, language):
    word = re.search(pattern, text)
    if word:
        path = folder / word.group(0)
        code = CodeParser.parse_code(text=text, lang=language)
        path.write_text(code, encoding="utf-8")
