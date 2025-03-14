#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : Steven Lee
@File    : mmdc_playwright.py
"""

import os
from typing import List, Optional
from urllib.parse import urljoin

from playwright.async_api import async_playwright

from metagpt.logs import logger


async def mermaid_to_file(
        mermaid_code, output_file_without_suffix, width=2048, height=2048, suffixes: Optional[List[str]] = None
) -> int:
    """将Mermaid代码转换为各种文件格式。

    参数：
        mermaid_code (str): 要转换的Mermaid代码。
        output_file_without_suffix (str): 输出文件名，不带后缀。
        width (int, 可选): 输出图像的宽度，默认为2048。
        height (int, 可选): 输出图像的高度，默认为2048。
        suffixes (Optional[List[str]], 可选): 要生成的文件后缀，支持 "png"、"pdf" 和 "svg"。默认为 ["png"]。

    返回：
        int: 如果转换成功，返回0；如果转换失败，返回-1。
    """
    # 设置默认文件后缀为 "png"
    suffixes = suffixes or ["png"]
    # 获取当前脚本所在的目录
    __dirname = os.path.dirname(os.path.abspath(__file__))

    # 使用Playwright启动浏览器
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        device_scale_factor = 1.0  # 设备的缩放比例
        context = await browser.new_context(
            viewport={"width": width, "height": height},  # 设置浏览器窗口大小
            device_scale_factor=device_scale_factor,  # 设置设备的缩放比例
        )
        page = await context.new_page()

        # 记录控制台输出信息
        async def console_message(msg):
            logger.info(msg.text)

        page.on("console", console_message)

        try:
            # 设置页面视口大小
            await page.set_viewport_size({"width": width, "height": height})

            # 构建本地的HTML文件路径，加载Mermaid渲染的页面
            mermaid_html_path = os.path.abspath(os.path.join(__dirname, "index.html"))
            mermaid_html_url = urljoin("file:", mermaid_html_path)
            await page.goto(mermaid_html_url)
            await page.wait_for_load_state("networkidle")

            # 等待Mermaid容器加载完成
            await page.wait_for_selector("div#container", state="attached")

            # Mermaid配置项及CSS
            mermaid_config = {}
            background_color = "#ffffff"  # 设置背景色
            my_css = ""  # 额外的CSS样式
            # 设置页面背景色
            await page.evaluate(f'document.body.style.background = "{background_color}";')

            # 执行Mermaid渲染并将结果插入页面
            await page.evaluate(
                """async ([definition, mermaidConfig, myCSS, backgroundColor]) => {
                const { mermaid, zenuml } = globalThis;
                await mermaid.registerExternalDiagrams([zenuml]);
                mermaid.initialize({ startOnLoad: false, ...mermaidConfig });
                const { svg } = await mermaid.render('my-svg', definition, document.getElementById('container'));
                document.getElementById('container').innerHTML = svg;
                const svgElement = document.querySelector('svg');
                svgElement.style.backgroundColor = backgroundColor;

                if (myCSS) {
                    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
                    style.appendChild(document.createTextNode(myCSS));
                    svgElement.appendChild(style);
                }

            }""",
                [mermaid_code, mermaid_config, my_css, background_color],
            )

            # 生成SVG文件
            if "svg" in suffixes:
                svg_xml = await page.evaluate(
                    """() => {
                        const svg = document.querySelector('svg');
                        if (!svg) {
                            throw new Error('SVG element not found');
                        }
                        const xmlSerializer = new XMLSerializer();
                        return xmlSerializer.serializeToString(svg);
                    }"""
                )
                logger.info(f"正在生成 {output_file_without_suffix}.svg..")
                with open(f"{output_file_without_suffix}.svg", "wb") as f:
                    f.write(svg_xml.encode("utf-8"))

            # 生成PNG文件
            if "png" in suffixes:
                # 获取SVG元素的边界矩形
                clip = await page.evaluate(
                    """() => {
                    const svg = document.querySelector('svg');
                    const rect = svg.getBoundingClientRect();
                    return {
                        x: Math.floor(rect.left),
                        y: Math.floor(rect.top),
                        width: Math.ceil(rect.width),
                        height: Math.ceil(rect.height)
                    };
                }"""
                )
                # 设置视口大小以适应SVG的大小
                await page.set_viewport_size({"width": clip["x"] + clip["width"], "height": clip["y"] + clip["height"]})
                # 截取页面的截图
                screenshot = await page.screenshot(clip=clip, omit_background=True, scale="device")
                logger.info(f"正在生成 {output_file_without_suffix}.png..")
                with open(f"{output_file_without_suffix}.png", "wb") as f:
                    f.write(screenshot)

            # 生成PDF文件
            if "pdf" in suffixes:
                pdf_data = await page.pdf(scale=device_scale_factor)
                logger.info(f"正在生成 {output_file_without_suffix}.pdf..")
                with open(f"{output_file_without_suffix}.pdf", "wb") as f:
                    f.write(pdf_data)
            return 0
        except Exception as e:
            # 捕获并记录异常
            logger.error(e)
            return -1
        finally:
            # 关闭浏览器
            await browser.close()
