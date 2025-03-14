#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : alitrack
@File    : mmdc_pyppeteer.py
"""
import os
from typing import List, Optional
from urllib.parse import urljoin

from pyppeteer import launch

from metagpt.config2 import Config
from metagpt.logs import logger


async def mermaid_to_file(
    mermaid_code, output_file_without_suffix, width=2048, height=2048, config=None, suffixes: Optional[List[str]] = None
) -> int:
    """将Mermaid代码转换为多种文件格式。

    参数:
        mermaid_code (str): 需要转换的Mermaid代码。
        output_file_without_suffix (str): 输出文件名（不包括后缀）。
        width (int, 可选): 输出图像的宽度，默认为2048。
        height (int, 可选): 输出图像的高度，默认为2048。
        config (Optional[Config], 可选): 转换时使用的配置，默认为None，使用默认配置。
        suffixes (Optional[List[str]], 可选): 生成的文件后缀，支持"png"、"pdf"和"svg"，默认为["png"]。

    返回:
        int: 如果转换成功返回0，失败返回-1。
    """
    config = config if config else Config.default()  # 使用默认配置
    suffixes = suffixes or ["png"]  # 如果没有提供后缀，则默认为["png"]
    __dirname = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录

    # 如果配置文件中有pyppeteer路径，使用它启动浏览器
    if config.mermaid.pyppeteer_path:
        browser = await launch(
            headless=True,
            executablePath=config.mermaid.pyppeteer_path,  # 使用配置中的pyppeteer路径
            args=["--disable-extensions", "--no-sandbox"],  # 禁用扩展和沙箱模式
        )
    else:
        logger.error("请在config2.yaml中设置mermaid.pyppeteer_path变量。")
        return -1  # 如果没有设置pyppeteer路径，返回-1

    page = await browser.newPage()  # 创建新页面
    device_scale_factor = 1.0  # 设备缩放因子

    async def console_message(msg):
        logger.info(msg.text)  # 打印浏览器控制台日志

    page.on("console", console_message)  # 监听控制台消息

    try:
        # 设置浏览器视口
        await page.setViewport(viewport={"width": width, "height": height, "deviceScaleFactor": device_scale_factor})

        # 加载本地HTML文件
        mermaid_html_path = os.path.abspath(os.path.join(__dirname, "index.html"))
        mermaid_html_url = urljoin("file:", mermaid_html_path)
        await page.goto(mermaid_html_url)

        await page.querySelector("div#container")  # 等待页面加载完成
        mermaid_config = {}
        background_color = "#ffffff"  # 设置背景颜色
        my_css = ""  # 自定义CSS样式
        await page.evaluate(f'document.body.style.background = "{background_color}";')  # 应用背景颜色

        # 渲染Mermaid图形
        await page.evaluate(
            """async ([definition, mermaidConfig, myCSS, backgroundColor]) => {
            const { mermaid, zenuml } = globalThis;
            await mermaid.registerExternalDiagrams([zenuml]);  // 注册外部图表
            mermaid.initialize({ startOnLoad: false, ...mermaidConfig });  // 初始化Mermaid
            const { svg } = await mermaid.render('my-svg', definition, document.getElementById('container'));  // 渲染SVG
            document.getElementById('container').innerHTML = svg;  // 将SVG插入容器
            const svgElement = document.querySelector('svg');
            svgElement.style.backgroundColor = backgroundColor;  // 设置SVG背景色

            if (myCSS) {
                const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
                style.appendChild(document.createTextNode(myCSS));
                svgElement.appendChild(style);  // 应用自定义CSS样式
            }
        }""",
            [mermaid_code, mermaid_config, my_css, background_color],
        )

        # 如果需要生成SVG文件
        if "svg" in suffixes:
            svg_xml = await page.evaluate(
                """() => {
                const svg = document.querySelector('svg');
                const xmlSerializer = new XMLSerializer();
                return xmlSerializer.serializeToString(svg);  // 序列化SVG为字符串
            }"""
            )
            logger.info(f"生成{output_file_without_suffix}.svg..")
            with open(f"{output_file_without_suffix}.svg", "wb") as f:
                f.write(svg_xml.encode("utf-8"))

        # 如果需要生成PNG文件
        if "png" in suffixes:
            clip = await page.evaluate(
                """() => {
                const svg = document.querySelector('svg');
                const rect = svg.getBoundingClientRect();  // 获取SVG的矩形区域
                return {
                    x: Math.floor(rect.left),
                    y: Math.floor(rect.top),
                    width: Math.ceil(rect.width),
                    height: Math.ceil(rect.height)
                };
            }"""
            )
            # 设置浏览器视口以适应SVG图形
            await page.setViewport(
                {
                    "width": clip["x"] + clip["width"],
                    "height": clip["y"] + clip["height"],
                    "deviceScaleFactor": device_scale_factor,
                }
            )
            screenshot = await page.screenshot(clip=clip, omit_background=True, scale="device")  # 截图
            logger.info(f"生成{output_file_without_suffix}.png..")
            with open(f"{output_file_without_suffix}.png", "wb") as f:
                f.write(screenshot)

        # 如果需要生成PDF文件
        if "pdf" in suffixes:
            pdf_data = await page.pdf(scale=device_scale_factor)  # 生成PDF
            logger.info(f"生成{output_file_without_suffix}.pdf..")
            with open(f"{output_file_without_suffix}.pdf", "wb") as f:
                f.write(pdf_data)

        return 0  # 成功完成转换
    except Exception as e:
        logger.error(e)  # 捕获并记录错误
        return -1  # 转换失败
    finally:
        await browser.close()  # 关闭浏览器
