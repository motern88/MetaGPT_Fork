import contextlib
from uuid import uuid4

from metagpt.tools.libs.browser import Browser
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.file import MemoryFileSystem
from metagpt.utils.parse_html import simplify_html


@register_tool(tags=["web scraping"])
async def view_page_element_to_scrape(url: str, requirement: str, keep_links: bool = False) -> str:
    """查看当前页面的HTML内容，以了解页面结构。

    参数:
        url (str): 要抓取的网页URL。
        requirement (str): 提供清晰且详细的需求有助于集中检查目标元素。
        keep_links (bool): 是否保留HTML内容中的超链接。如果需要链接，可以设置为True。

    返回:
        str: 页面HTML内容。
    """
    # 使用浏览器访问页面
    async with Browser() as browser:
        await browser.goto(url)  # 跳转到目标URL
        page = browser.page
        html = await page.content()  # 获取页面的HTML内容
        # 简化HTML内容，去除多余部分
        html = simplify_html(html, url=page.url, keep_links=keep_links)

    # 使用内存文件系统存储HTML内容
    mem_fs = MemoryFileSystem()
    filename = f"{uuid4().hex}.html"  # 生成唯一的文件名
    with mem_fs.open(filename, "w") as f:
        f.write(html)

    # 尝试使用RAG优化，失败时回退到简化后的HTML内容
    with contextlib.suppress(Exception):
        from metagpt.rag.engines import SimpleEngine  # 避免循环导入

        # TODO：将 `from_docs` 转换为异步操作
        engine = SimpleEngine.from_docs(input_files=[filename], fs=mem_fs)
        nodes = await engine.aretrieve(requirement)  # 使用需求来检索相关内容
        html = "\n".join(i.text for i in nodes)  # 合并检索到的文本

    # 删除临时文件
    mem_fs.rm_file(filename)
    return html


# async def get_elements_outerhtml(self, element_ids: list[int]):
#     """Inspect the outer HTML of the elements in Current Browser Viewer.
#     """
#     page = self.page
#     data = []
#     for element_id in element_ids:
#         html = await get_element_outer_html(page, get_backend_node_id(element_id, self.accessibility_tree))
#         data.append(html)
#     return "\n".join(f"[{element_id}]. {html}" for element_id, html in zip(element_ids, data))
