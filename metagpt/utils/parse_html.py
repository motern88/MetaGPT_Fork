#!/usr/bin/env python
from __future__ import annotations

from typing import Generator, Optional
from urllib.parse import urljoin, urlparse

import htmlmin
from bs4 import BeautifulSoup
from pydantic import BaseModel, PrivateAttr


class WebPage(BaseModel):
    """
    用于表示网页的类，包含网页的内文、HTML 内容和 URL 地址。
    """

    inner_text: str  # 网页的纯文本内容
    html: str         # 网页的 HTML 内容
    url: str          # 网页的 URL 地址

    # 私有属性，用于存储解析后的 BeautifulSoup 对象和网页标题
    _soup: Optional[BeautifulSoup] = PrivateAttr(default=None)
    _title: Optional[str] = PrivateAttr(default=None)

    @property
    def soup(self) -> BeautifulSoup:
        """
        解析 HTML 内容为 BeautifulSoup 对象，若已经解析过则直接返回缓存的对象。

        Returns:
            BeautifulSoup: 解析后的 BeautifulSoup 对象。
        """
        if self._soup is None:
            self._soup = BeautifulSoup(self.html, "html.parser")
        return self._soup

    @property
    def title(self):
        """
        获取网页标题，如果缓存中没有，则从 HTML 中提取标题。

        Returns:
            str: 网页的标题。
        """
        if self._title is None:
            title_tag = self.soup.find("title")
            self._title = title_tag.text.strip() if title_tag is not None else ""
        return self._title

    def get_links(self) -> Generator[str, None, None]:
        """
        提取网页中的所有链接（href 属性），并根据 URL 类型进行规范化。

        Yields:
            str: 网页中的每个链接 URL。
        """
        for i in self.soup.find_all("a", href=True):
            url = i["href"]
            result = urlparse(url)
            if not result.scheme and result.path:
                # 如果链接没有 scheme，使用网页的 URL 进行拼接
                yield urljoin(self.url, url)
            elif url.startswith(("http://", "https://")):
                # 如果链接已经是完整的 URL，则直接返回
                yield urljoin(self.url, url)

    def get_slim_soup(self, keep_links: bool = False):
        """
        获取一个精简版的 BeautifulSoup 对象，移除多余的 HTML 元素，保留指定的属性和链接。

        Args:
            keep_links (bool): 是否保留链接元素。

        Returns:
            BeautifulSoup: 精简后的 BeautifulSoup 对象。
        """
        soup = _get_soup(self.html)
        keep_attrs = ["class", "id"]  # 保留的属性

        if keep_links:
            keep_attrs.append("href")  # 如果需要保留链接，添加 href 属性

        # 删除不需要的属性
        for i in soup.find_all(True):
            for name in list(i.attrs):
                if i[name] and name not in keep_attrs:
                    del i[name]

        # 删除不需要的媒体元素（如图片、视频、音频等）
        for i in soup.find_all(["svg", "img", "video", "audio"]):
            i.decompose()

        return soup


def get_html_content(page: str, base: str):
    """
    获取网页的纯文本内容。

    Args:
        page (str): 网页的 HTML 内容。
        base (str): 网页的基本 URL 地址。

    Returns:
        str: 网页的纯文本内容。
    """
    soup = _get_soup(page)
    return soup.get_text(strip=True)


def _get_soup(page: str):
    """
    解析网页内容为 BeautifulSoup 对象，并移除不需要的 HTML 元素（如 style、script、head 等）。

    Args:
        page (str): 网页的 HTML 内容。

    Returns:
        BeautifulSoup: 解析后的 BeautifulSoup 对象。
    """
    soup = BeautifulSoup(page, "html.parser")
    # 移除不需要的标签（style、script、document、head、title、footer）
    for s in soup(["style", "script", "[document]", "head", "title", "footer"]):
        s.extract()

    return soup


def simplify_html(html: str, url: str, keep_links: bool = False):
    """
    将 HTML 内容精简并最小化，移除不必要的空格和注释。

    Args:
        html (str): 网页的 HTML 内容。
        url (str): 网页的 URL 地址。
        keep_links (bool): 是否保留链接元素。

    Returns:
        str: 最小化后的 HTML 内容。
    """
    html = WebPage(inner_text="", html=html, url=url).get_slim_soup(keep_links).decode()
    return htmlmin.minify(html, remove_comments=True, remove_empty_space=True)
