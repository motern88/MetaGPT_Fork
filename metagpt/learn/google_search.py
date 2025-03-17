from metagpt.tools.search_engine import SearchEngine


async def google_search(query: str, max_results: int = 6, **kwargs):
    """执行网页搜索并获取搜索结果。

    :param query: 搜索关键词。
    :param max_results: 要获取的搜索结果数量，默认为 6。
    :return: 以 Markdown 格式返回网页搜索结果。
    """
    results = await SearchEngine(**kwargs).run(query, max_results=max_results, as_string=False)
    return "\n".join(f"{i}. [{j['title']}]({j['link']}): {j['snippet']}" for i, j in enumerate(results, 1))