"""See https://github.com/web-arena-x/webarena
"""
from __future__ import annotations

import re

from playwright.async_api import BrowserContext, Page


async def get_accessibility_tree(page: Page):
    """
    获取页面的无障碍树（Accessibility Tree）。

    参数：
        page (Page): Playwright Page 对象。

    返回：
        list: 无障碍树的节点列表。
    """
    cdp_session = await get_page_cdp_session(page)
    resp = await cdp_session.send("Accessibility.getFullAXTree")

    seen_ids = set()
    accessibility_tree = []
    for node in resp["nodes"]:
        if node["nodeId"] not in seen_ids:
            accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    return accessibility_tree


async def execute_step(step: str, page: Page, browser_ctx: BrowserContext, accessibility_tree: list):
    """
    执行单个用户操作步骤，如点击、输入、滚动等。

    参数：
        step (str): 要执行的操作步骤，如 "click [1]"。
        page (Page): Playwright Page 对象。
        browser_ctx (BrowserContext): Playwright 浏览器上下文对象。
        accessibility_tree (list): 无障碍树信息。

    返回：
        Page 或 str: 执行完步骤后的页面对象，或 `stop` 操作的返回值（字符串）。
    """
    step = step.strip()
    func = step.split("[")[0].strip() if "[" in step else step.split()[0].strip()
    if func == "None":
        return ""
    elif func == "click":
        match = re.search(r"click ?\[(\d+)\]", step)
        if not match:
            raise ValueError(f"无效的 click 操作 {step}")
        element_id = match.group(1)
        await click_element(page, get_backend_node_id(element_id, accessibility_tree))
    elif func == "hover":
        match = re.search(r"hover ?\[(\d+)\]", step)
        if not match:
            raise ValueError(f"无效的 hover 操作 {step}")
        element_id = match.group(1)
        await hover_element(page, get_backend_node_id(element_id, accessibility_tree))
    elif func == "type":
        # 默认加上回车标志
        if not (step.endswith("[0]") or step.endswith("[1]")):
            step += " [1]"

        match = re.search(r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", step)
        if not match:
            raise ValueError(f"无效的 type 操作 {step}")
        element_id, text, enter_flag = match.groups()
        if enter_flag == "1":
            text += "\n"
        await click_element(page, get_backend_node_id(element_id, accessibility_tree))
        await type_text(page, text)
    elif func == "press":
        match = re.search(r"press ?\[(.+)\]", step)
        if not match:
            raise ValueError(f"无效的 press 操作 {step}")
        key = match.group(1)
        await key_press(page, key)
    elif func == "scroll":
        match = re.search(r"scroll ?\[?(up|down)\]?", step)
        if not match:
            raise ValueError(f"无效的 scroll 操作 {step}")
        direction = match.group(1)
        await scroll_page(page, direction)
    elif func == "goto":
        match = re.search(r"goto ?\[(.+)\]", step)
        if not match:
            raise ValueError(f"无效的 goto 操作 {step}")
        url = match.group(1)
        await page.goto(url)
    elif func == "new_tab":
        page = await browser_ctx.new_page()
    elif func == "go_back":
        await page.go_back()
    elif func == "go_forward":
        await page.go_forward()
    elif func == "tab_focus":
        match = re.search(r"tab_focus ?\[(\d+)\]", step)
        if not match:
            raise ValueError(f"无效的 tab_focus 操作 {step}")
        page_number = int(match.group(1))
        page = browser_ctx.pages[page_number]
        await page.bring_to_front()
    elif func == "close_tab":
        await page.close()
        page = browser_ctx.pages[-1] if browser_ctx.pages else await browser_ctx.new_page()
    elif func == "stop":
        match = re.search(r'stop\(?"(.+)?"\)', step)
        return match.group(1) if match else ""
    else:
        raise ValueError(f"未知的操作 {func}")

    await page.wait_for_load_state("domcontentloaded")
    return page


async def click_element(page: Page, backend_node_id: int):
    """
    点击指定的元素。

    参数：
        page (Page): Playwright Page 对象。
        backend_node_id (int): 元素的后端节点 ID。
    """
    cdp_session = await get_page_cdp_session(page)
    resp = await get_bounding_rect(cdp_session, backend_node_id)
    node_info = resp["result"]["value"]
    x, y = await get_element_center(node_info)

    # 滚动页面，使元素位于可视区域
    await page.evaluate(f"window.scrollTo({x} - window.innerWidth / 2, {y} - window.innerHeight / 2);")

    # 重新获取元素位置，确保准确点击
    resp = await get_bounding_rect(cdp_session, backend_node_id)
    node_info = resp["result"]["value"]
    x, y = await get_element_center(node_info)
    await page.mouse.click(x, y)


async def hover_element(page: Page, backend_node_id: int):
    """
    将鼠标悬停在指定的元素上。

    参数：
        page (Page): Playwright Page 对象。
        backend_node_id (int): 元素的后端节点 ID。
    """
    cdp_session = await get_page_cdp_session(page)
    resp = await get_bounding_rect(cdp_session, backend_node_id)
    node_info = resp["result"]["value"]
    x, y = await get_element_center(node_info)
    await page.mouse.move(x, y)


async def scroll_page(page: Page, direction: str):
    """
    滚动页面。

    参数：
        page (Page): Playwright Page 对象。
        direction (str): 滚动方向（"up" 或 "down"）。
    """
    if direction == "up":
        await page.evaluate(
            "(document.scrollingElement || document.body).scrollTop -= window.innerHeight;"
        )
    elif direction == "down":
        await page.evaluate(
            "(document.scrollingElement || document.body).scrollTop += window.innerHeight;"
        )


async def key_press(page: Page, key: str):
    """
    模拟键盘按键操作。

    参数：
        page (Page): Playwright Page 对象。
        key (str): 按键名称。
    """
    if "Meta" in key and "Mac" not in await page.evaluate("navigator.platform"):
        key = key.replace("Meta", "Control")
    await page.keyboard.press(key)


async def get_element_outer_html(page: Page, backend_node_id: int):
    cdp_session = await get_page_cdp_session(page)
    try:
        outer_html = await cdp_session.send("DOM.getOuterHTML", {"backendNodeId": int(backend_node_id)})
        return outer_html["outerHTML"]
    except Exception as e:
        raise ValueError("Element not found") from e


async def get_element_center(node_info):
    """
    计算元素的中心坐标。

    参数：
        node_info (dict): 元素的位置信息。

    返回：
        tuple: (x, y) 代表元素的中心坐标。
    """
    x, y, width, height = node_info["x"], node_info["y"], node_info["width"], node_info["height"]
    return x + width / 2, y + height / 2


def extract_step(response: str, action_splitter: str = "```") -> str:
    """
    从给定的字符串 `response` 中提取特定格式的内容。

    参数：
        response (str): 输入的字符串，需要从中提取内容。
        action_splitter (str): 作为分隔符的字符串，默认为 "```"。

    返回：
        str: 提取出的内容（去除前后空白字符）。

    异常：
        ValueError: 如果未找到符合格式的内容，则抛出异常。
    """
    # 定义正则表达式模式，匹配 `action_splitter` 之间的内容
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()  # 返回提取出的内容，并去除首尾空白
    else:
        raise ValueError(f'Cannot find the answer phrase "{response}"')


async def get_bounding_rect(cdp_session, backend_node_id: str):
    """
    通过 Chrome DevTools Protocol (CDP) 获取网页元素的边界矩形信息。

    参数：
        cdp_session: CDP 会话对象，用于与浏览器通信。
        backend_node_id (str): 目标元素的后端节点 ID。

    返回：
        dict: 元素的边界矩形信息。

    异常：
        ValueError: 如果无法找到指定的元素，则抛出异常。
    """
    try:
        # 解析节点，获取 JavaScript 访问对象 ID
        remote_object = await cdp_session.send("DOM.resolveNode", {"backendNodeId": int(backend_node_id)})
        remote_object_id = remote_object["object"]["objectId"]

        # 运行 JavaScript 代码获取元素的边界矩形
        response = await cdp_session.send(
            "Runtime.callFunctionOn",
            {
                "objectId": remote_object_id,
                "functionDeclaration": """
                    function() {
                        if (this.nodeType == 3) { // 如果是文本节点
                            var range = document.createRange();
                            range.selectNode(this);
                            var rect = range.getBoundingClientRect().toJSON();
                            range.detach(); // 释放 range
                            return rect;
                        } else { // 其他普通元素
                            return this.getBoundingClientRect().toJSON();
                        }
                    }
                """,
                "returnByValue": True,  # 直接返回 JSON 格式的结果
            },
        )
        return response
    except Exception as e:
        raise ValueError("Element not found") from e  # 如果出错，则抛出异常


# 定义需要忽略的无障碍属性（即不纳入解析的属性）
IGNORED_ACTREE_PROPERTIES = (
    "focusable", "editable", "readonly", "level",
    "settable", "multiline", "invalid"
)


def parse_accessibility_tree(accessibility_tree):
    """
    解析无障碍（Accessibility）树，并转换为可读的文本格式。

    参数：
        accessibility_tree (list): 无障碍树的数据结构，包含网页的可访问性信息。

    返回：
        tuple:
            - str: 解析后的无障碍树文本表示。
            - dict: 解析出的节点信息，包含节点 ID、后端 ID、边界信息等。
    """
    # 创建节点 ID 到索引的映射，方便查找
    node_id_to_idx = {node["nodeId"]: idx for idx, node in enumerate(accessibility_tree)}

    # 存储有用的节点信息
    obs_nodes_info = {}

    def dfs(idx: int, obs_node_id: str, depth: int) -> str:
        """
        递归遍历无障碍树，并生成格式化的文本表示。

        参数：
            idx (int): 当前遍历到的节点索引。
            obs_node_id (str): 该节点的无障碍 ID。
            depth (int): 当前递归的深度，用于控制缩进。

        返回：
            str: 该节点及其子节点的文本表示。
        """
        tree_str = ""
        node = accessibility_tree[idx]
        indent = "\t" * depth  # 计算缩进
        valid_node = True  # 标记当前节点是否有效

        try:
            # 获取节点角色和名称
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"[{obs_node_id}] {role} {repr(name)}"

            # 解析节点的属性
            properties = []
            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue  # 忽略不需要的属性
                    properties.append(f'{property["name"]}: {property["value"]["value"]}')
                except KeyError:
                    pass  # 忽略 KeyError

            # 如果有属性，则添加到节点字符串中
            if properties:
                node_str += " " + " ".join(properties)

            # 检查该节点是否有效
            if not node_str.strip():
                valid_node = False

            # 处理无名称的通用节点
            if not name.strip():
                if not properties:
                    if role in [
                        "generic", "img", "list", "strong", "paragraph",
                        "banner", "navigation", "Section", "LabelText",
                        "Legend", "listitem"
                    ]:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            # 如果该节点有效，则记录信息
            if valid_node:
                tree_str += f"{indent}{node_str}"
                obs_nodes_info[obs_node_id] = {
                    "backend_id": node["backendDOMNodeId"],  # 该节点的后端 ID
                    "union_bound": node["union_bound"],  # 该节点的边界信息
                    "text": node_str,  # 该节点的文本表示
                }

        except Exception:
            valid_node = False  # 解析出错则标记为无效

        # 遍历该节点的所有子节点
        for _, child_node_id in enumerate(node["childIds"]):
            if child_node_id not in node_id_to_idx:
                continue  # 如果子节点不在索引映射中，则跳过
            # 如果当前节点有效，则增加深度，否则保持原深度
            child_depth = depth + 1 if valid_node else depth
            child_str = dfs(node_id_to_idx[child_node_id], child_node_id, child_depth)
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str  # 递归拼接子节点字符串

        return tree_str

    # 递归解析无障碍树，从根节点开始
    tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
    return tree_str, obs_nodes_info


async def get_page_cdp_session(page):
    """
    获取页面的 CDP（Chrome DevTools Protocol）会话。

    参数：
        page (Page): Playwright Page 对象。

    返回：
        CDPSession: CDP 会话对象。
    """
    if hasattr(page, "cdp_session"):
        return page.cdp_session
    cdp_session = await page.context.new_cdp_session(page)
    page.cdp_session = cdp_session
    return cdp_session


def get_backend_node_id(element_id, accessibility_tree):
    """
    通过元素 ID 获取后端节点 ID。

    参数：
        element_id (str): 元素 ID。
        accessibility_tree (list): 无障碍树。

    返回：
        int: 后端节点 ID。
    """
    for node in accessibility_tree:
        if node["nodeId"] == str(element_id):
            return node.get("backendDOMNodeId")
    raise ValueError(f"未找到元素 {element_id}")
