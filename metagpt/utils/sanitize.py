"""
@Time    : 2024/7/24 16:37
@Author  : didi
@File    : utils.py
@Acknowledgement https://github.com/evalplus/evalplus/blob/master/evalplus/sanitize.py
"""

import ast
import traceback
from enum import Enum
from typing import Dict, Generator, List, Optional, Set, Tuple

import tree_sitter_python
from tree_sitter import Language, Node, Parser

class NodeType(Enum):
    """定义代码中不同节点的类型"""
    CLASS = "class_definition"  # 类定义
    FUNCTION = "function_definition"  # 函数定义
    IMPORT = ["import_statement", "import_from_statement"]  # 导入语句
    IDENTIFIER = "identifier"  # 标识符
    ATTRIBUTE = "attribute"  # 属性
    RETURN = "return_statement"  # 返回语句
    EXPRESSION = "expression_statement"  # 表达式语句
    ASSIGNMENT = "assignment"  # 赋值语句


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    """
    遍历从给定节点开始的树结构。

    :param node: 起始遍历的根节点。
    :return: 一个生成器，逐个返回树中的节点。
    """
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def syntax_check(code, verbose=False):
    """
    检查代码的语法是否正确。

    :param code: 要检查的代码字符串。
    :param verbose: 是否打印详细的异常信息。
    :return: 如果语法正确则返回 True，否则返回 False。
    """
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    """
    提取代码中最长的有效语法块。

    :param text: 输入的代码字符串。
    :return: 提取出的最长有效代码块。
    """
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_definition_name(node: Node) -> str:
    """
    获取节点定义的名称（如类或函数名称）。

    :param node: 代码节点。
    :return: 定义的名称（字符串）。
    """
    for child in node.children:
        if child.type == NodeType.IDENTIFIER.value:
            return child.text.decode("utf8")


def has_return_statement(node: Node) -> bool:
    """
    检查给定节点是否包含返回语句。

    :param node: 代码节点。
    :return: 如果包含返回语句则返回 True，否则返回 False。
    """
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == NodeType.RETURN.value:
            return True
    return False


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    """
    获取节点之间的依赖关系。

    :param nodes: 节点名称与节点对象的元组列表。
    :return: 一个字典，键为节点名称，值为该节点的依赖节点名称集合。
    """
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == NodeType.IDENTIFIER.value:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    """
    获取某个入口点函数的依赖关系。

    :param entrypoint: 入口点函数的名称。
    :param call_graph: 函数调用图，键为函数名称，值为该函数的调用节点。
    :return: 一个集合，包含所有与入口点相关的函数名称。
    """
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    """
    清理并提取给定Python代码的相关部分。
    此函数解析输入代码，提取导入语句、类和函数定义以及变量赋值。如果提供了入口点，则仅包括入口点可达的定义部分。

    :param code: 输入的Python代码。
    :param entrypoint: 可选的入口点函数名称，用于依赖关系分析。
    :return: 清理后的代码字符串，只包含相关部分。
    """
    code = code_extract(code)
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in NodeType.IMPORT.value:
            import_nodes.append(child)
        elif child.type == NodeType.CLASS.value:
            name = get_definition_name(child)
            if not (name in class_names or name in variable_names or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == NodeType.FUNCTION.value:
            name = get_definition_name(child)
            if not (name in function_names or name in variable_names or name in class_names) and has_return_statement(
                child
            ):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif child.type == NodeType.EXPRESSION.value and child.children[0].type == NodeType.ASSIGNMENT.value:
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (name in variable_names or name in function_names or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and name not in reacheable:
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    return sanitized_output[:-1].decode("utf8")
