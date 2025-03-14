from __future__ import annotations

from typing import Union

import libcst as cst
from libcst._nodes.module import Module

DocstringNode = Union[cst.Module, cst.ClassDef, cst.FunctionDef]


def get_docstring_statement(body: DocstringNode) -> cst.SimpleStatementLine:
    """从节点的主体中提取文档字符串。

    参数:
        body: 节点的主体。

    返回:
        如果存在文档字符串，返回文档字符串语句；否则返回 None。
    """
    if isinstance(body, cst.Module):
        body = body.body  # 如果是模块，获取其主体
    else:
        body = body.body.body  # 获取其他类型节点的主体

    if not body:
        return

    statement = body[0]
    if not isinstance(statement, cst.SimpleStatementLine):
        return

    expr = statement
    while isinstance(expr, (cst.BaseSuite, cst.SimpleStatementLine)):
        if len(expr.body) == 0:
            return None
        expr = expr.body[0]

    if not isinstance(expr, cst.Expr):
        return None

    val = expr.value
    if not isinstance(val, (cst.SimpleString, cst.ConcatenatedString)):
        return None

    evaluated_value = val.evaluated_value
    if isinstance(evaluated_value, bytes):
        return None

    return statement


def has_decorator(node: DocstringNode, name: str) -> bool:
    """检查节点是否具有指定的装饰器。

    参数:
        node: 要检查的节点。
        name: 装饰器名称。

    返回:
        如果节点具有指定的装饰器，返回 True；否则返回 False。
    """
    return hasattr(node, "decorators") and any(
        (hasattr(i.decorator, "value") and i.decorator.value == name)
        or (hasattr(i.decorator, "func") and hasattr(i.decorator.func, "value") and i.decorator.func.value == name)
        for i in node.decorators
    )


class DocstringCollector(cst.CSTVisitor):
    """一个用于收集 CST（抽象语法树）中所有文档字符串的访问者类。

    属性:
        stack: 用于跟踪当前 CST 路径的列表。
        docstrings: 一个字典，将 CST 中的路径映射到相应的文档字符串。
    """

    def __init__(self):
        self.stack: list[str] = []  # 用于存储路径信息
        self.docstrings: dict[tuple[str, ...], cst.SimpleStatementLine] = {}  # 存储路径与文档字符串的映射

    def visit_Module(self, node: cst.Module) -> bool | None:
        self.stack.append("")  # 处理模块节点时，路径为 ""

    def leave_Module(self, node: cst.Module) -> None:
        return self._leave(node)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)  # 处理类定义时，将类名加入路径

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        return self._leave(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)  # 处理函数定义时，将函数名加入路径

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        return self._leave(node)

    def _leave(self, node: DocstringNode) -> None:
        key = tuple(self.stack)  # 获取当前路径的元组形式
        self.stack.pop()
        if has_decorator(node, "overload"):  # 如果节点有 overload 装饰器，跳过处理
            return

        statement = get_docstring_statement(node)  # 获取文档字符串语句
        if statement:
            self.docstrings[key] = statement  # 将文档字符串与路径映射


class DocstringTransformer(cst.CSTTransformer):
    """一个用于在 CST 中替换文档字符串的转换器类。

    属性:
        stack: 用于跟踪当前路径的列表。
        docstrings: 一个字典，将 CST 中的路径映射到相应的文档字符串。
    """

    def __init__(
        self,
        docstrings: dict[tuple[str, ...], cst.SimpleStatementLine],
    ):
        self.stack: list[str] = []  # 用于存储路径信息
        self.docstrings = docstrings  # 存储路径与文档字符串的映射

    def visit_Module(self, node: cst.Module) -> bool | None:
        self.stack.append("")  # 处理模块节点时，路径为 ""

    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        return self._leave(original_node, updated_node)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)  # 处理类定义时，将类名加入路径

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        return self._leave(original_node, updated_node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)  # 处理函数定义时，将函数名加入路径

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        return self._leave(original_node, updated_node)

    def _leave(self, original_node: DocstringNode, updated_node: DocstringNode) -> DocstringNode:
        key = tuple(self.stack)  # 获取当前路径的元组形式
        self.stack.pop()

        if has_decorator(updated_node, "overload"):  # 如果节点有 overload 装饰器，跳过处理
            return updated_node

        statement = self.docstrings.get(key)  # 获取对应路径的文档字符串
        if not statement:
            return updated_node

        original_statement = get_docstring_statement(original_node)

        # 如果是模块节点，替换模块的文档字符串
        if isinstance(updated_node, cst.Module):
            body = updated_node.body
            if original_statement:
                return updated_node.with_changes(body=(statement, *body[1:]))  # 替换文档字符串
            else:
                updated_node = updated_node.with_changes(body=(statement, cst.EmptyLine(), *body))
                return updated_node

        # 如果是类或函数节点，替换其文档字符串
        body = updated_node.body.body[1:] if original_statement else updated_node.body.body
        return updated_node.with_changes(body=updated_node.body.with_changes(body=(statement, *body)))


def merge_docstring(code: str, documented_code: str) -> str:
    """将文档化代码中的文档字符串合并到原始代码中。

    参数:
        code: 原始代码。
        documented_code: 文档化的代码。

    返回:
        合并后的代码，包含来自文档化代码的文档字符串。
    """
    code_tree = cst.parse_module(code)  # 解析原始代码
    documented_code_tree = cst.parse_module(documented_code)  # 解析文档化代码

    visitor = DocstringCollector()  # 创建文档字符串收集器
    documented_code_tree.visit(visitor)  # 收集文档字符串
    transformer = DocstringTransformer(visitor.docstrings)  # 创建文档字符串转换器
    modified_tree = code_tree.visit(transformer)  # 转换原始代码
    return modified_tree.code  # 返回修改后的代码
