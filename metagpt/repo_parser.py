#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a symbols repository from source code.

This script is designed to create a symbols repository from the provided source code.

@Time    : 2023/11/17 17:58
@Author  : alexanderwu
@File    : repo_parser.py
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from metagpt.const import AGGREGATION, COMPOSITION, GENERALIZATION
from metagpt.logs import logger
from metagpt.utils.common import any_to_str, aread, remove_white_spaces
from metagpt.utils.exceptions import handle_exception


class RepoFileInfo(BaseModel):
    """
    代表仓库中的文件信息的数据类。

    属性:
        file (str): 文件的名称或路径。
        classes (List): 文件中定义的类名列表。
        functions (List): 文件中定义的函数名列表。
        globals (List): 文件中定义的全局变量名列表。
        page_info (List): 与文件相关的页面信息列表。
    """

    file: str
    classes: List = Field(default_factory=list)
    functions: List = Field(default_factory=list)
    globals: List = Field(default_factory=list)
    page_info: List = Field(default_factory=list)


class CodeBlockInfo(BaseModel):
    """
    代表代码块信息的数据类。

    属性:
        lineno (int): 代码块的起始行号。
        end_lineno (int): 代码块的结束行号。
        type_name (str): 代码块的类型或类别。
        tokens (List): 代码块中的标记列表。
        properties (Dict): 包含代码块附加属性的字典。
    """

    lineno: int
    end_lineno: int
    type_name: str
    tokens: List = Field(default_factory=list)
    properties: Dict = Field(default_factory=dict)


class DotClassAttribute(BaseModel):
    """
    代表类属性（dot格式）信息的数据类。

    属性:
        name (str): 类属性的名称。
        type_ (str): 类属性的类型。
        default_ (str): 类属性的默认值。
        description (str): 类属性的描述信息。
        compositions (List[str]): 与该类属性相关的组合类型列表。
    """

    name: str = ""
    type_: str = ""
    default_: str = ""
    description: str
    compositions: List[str] = Field(default_factory=list)

    @classmethod
    def parse(cls, v: str) -> "DotClassAttribute":
        """
        解析 dot 格式的文本并返回 DotClassAttribute 对象。

        参数:
            v (str): 需要解析的 dot 格式字符串。

        返回:
            DotClassAttribute: 解析后的 DotClassAttribute 实例。
        """
        val = ""
        meet_colon = False  # 是否遇到 `:`，用于标识类型
        meet_equals = False  # 是否遇到 `=`，用于标识默认值

        for c in v:
            if c == ":":
                meet_colon = True  # 标记遇到了 `:`，表示类型部分开始
            elif c == "=":
                meet_equals = True  # 标记遇到了 `=`，表示默认值部分开始
                if not meet_colon:
                    val += ":"  # 如果 `:` 之前没有出现，补上 `:`
                    meet_colon = True
            val += c  # 拼接原始字符串

        if not meet_colon:
            val += ":"  # 如果 `:` 仍未出现，补上 `:`
        if not meet_equals:
            val += "="  # 如果 `=` 仍未出现，补上 `=`

        # 查找 `:` 和 `=`
        cix = val.find(":")
        eix = val.rfind("=")

        # 提取 name, type_, default_
        name = val[0:cix].strip()
        type_ = val[cix + 1: eix]
        default_ = val[eix + 1:].strip()

        # 清理 type_ 字符串
        type_ = remove_white_spaces(type_)
        if type_ == "NoneType":
            type_ = ""

        # 处理 Literal 类型
        if "Literal[" in type_:
            pre_l, literal, post_l = cls._split_literal(type_)
            composition_val = pre_l + "Literal" + post_l  # 替换 Literal[...] 为 Literal
            type_ = pre_l + literal + post_l
        else:
            type_ = re.sub(r"['\"]+", "", type_)  # 去除引号
            composition_val = type_

        if default_ == "None":
            default_ = ""

        # 解析组合类型
        compositions = cls.parse_compositions(composition_val)

        return cls(name=name, type_=type_, default_=default_, description=v, compositions=compositions)

    @staticmethod
    def parse_compositions(types_part) -> List[str]:
        """
        解析类型定义代码块，并返回提取出的组合类型列表。

        参数:
            types_part: 需要解析的类型定义代码块。

        返回:
            List[str]: 提取出的组合类型列表。
        """
        if not types_part:
            return []

        # 用 `|` 替换 `[]`, `()`, `,` 以便拆分
        modified_string = re.sub(r"[\[\],\(\)]", "|", types_part)
        types = modified_string.split("|")

        # 过滤掉基础类型，仅保留自定义类型
        filters = {
            "str",
            "frozenset",
            "set",
            "int",
            "float",
            "complex",
            "bool",
            "dict",
            "list",
            "Union",
            "Dict",
            "Set",
            "Tuple",
            "NoneType",
            "None",
            "Any",
            "Optional",
            "Iterator",
            "Literal",
            "List",
        }

        result = set()
        for t in types:
            t = re.sub(r"['\"]+", "", t.strip())  # 去除引号并去空格
            if t and t not in filters:
                result.add(t)

        return list(result)


    @staticmethod
    def _split_literal(v):
        """
        解析字面量类型定义，并返回三部分: 前缀部分、字面量部分、后缀部分。

        参数:
            v: 需要解析的字面量类型代码块。

        返回:
            Tuple[str, str, str]: 解析出的前缀部分、字面量部分和后缀部分。
        """
        tag = "Literal["  # 目标标记
        bix = v.find(tag)  # 获取 `Literal[` 起始索引
        eix = len(v) - 1  # 初始化结尾索引
        counter = 1  # 记录 `[` 的嵌套层数

        # 遍历字符串，找到 `Literal[...]` 的匹配结束位置
        for i in range(bix + len(tag), len(v) - 1):
            c = v[i]
            if c == "[":
                counter += 1
                continue
            if c == "]":
                counter -= 1
                if counter > 0:
                    continue
                eix = i  # 找到匹配的 `]`
                break

        # 切割字符串
        pre_l = v[0:bix]
        post_l = v[eix + 1:]

        # 去除前后部分的引号
        pre_l = re.sub(r"['\"]", "", pre_l)
        post_l = re.sub(r"['\"]", "", post_l)

        return pre_l, v[bix: eix + 1], post_l

    @field_validator("compositions", mode="after")
    @classmethod
    def sort(cls, lst: List) -> List:
        """
        在修改 `compositions` 或 `aggregations` 列表后自动进行排序。

        参数:
            lst (List): 需要排序的列表。

        返回:
            List: 排序后的列表。
        """
        lst.sort()
        return lst


class DotClassInfo(BaseModel):
    """
    代表 dot 格式中的类信息的仓库数据元素。

    属性:
        name (str): 类的名称。
        package (Optional[str]): 类所属的包（可选）。
        attributes (Dict[str, DotClassAttribute]): 与类相关联的属性字典。
        methods (Dict[str, DotClassMethod]): 与类相关联的方法字典。
        compositions (List[str]): 与类相关联的组合（集合）列表。
        aggregations (List[str]): 与类相关联的聚合列表。
    """

    name: str
    package: Optional[str] = None
    attributes: Dict[str, DotClassAttribute] = Field(default_factory=dict)
    methods: Dict[str, DotClassMethod] = Field(default_factory=dict)
    compositions: List[str] = Field(default_factory=list)
    aggregations: List[str] = Field(default_factory=list)

    @field_validator("compositions", "aggregations", mode="after")
    @classmethod
    def sort(cls, lst: List) -> List:
        """
        在修改后自动对列表属性进行排序。

        参数:
            lst (List): 需要排序的列表属性。

        返回:
            List: 排序后的列表。
        """
        lst.sort()
        return lst


class DotClassRelationship(BaseModel):
    """
    代表 dot 格式中的两个类之间的关系的仓库数据元素。

    属性:
        src (str): 关系的源类。
        dest (str): 关系的目标类。
        relationship (str): 关系的类型或性质。
        label (Optional[str]): 关系的可选标签。
    """

    src: str = ""
    dest: str = ""
    relationship: str = ""
    label: Optional[str] = None


class DotReturn(BaseModel):
    """
    代表 dot 格式中的函数或方法返回类型的仓库数据元素。

    属性:
        type_ (str): 返回类型。
        description (str): 返回类型的描述。
        compositions (List[str]): 与返回类型相关联的组合（集合）列表。
    """

    type_: str = ""
    description: str
    compositions: List[str] = Field(default_factory=list)

    @classmethod
    def parse(cls, v: str) -> "DotReturn" | None:
        """
        解析 dot 格式文本中的返回类型部分，并返回一个 DotReturn 对象。

        参数:
            v (str): 包含要解析的返回类型部分的 dot 格式文本。

        返回:
            DotReturn | None: 一个表示解析后的返回类型的 DotReturn 实例，
                              如果解析失败则返回 None。
        """
        if not v:
            return DotReturn(description=v)
        type_ = remove_white_spaces(v)
        compositions = DotClassAttribute.parse_compositions(type_)
        return cls(type_=type_, description=v, compositions=compositions)

    @field_validator("compositions", mode="after")
    @classmethod
    def sort(cls, lst: List) -> List:
        """
        在修改后自动对列表属性进行排序。

        参数:
            lst (List): 需要排序的列表属性。

        返回:
            List: 排序后的列表。
        """
        lst.sort()
        return lst


class DotClassMethod(BaseModel):
    """
    代表 dot 格式中的方法信息的仓库数据元素。

    属性:
        name (str): 方法的名称。
        args (List[DotClassAttribute]): 方法的参数列表，每个参数为 DotClassAttribute 类型。
        return_args (Optional[DotReturn]): 方法的返回类型，使用 DotReturn 类型表示。
        description (str): 方法的描述信息。
        aggregations (List[str]): 与方法相关联的聚合列表。
    """
    name: str
    args: List[DotClassAttribute] = Field(default_factory=list)
    return_args: Optional[DotReturn] = None
    description: str
    aggregations: List[str] = Field(default_factory=list)

    @classmethod
    def parse(cls, v: str) -> "DotClassMethod":
        """
        解析 dot 格式方法文本并返回一个 DotClassMethod 对象。

        参数:
            v (str): 包含方法信息的 dot 格式文本。

        返回:
            DotClassMethod: 代表解析后的方法的 DotClassMethod 实例。
        """
        bix = v.find("(")
        eix = v.rfind(")")
        rix = v.rfind(":")
        if rix < 0 or rix < eix:
            rix = eix
        name_part = v[0:bix].strip()
        args_part = v[bix + 1 : eix].strip()
        return_args_part = v[rix + 1 :].strip()

        name = cls._parse_name(name_part)
        args = cls._parse_args(args_part)
        return_args = DotReturn.parse(return_args_part)
        aggregations = set()
        for i in args:
            aggregations.update(set(i.compositions))
        aggregations.update(set(return_args.compositions))

        return cls(name=name, args=args, description=v, return_args=return_args, aggregations=list(aggregations))

    @staticmethod
    def _parse_name(v: str) -> str:
        """
        解析 dot 格式方法名称部分并返回方法名称。

        参数:
            v (str): 包含方法名称部分的 dot 格式文本。

        返回:
            str: 解析后的方法名称。
        """
        tags = [">", "</"]
        if tags[0] in v:
            bix = v.find(tags[0]) + len(tags[0])
            eix = v.rfind(tags[1])
            return v[bix:eix].strip()
        return v.strip()

    @staticmethod
    def _parse_args(v: str) -> List[DotClassAttribute]:
        """
        解析 dot 格式方法参数部分并返回解析后的参数列表。

        参数:
            v (str): 包含方法参数部分的 dot 格式文本。

        返回:
            List[DotClassAttribute]: 解析后的方法参数列表，每个参数为 DotClassAttribute 类型。
        """
        if not v:
            return []
        parts = []
        bix = 0
        counter = 0
        for i in range(0, len(v)):
            c = v[i]
            if c == "[":
                counter += 1
                continue
            elif c == "]":
                counter -= 1
                continue
            elif c == "," and counter == 0:
                parts.append(v[bix:i].strip())
                bix = i + 1
        parts.append(v[bix:].strip())

        attrs = []
        for p in parts:
            if p:
                attr = DotClassAttribute.parse(p)
                attrs.append(attr)
        return attrs


class RepoParser(BaseModel):
    """
    工具类，用于从项目目录构建符号仓库。

    属性:
        base_directory (Path): 项目目录的基础路径。
    """

    base_directory: Path = Field(default=None)

    @classmethod
    @handle_exception(exception_type=Exception, default_return=[])
    def _parse_file(cls, file_path: Path) -> list:
        """
        解析仓库中的 Python 文件。

        参数:
            file_path (Path): 需要解析的 Python 文件路径。

        返回:
            list: 包含解析后符号信息的列表。
        """
        return ast.parse(file_path.read_text()).body

    def extract_class_and_function_info(self, tree, file_path) -> RepoFileInfo:
        """
        从抽象语法树（AST）中提取类、函数和全局变量的信息。

        参数:
            tree: Python 文件的抽象语法树（AST）。
            file_path: Python 文件的路径。

        返回:
            RepoFileInfo: 包含提取信息的 RepoFileInfo 对象。
        """
        file_info = RepoFileInfo(file=str(file_path.relative_to(self.base_directory)))
        for node in tree:
            info = RepoParser.node_to_str(node)
            if info:
                file_info.page_info.append(info)
            if isinstance(node, ast.ClassDef):
                class_methods = [m.name for m in node.body if is_func(m)]
                file_info.classes.append({"name": node.name, "methods": class_methods})
            elif is_func(node):
                file_info.functions.append(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(target, ast.Name):
                        file_info.globals.append(target.id)
        return file_info

    def generate_symbols(self) -> List[RepoFileInfo]:
        """
        从项目目录中的 `.py` 和 `.js` 文件构建符号仓库。

        返回:
            List[RepoFileInfo]: 包含提取信息的 RepoFileInfo 对象的列表。
        """
        files_classes = []
        directory = self.base_directory

        matching_files = []
        extensions = ["*.py"]
        for ext in extensions:
            matching_files += directory.rglob(ext)
        for path in matching_files:
            tree = self._parse_file(path)
            file_info = self.extract_class_and_function_info(tree, path)
            files_classes.append(file_info)

        return files_classes

    def generate_json_structure(self, output_path: Path):
        """
        生成一个 JSON 文件，记录仓库结构。

        参数:
            output_path (Path): 生成的 JSON 文件路径。
        """
        files_classes = [i.model_dump() for i in self.generate_symbols()]
        output_path.write_text(json.dumps(files_classes, indent=4))

    def generate_dataframe_structure(self, output_path: Path):
        """
        生成一个 DataFrame，记录仓库结构，并保存为 CSV 文件。

        参数:
            output_path (Path): 生成的 CSV 文件路径。
        """
        files_classes = [i.model_dump() for i in self.generate_symbols()]
        df = pd.DataFrame(files_classes)
        df.to_csv(output_path, index=False)

    def generate_structure(self, output_path: str | Path = None, mode="json") -> Path:
        """
        以指定格式生成仓库结构。

        参数:
            output_path (str | Path): 输出文件或目录的路径。默认为 None。
            mode (str): 输出格式模式。选项: "json"（默认）、"csv" 等。

        返回:
            Path: 生成的输出文件或目录的路径。
        """
        output_file = self.base_directory / f"{self.base_directory.name}-structure.{mode}"
        output_path = Path(output_path) if output_path else output_file

        if mode == "json":
            self.generate_json_structure(output_path)
        elif mode == "csv":
            self.generate_dataframe_structure(output_path)
        return output_path

    @staticmethod
    def node_to_str(node) -> CodeBlockInfo | None:
        """
        解析并将抽象语法树（AST）节点转换为 CodeBlockInfo 对象。

        参数:
            node: 要转换的 AST 节点。

        返回:
            CodeBlockInfo | None: 表示解析后的 AST 节点的 CodeBlockInfo 对象，
                                  如果转换失败，则返回 None。
        """
        if isinstance(node, ast.Try):
            return None
        # 如果节点是表达式类型，则解析并返回对应的 CodeBlockInfo
        if any_to_str(node) == any_to_str(ast.Expr):
            return CodeBlockInfo(
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                type_name=any_to_str(node),
                tokens=RepoParser._parse_expr(node),
            )
        # 定义各种 AST 节点类型及其对应的解析方法
        mappings = {
            any_to_str(ast.Import): lambda x: [RepoParser._parse_name(n) for n in x.names],
            any_to_str(ast.Assign): RepoParser._parse_assign,
            any_to_str(ast.ClassDef): lambda x: x.name,
            any_to_str(ast.FunctionDef): lambda x: x.name,
            any_to_str(ast.ImportFrom): lambda x: {
                "module": x.module,
                "names": [RepoParser._parse_name(n) for n in x.names],
            },
            any_to_str(ast.If): RepoParser._parse_if,
            any_to_str(ast.AsyncFunctionDef): lambda x: x.name,
            any_to_str(ast.AnnAssign): lambda x: RepoParser._parse_variable(x.target),
        }
        func = mappings.get(any_to_str(node))
        if func:
            # 创建 CodeBlockInfo 对象，并根据类型调用对应的解析函数
            code_block = CodeBlockInfo(lineno=node.lineno, end_lineno=node.end_lineno, type_name=any_to_str(node))
            val = func(node)
            if isinstance(val, dict):
                code_block.properties = val
            elif isinstance(val, list):
                code_block.tokens = val
            elif isinstance(val, str):
                code_block.tokens = [val]
            else:
                raise NotImplementedError(f"Not implement:{val}")
            return code_block
        # 如果不支持该节点类型，则发出警告并返回 None
        logger.warning(f"Unsupported code block:{node.lineno}, {node.end_lineno}, {any_to_str(node)}")
        return None

    @staticmethod
    def _parse_expr(node) -> List:
        """
        解析表达式类型的抽象语法树（AST）节点。

        参数:
            node: 表示表达式的 AST 节点。

        返回:
            List: 包含从表达式节点解析得到的信息的列表。
        """
        # 定义不同表达式类型及其解析方法
        funcs = {
            any_to_str(ast.Constant): lambda x: [any_to_str(x.value), RepoParser._parse_variable(x.value)],
            any_to_str(ast.Call): lambda x: [any_to_str(x.value), RepoParser._parse_variable(x.value.func)],
            any_to_str(ast.Tuple): lambda x: [any_to_str(x.value), RepoParser._parse_variable(x.value)],
        }
        func = funcs.get(any_to_str(node.value))
        if func:
            return func(node)
        # 如果没有对应的解析方法，则抛出未实现异常
        raise NotImplementedError(f"Not implement: {node.value}")

    @staticmethod
    def _parse_name(n):
        """
        获取抽象语法树（AST）节点的 'name' 值。

        参数:
            n: AST 节点。

        返回:
            'name' 值，表示该节点的名称。
        """
        if n.asname:
            return f"{n.name} as {n.asname}"
        return n.name

    @staticmethod
    def _parse_if(n):
        """
        解析 'if' 语句的抽象语法树（AST）节点。

        参数:
            n: 表示 'if' 语句的 AST 节点。

        返回:
            None 或者从 'if' 语句节点解析出的信息。
        """
        tokens = []
        try:
            # 如果 'if' 条件是布尔运算符（BoolOp），解析布尔值
            if isinstance(n.test, ast.BoolOp):
                tokens = []
                for v in n.test.values:
                    tokens.extend(RepoParser._parse_if_compare(v))
                return tokens
            # 如果 'if' 条件是比较运算（Compare），解析左侧的变量
            if isinstance(n.test, ast.Compare):
                v = RepoParser._parse_variable(n.test.left)
                if v:
                    tokens.append(v)
            # 如果 'if' 条件是变量名（Name），解析该变量
            if isinstance(n.test, ast.Name):
                v = RepoParser._parse_variable(n.test)
                tokens.append(v)
            # 如果 'if' 条件有比较器（comparators），解析比较器中的变量
            if hasattr(n.test, "comparators"):
                for item in n.test.comparators:
                    v = RepoParser._parse_variable(item)
                    if v:
                        tokens.append(v)
            return tokens
        except Exception as e:
            logger.warning(f"Unsupported if: {n}, err:{e}")
        return tokens

    @staticmethod
    def _parse_if_compare(n):
        """
        解析 'if' 条件的抽象语法树（AST）节点。

        参数:
            n: 表示 'if' 条件的 AST 节点。

        返回:
            None 或者从 'if' 条件节点解析出的信息。
        """
        if hasattr(n, "left"):
            return RepoParser._parse_variable(n.left)
        else:
            return []

    @staticmethod
    def _parse_variable(node):
        """
        解析变量的抽象语法树（AST）节点。

        参数：
            node：表示变量的AST节点。

        返回：
            None 或者 解析后的变量信息。
        """
        try:
            funcs = {
                any_to_str(ast.Constant): lambda x: x.value,  # 常量节点返回值
                any_to_str(ast.Name): lambda x: x.id,  # 名称节点返回id
                any_to_str(ast.Attribute): lambda x: f"{x.value.id}.{x.attr}"  # 属性节点返回格式化的值
                if hasattr(x.value, "id")
                else f"{x.attr}",
                any_to_str(ast.Call): lambda x: RepoParser._parse_variable(x.func),  # 调用节点解析函数
                any_to_str(ast.Tuple): lambda x: [d.value for d in x.dims],  # 元组节点返回维度值
            }
            func = funcs.get(any_to_str(node))  # 获取对应的解析函数
            if not func:
                raise NotImplementedError(f"未实现:{node}")  # 如果没有实现解析，抛出异常
            return func(node)
        except Exception as e:
            logger.warning(f"不支持的变量:{node}, 错误:{e}")

    @staticmethod
    def _parse_assign(node):
        """
        解析赋值的抽象语法树（AST）节点。

        参数：
            node：表示赋值的AST节点。

        返回：
            None 或者 解析后的赋值信息。
        """
        return [RepoParser._parse_variable(t) for t in node.targets]  # 解析赋值目标

    async def rebuild_class_views(self, path: str | Path = None):
        """
        执行`pylint`重新构建点格式的类视图仓库文件。

        参数：
            path (str | Path): 目标目录或文件的路径，默认值为 None。
        """
        if not path:
            path = self.base_directory  # 如果没有指定路径，使用默认路径
        path = Path(path)
        if not path.exists():
            return  # 如果路径不存在，返回
        init_file = path / "__init__.py"  # 检查 __init__.py 文件是否存在
        if not init_file.exists():
            raise ValueError("无法导入模块 __init__，错误：没有 __init__ 模块。")
        command = f"pyreverse {str(path)} -o dot"  # 使用 pyreverse 生成 dot 格式文件
        output_dir = path / "__dot__"
        output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
        result = subprocess.run(command, shell=True, check=True, cwd=str(output_dir))  # 执行命令
        if result.returncode != 0:
            raise ValueError(f"{result}")  # 如果执行失败，抛出异常
        class_view_pathname = output_dir / "classes.dot"
        class_views = await self._parse_classes(class_view_pathname)  # 解析类视图
        relationship_views = await self._parse_class_relationships(class_view_pathname)  # 解析类关系
        packages_pathname = output_dir / "packages.dot"
        class_views, relationship_views, package_root = RepoParser._repair_namespaces(
            class_views=class_views, relationship_views=relationship_views, path=path
        )  # 修复命名空间
        class_view_pathname.unlink(missing_ok=True)  # 删除临时文件
        packages_pathname.unlink(missing_ok=True)
        return class_views, relationship_views, package_root

    @staticmethod
    async def _parse_classes(class_view_pathname: Path) -> List[DotClassInfo]:
        """
        解析点格式的类视图仓库文件。

        参数：
            class_view_pathname (Path): 点格式类视图仓库文件的路径。

        返回：
            List[DotClassInfo]: 解析后的 DotClassInfo 对象列表。
        """
        class_views = []
        if not class_view_pathname.exists():
            return class_views  # 如果文件不存在，返回空列表
        data = await aread(filename=class_view_pathname, encoding="utf-8")  # 读取文件内容
        lines = data.split("\n")  # 按行分割数据
        for line in lines:
            package_name, info = RepoParser._split_class_line(line)  # 解析类信息
            if not package_name:
                continue
            class_name, members, functions = re.split(r"(?<!\\)\|", info)  # 分割类成员和方法
            class_info = DotClassInfo(name=class_name)
            class_info.package = package_name
            for m in members.split("\n"):  # 解析类成员
                if not m:
                    continue
                attr = DotClassAttribute.parse(m)
                class_info.attributes[attr.name] = attr
                for i in attr.compositions:
                    if i not in class_info.compositions:
                        class_info.compositions.append(i)
            for f in functions.split("\n"):  # 解析类方法
                if not f:
                    continue
                method = DotClassMethod.parse(f)
                class_info.methods[method.name] = method
                for i in method.aggregations:
                    if i not in class_info.compositions and i not in class_info.aggregations:
                        class_info.aggregations.append(i)
            class_views.append(class_info)  # 添加解析的类视图
        return class_views

    @staticmethod
    async def _parse_class_relationships(class_view_pathname: Path) -> List[DotClassRelationship]:
        """
        解析点格式的类关系仓库文件。

        参数：
            class_view_pathname (Path): 点格式类关系仓库文件的路径。

        返回：
            List[DotClassRelationship]: 解析后的 DotClassRelationship 对象列表。
        """
        relationship_views = []
        if not class_view_pathname.exists():
            return relationship_views  # 如果文件不存在，返回空列表
        data = await aread(filename=class_view_pathname, encoding="utf-8")  # 读取文件内容
        lines = data.split("\n")  # 按行分割数据
        for line in lines:
            relationship = RepoParser._split_relationship_line(line)  # 解析类关系
            if not relationship:
                continue
            relationship_views.append(relationship)  # 添加解析的类关系
        return relationship_views

    @staticmethod
    def _split_class_line(line: str) -> (str, str):
        """
        解析一个 dot 格式的类信息行，并返回类名部分和类成员部分。

        参数:
            line (str): 包含类信息的 dot 格式行。

        返回:
            Tuple[str, str]: 返回一个包含类名部分和类成员部分的元组。
        """
        part_splitor = '" ['  # 定义分隔符
        if part_splitor not in line:
            return None, None
        ix = line.find(part_splitor)  # 查找分隔符的位置
        class_name = line[0:ix].replace('"', "")  # 获取类名并去除引号
        left = line[ix:]
        begin_flag = "label=<{"
        end_flag = "}>"
        if begin_flag not in left or end_flag not in left:
            return None, None
        bix = left.find(begin_flag)  # 查找标签开始的位置
        eix = left.rfind(end_flag)  # 查找标签结束的位置
        info = left[bix + len(begin_flag): eix]  # 获取类成员信息
        info = re.sub(r"<br[^>]*>", "\n", info)  # 将<br>标签替换为换行符
        return class_name, info

    @staticmethod
    def _split_relationship_line(line: str) -> DotClassRelationship:
        """
        解析一个 dot 格式的类关系行，并返回关系类型（Generalize，Composite，或 Aggregate）。

        参数:
            line (str): 包含类关系信息的 dot 格式行。

        返回:
            DotClassRelationship: 返回表示关系类型的 DotClassRelationship 对象。
        """
        splitters = [" -> ", " [", "];"]  # 定义分隔符
        idxs = []
        for tag in splitters:
            if tag not in line:
                return None
            idxs.append(line.find(tag))  # 查找各个分隔符的位置
        ret = DotClassRelationship()  # 创建一个关系对象
        ret.src = line[0: idxs[0]].strip('"')  # 获取源类
        ret.dest = line[idxs[0] + len(splitters[0]): idxs[1]].strip('"')  # 获取目标类
        properties = line[idxs[1] + len(splitters[1]): idxs[2]].strip(" ")  # 获取关系的属性
        mappings = {
            'arrowhead="empty"': GENERALIZATION,  # 一般化关系
            'arrowhead="diamond"': COMPOSITION,  # 组合关系
            'arrowhead="odiamond"': AGGREGATION,  # 聚合关系
        }
        for k, v in mappings.items():
            if k in properties:  # 根据属性确定关系类型
                ret.relationship = v
                if v != GENERALIZATION:
                    ret.label = RepoParser._get_label(properties)  # 获取标签
                break
        return ret

    @staticmethod
    def _get_label(line: str) -> str:
        """
        解析 dot 格式行并返回标签信息。

        参数:
            line (str): 包含标签信息的 dot 格式行。

        返回:
            str: 从行中解析出的标签信息。
        """
        tag = 'label="'
        if tag not in line:
            return ""
        ix = line.find(tag)  # 查找标签的起始位置
        eix = line.find('"', ix + len(tag))  # 查找标签的结束位置
        return line[ix + len(tag): eix]

    @staticmethod
    def _create_path_mapping(path: str | Path) -> Dict[str, str]:
        """
        创建源代码文件路径与模块名称之间的映射表。

        参数:
            path (str | Path): 源代码文件或目录的路径。

        返回:
            Dict[str, str]: 一个字典，将源代码文件路径映射到对应的模块名称。
        """
        mappings = {
            str(path).replace("/", "."): str(path),  # 将路径中的“/”替换为“.”
        }
        files = []
        try:
            directory_path = Path(path)
            if not directory_path.exists():
                return mappings
            for file_path in directory_path.iterdir():
                if file_path.is_file():
                    files.append(str(file_path))
                else:
                    subfolder_files = RepoParser._create_path_mapping(path=file_path)  # 递归处理子目录
                    mappings.update(subfolder_files)
        except Exception as e:
            logger.error(f"Error: {e}")
        for f in files:
            mappings[str(Path(f).with_suffix("")).replace("/", ".")] = str(f)  # 将文件路径添加到映射中

        return mappings

    @staticmethod
    def _repair_namespaces(
            class_views: List[DotClassInfo], relationship_views: List[DotClassRelationship], path: str | Path
    ) -> (List[DotClassInfo], List[DotClassRelationship], str):
        """
        增强命名空间，将类和关系中的路径前缀补充到类和关系中。

        参数:
            class_views (List[DotClassInfo]): 代表类视图的 DotClassInfo 对象列表。
            relationship_views (List[DotClassRelationship]): 代表类关系的 DotClassRelationship 对象列表。
            path (str | Path): 源代码文件或目录的路径。

        返回:
            Tuple[List[DotClassInfo], List[DotClassRelationship], str]: 返回增强后的类视图、关系和包的根路径。
        """
        if not class_views:
            return [], [], ""
        c = class_views[0]
        full_key = str(path).lstrip("/").replace("/", ".")
        root_namespace = RepoParser._find_root(full_key, c.package)  # 查找根命名空间
        root_path = root_namespace.replace(".", "/")  # 获取根路径

        mappings = RepoParser._create_path_mapping(path=path)  # 获取路径映射
        new_mappings = {}
        ix_root_namespace = len(root_namespace)
        ix_root_path = len(root_path)
        for k, v in mappings.items():
            nk = k[ix_root_namespace:]  # 去掉根命名空间的部分
            nv = v[ix_root_path:]  # 去掉根路径的部分
            new_mappings[nk] = nv  # 更新映射

        for c in class_views:
            c.package = RepoParser._repair_ns(c.package, new_mappings)  # 修复命名空间
        for _, v in enumerate(relationship_views):
            v.src = RepoParser._repair_ns(v.src, new_mappings)  # 修复源类命名空间
            v.dest = RepoParser._repair_ns(v.dest, new_mappings)  # 修复目标类命名空间
        return class_views, relationship_views, str(path)[: len(root_path)]

    @staticmethod
    def _repair_ns(package: str, mappings: Dict[str, str]) -> str:
        """
        将包前缀替换为命名空间前缀。

        参数:
            package (str): 需要修复的包名。
            mappings (Dict[str, str]): 一个字典，映射源代码文件路径到其对应的包名。

        返回:
            str: 修复后的命名空间。
        """
        file_ns = package
        ix = 0
        while file_ns != "":
            if file_ns not in mappings:
                ix = file_ns.rfind(".")
                file_ns = file_ns[0:ix]
                continue
            break
        if file_ns == "":
            return ""
        internal_ns = package[ix + 1:]
        ns = mappings[file_ns] + ":" + internal_ns.replace(".", ":")
        return ns

    @staticmethod
    def _find_root(full_key: str, package: str) -> str:
        """
        根据键（即完整路径）和包信息，返回包的根路径。

        参数:
            full_key (str): 代表完整路径的键。
            package (str): 包信息。

        返回:
            str: 包的根路径。
        """
        left = full_key
        while left != "":
            if left in package:
                break
            if "." not in left:
                break
            ix = left.find(".")
            left = left[ix + 1:]
        ix = full_key.rfind(left)
        return "." + full_key[0:ix]


def is_func(node) -> bool:
    """
    如果给定的节点表示一个函数，返回 True。

    参数:
        node: 抽象语法树（AST）节点。

    返回:
        bool: 如果节点表示一个函数，返回 True；否则返回 False。
    """
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
