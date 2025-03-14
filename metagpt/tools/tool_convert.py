import ast
import inspect

from metagpt.utils.parse_docstring import GoogleDocstringParser, remove_spaces

PARSER = GoogleDocstringParser  # 使用 GoogleDocstringParser 解析文档字符串

def convert_code_to_tool_schema(obj, include: list[str] = None) -> dict:
    """通过检查对象（函数或类）将其转换为工具模式"""
    docstring = inspect.getdoc(obj)  # 获取对象的文档字符串

    if inspect.isclass(obj):
        # 如果对象是类，生成类模式
        schema = {"type": "class", "description": remove_spaces(docstring), "methods": {}}
        for name, method in inspect.getmembers(obj, inspect.isfunction):  # 遍历类的方法
            if name.startswith("_") and name != "__init__":  # 跳过私有方法
                continue
            if include and name not in include:  # 如果指定了方法名的列表且当前方法不在列表中，跳过
                continue
            method_doc = get_class_method_docstring(obj, name)  # 获取方法的文档字符串
            schema["methods"][name] = function_docstring_to_schema(method, method_doc)  # 转换为工具模式

    elif inspect.isfunction(obj):
        # 如果对象是函数，直接转换为工具模式
        schema = function_docstring_to_schema(obj, docstring)

    return schema  # 返回转换后的模式


def convert_code_to_tool_schema_ast(code: str) -> list[dict]:
    """通过解析代码的抽象语法树（AST）将代码字符串转换为工具模式列表"""

    visitor = CodeVisitor(code)  # 创建 AST 访问器
    parsed_code = ast.parse(code)  # 解析代码为 AST
    visitor.visit(parsed_code)  # 访问 AST 节点

    return visitor.get_tool_schemas()  # 返回工具模式列表


def function_docstring_to_schema(fn_obj, docstring="") -> dict:
    """
    将函数的文档字符串转换为模式字典。

    参数：
        fn_obj: 函数对象。
        docstring: 函数的文档字符串。

    返回：
        一个表示函数文档字符串的模式字典。字典包含以下键：
        - 'type': 函数的类型（'function' 或 'async_function'）。
        - 'description': 文档字符串中描述函数的第一部分，用于 LLM 的推荐和使用。
        - 'signature': 函数签名，帮助 LLM 理解如何调用函数。
        - 'parameters': 文档字符串中描述参数的部分，包含 args 和返回值，作为额外的细节供 LLM 使用。
    """
    signature = inspect.signature(fn_obj)  # 获取函数签名

    docstring = remove_spaces(docstring)  # 清理文档字符串

    overall_desc, param_desc = PARSER.parse(docstring)  # 解析文档字符串

    function_type = "function" if not inspect.iscoroutinefunction(fn_obj) else "async_function"  # 判断函数类型

    return {"type": function_type, "description": overall_desc, "signature": str(signature), "parameters": param_desc}


def get_class_method_docstring(cls, method_name):
    """检索方法的文档字符串，如果必要的话，搜索类的继承层次结构"""
    for base_class in cls.__mro__:
        if method_name in base_class.__dict__:
            method = base_class.__dict__[method_name]
            if method.__doc__:
                return method.__doc__  # 如果找到文档字符串，则返回
    return None  # 如果在类继承层次结构中找不到文档字符串


class CodeVisitor(ast.NodeVisitor):
    """访问并将代码文件中的 AST 节点转换为工具模式"""

    def __init__(self, source_code: str):
        self.tool_schemas = {}  # {工具名: 工具模式}
        self.source_code = source_code  # 源代码

    def visit_ClassDef(self, node):
        # 处理类定义
        class_schemas = {"type": "class", "description": remove_spaces(ast.get_docstring(node)), "methods": {}}
        for body_node in node.body:
            if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                not body_node.name.startswith("_") or body_node.name == "__init__"
            ):  # 跳过私有方法和非初始化方法
                func_schemas = self._get_function_schemas(body_node)  # 获取方法模式
                class_schemas["methods"].update({body_node.name: func_schemas})  # 更新类的方法模式
        class_schemas["code"] = ast.get_source_segment(self.source_code, node)  # 获取源代码段
        self.tool_schemas[node.name] = class_schemas  # 保存类模式

    def visit_FunctionDef(self, node):
        self._visit_function(node)  # 处理函数定义

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node)  # 处理异步函数定义

    def _visit_function(self, node):
        if node.name.startswith("_"):  # 跳过私有函数
            return
        function_schemas = self._get_function_schemas(node)  # 获取函数模式
        function_schemas["code"] = ast.get_source_segment(self.source_code, node)  # 获取源代码段
        self.tool_schemas[node.name] = function_schemas  # 保存函数模式

    def _get_function_schemas(self, node):
        docstring = remove_spaces(ast.get_docstring(node))  # 获取并清理文档字符串
        overall_desc, param_desc = PARSER.parse(docstring)  # 解析文档字符串
        return {
            "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",  # 判断函数类型
            "description": overall_desc,  # 函数描述
            "signature": self._get_function_signature(node),  # 函数签名
            "parameters": param_desc,  # 参数描述
        }

    def _get_function_signature(self, node):
        # 获取函数签名
        args = []
        defaults = dict(zip([arg.arg for arg in node.args.args][-len(node.args.defaults) :], node.args.defaults))
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                annotation = ast.unparse(arg.annotation)  # 获取参数注解
                arg_str += f": {annotation}"
            if arg.arg in defaults:
                default_value = ast.unparse(defaults[arg.arg])  # 获取默认值
                arg_str += f" = {default_value}"
            args.append(arg_str)

        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"  # 获取返回值类型

        return f"({', '.join(args)}){return_annotation}"  # 返回函数签名

    def get_tool_schemas(self):
        return self.tool_schemas  # 返回工具模式字典
