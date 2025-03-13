"""Code Docstring Generator.

This script provides a tool to automatically generate docstrings for Python code. It uses the specified style to create
docstrings for the given code and system text.

Usage:
    python3 -m metagpt.actions.write_docstring <filename> [--overwrite] [--style=<docstring_style>]

Arguments:
    filename           The path to the Python file for which you want to generate docstrings.

Options:
    --overwrite        If specified, overwrite the original file with the code containing docstrings.
    --style=<docstring_style>   Specify the style of the generated docstrings.
                                Valid values: 'google', 'numpy', or 'sphinx'.
                                Default: 'google'

Example:
    python3 -m metagpt.actions.write_docstring ./metagpt/software_company.py --overwrite False --style=numpy

This script uses the 'fire' library to create a command-line interface. It generates docstrings for the given Python code using
the specified docstring style and adds them to the code.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Optional

from metagpt.actions.action import Action
from metagpt.utils.common import OutputParser, aread, awrite
from metagpt.utils.pycst import merge_docstring

PYTHON_DOCSTRING_SYSTEM = """### Requirements
1. Add docstrings to the given code following the {style} style.
2. Replace the function body with an Ellipsis object(...) to reduce output.
3. If the types are already annotated, there is no need to include them in the docstring.
4. Extract only class, function or the docstrings for the module parts from the given Python code, avoiding any other text.

### Input Example
```python
def function_with_pep484_type_annotations(param1: int) -> bool:
    return isinstance(param1, int)

class ExampleError(Exception):
    def __init__(self, msg: str):
        self.msg = msg
```

### Output Example
```python
{example}
```
"""

# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

PYTHON_DOCSTRING_EXAMPLE_GOOGLE = '''
def function_with_pep484_type_annotations(param1: int) -> bool:
    """Example function with PEP 484 type annotations.

    Extended description of function.

    Args:
        param1: The first parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    ...

class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method was documented in the class level docstring.

    Args:
        msg: Human readable string describing the exception.

    Attributes:
        msg: Human readable string describing the exception.
    """
    ...
'''

PYTHON_DOCSTRING_EXAMPLE_NUMPY = '''
def function_with_pep484_type_annotations(param1: int) -> bool:
    """
    Example function with PEP 484 type annotations.

    Extended description of function.

    Parameters
    ----------
    param1
        The first parameter.

    Returns
    -------
    bool
        The return value. True for success, False otherwise.
    """
    ...

class ExampleError(Exception):
    """
    Exceptions are documented in the same way as classes.

    The __init__ method was documented in the class level docstring.

    Parameters
    ----------
    msg
        Human readable string describing the exception.

    Attributes
    ----------
    msg
        Human readable string describing the exception.
    """
    ...
'''

PYTHON_DOCSTRING_EXAMPLE_SPHINX = '''
def function_with_pep484_type_annotations(param1: int) -> bool:
    """Example function with PEP 484 type annotations.

    Extended description of function.

    :param param1: The first parameter.
    :type param1: int

    :return: The return value. True for success, False otherwise.
    :rtype: bool
    """
    ...

class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method was documented in the class level docstring.

    :param msg: Human-readable string describing the exception.
    :type msg: str
    """
    ...
'''

_python_docstring_style = {
    "google": PYTHON_DOCSTRING_EXAMPLE_GOOGLE.strip(),
    "numpy": PYTHON_DOCSTRING_EXAMPLE_NUMPY.strip(),
    "sphinx": PYTHON_DOCSTRING_EXAMPLE_SPHINX.strip(),
}


class WriteDocstring(Action):
    """此类用于为代码编写文档字符串（docstring）。

    属性：
        desc: 描述该动作的字符串。
    """

    desc: str = "为代码编写文档字符串。"
    i_context: Optional[str] = None

    async def run(
        self,
        code: str,
        system_text: str = PYTHON_DOCSTRING_SYSTEM,
        style: Literal["google", "numpy", "sphinx"] = "google",
    ) -> str:
        """根据指定的风格为给定的代码和系统文本编写文档字符串。

        参数：
            code: 一个 Python 代码字符串。
            system_text: 一个系统文本字符串。
            style: 一个指定文档字符串风格的字符串，可以是 'google'，'numpy' 或 'sphinx'。

        返回：
            添加了文档字符串的 Python 代码。
        """
        system_text = system_text.format(style=style, example=_python_docstring_style[style])
        simplified_code = _simplify_python_code(code)  # 简化代码
        documented_code = await self._aask(f"```python\n{simplified_code}\n```", [system_text])
        documented_code = OutputParser.parse_python_code(documented_code)  # 解析文档代码
        return merge_docstring(code, documented_code)  # 合并原始代码与文档字符串

    @staticmethod
    async def write_docstring(
        filename: str | Path, overwrite: bool = False, style: Literal["google", "numpy", "sphinx"] = "google"
    ) -> str:
        """为文件中的代码添加文档字符串。

        参数：
            filename: 代码文件的路径。
            overwrite: 是否覆盖原文件中的代码。默认为 False。
            style: 文档字符串的风格，默认为 'google'。

        返回：
            带有文档字符串的 Python 代码。
        """
        data = await aread(str(filename))  # 读取文件内容
        code = await WriteDocstring().run(data, style=style)  # 生成文档字符串
        if overwrite:  # 如果设置为覆盖，写回文件
            await awrite(filename, code)
        return code


def _simplify_python_code(code: str) -> None:
    """简化给定的 Python 代码，通过移除表达式和最后的 if 语句。

    参数：
        code: 一个 Python 代码字符串。

    返回：
        简化后的 Python 代码。
    """
    code_tree = ast.parse(code)  # 解析代码
    code_tree.body = [i for i in code_tree.body if not isinstance(i, ast.Expr)]  # 移除表达式
    if isinstance(code_tree.body[-1], ast.If):  # 如果最后是 if 语句，移除它
        code_tree.body.pop()
    return ast.unparse(code_tree)  # 反序列化简化后的代码


if __name__ == "__main__":
    import fire

    fire.Fire(WriteDocstring.write_docstring)  # 通过 fire 库调用命令行功能
