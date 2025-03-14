import re
from typing import Tuple


def remove_spaces(text: str) -> str:
    """
    去除字符串中的多余空格，将连续的空格替换为一个空格，并去掉首尾空格。

    Args:
        text (str): 需要处理的文本字符串。

    Returns:
        str: 去除空格后的字符串。
    """
    return re.sub(r"\s+", " ", text).strip() if text else ""


class DocstringParser:
    """
    Docstring 解析器基类，用于解析 Python 函数或类的 docstring。
    """

    @staticmethod
    def parse(docstring: str) -> Tuple[str, str]:
        """
        解析 docstring 并返回整体描述和参数描述。

        Args:
            docstring (str): 需要解析的 docstring。

        Returns:
            Tuple[str, str]: 返回一个元组，包含整体描述和参数描述。
        """
        raise NotImplementedError("Subclasses must implement this method.")


class reSTDocstringParser(DocstringParser):
    """
    reStructuredText (reST) 格式的 docstring 解析器。
    """

    pass


class GoogleDocstringParser(DocstringParser):
    """
    Google 风格的 docstring 解析器。

    解析时，整体描述和参数描述分开处理，参数部分从 'Args:' 关键字开始。
    """

    @staticmethod
    def parse(docstring: str) -> Tuple[str, str]:
        """
        解析 Google 风格的 docstring。

        Args:
            docstring (str): 需要解析的 docstring。

        Returns:
            Tuple[str, str]: 返回整体描述和参数描述的元组。
        """
        if not docstring:
            return "", ""

        # 去除多余的空格
        docstring = remove_spaces(docstring)

        # 判断是否包含 'Args:' 关键字
        if "Args:" in docstring:
            # 如果包含 'Args:'，将 docstring 分成整体描述和参数描述
            overall_desc, param_desc = docstring.split("Args:", 1)
            param_desc = "Args:" + param_desc
        else:
            overall_desc = docstring
            param_desc = ""

        return overall_desc, param_desc
