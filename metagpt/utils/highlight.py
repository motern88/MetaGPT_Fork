# 添加代码语法高亮显示
from pygments import highlight as highlight_
from pygments.formatters import HtmlFormatter, TerminalFormatter
from pygments.lexers import PythonLexer, SqlLexer


def highlight(code: str, language: str = "python", formatter: str = "terminal"):
    # 指定要高亮显示的编程语言
    if language.lower() == "python":
        lexer = PythonLexer()  # 如果是 Python 语言，使用 PythonLexer 进行高亮
    elif language.lower() == "sql":
        lexer = SqlLexer()  # 如果是 SQL 语言，使用 SqlLexer 进行高亮
    else:
        raise ValueError(f"不支持的语言: {language}")  # 如果语言不在支持的范围内，抛出异常

    # 指定输出格式
    if formatter.lower() == "terminal":
        formatter = TerminalFormatter()  # 如果格式是 terminal，使用 TerminalFormatter
    elif formatter.lower() == "html":
        formatter = HtmlFormatter()  # 如果格式是 HTML，使用 HtmlFormatter
    else:
        raise ValueError(f"不支持的格式化器: {formatter}")  # 如果格式不在支持的范围内，抛出异常

    # 使用 Pygments 对代码进行高亮
    return highlight_(code, lexer, formatter)  # 返回高亮后的代码
