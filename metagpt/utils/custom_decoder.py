import json
import re
from json import JSONDecodeError
from json.decoder import _decode_uXXXX

# 匹配数字的正则表达式
NUMBER_RE = re.compile(r"(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?", (re.VERBOSE | re.MULTILINE | re.DOTALL))

# 用于创建扫描器的函数
def py_make_scanner(context):
    parse_object = context.parse_object  # 解析对象
    parse_array = context.parse_array  # 解析数组
    parse_string = context.parse_string  # 解析字符串
    match_number = NUMBER_RE.match  # 匹配数字
    strict = context.strict  # 是否严格模式
    parse_float = context.parse_float  # 解析浮点数
    parse_int = context.parse_int  # 解析整数
    parse_constant = context.parse_constant  # 解析常量
    object_hook = context.object_hook  # 处理对象的回调函数
    object_pairs_hook = context.object_pairs_hook  # 处理键值对的回调函数
    memo = context.memo  # 用于优化的内存

    # 执行单次扫描的函数
    def _scan_once(string, idx):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar in ("'", '"'):
            # 处理三引号字符串
            if idx + 2 < len(string) and string[idx + 1] == nextchar and string[idx + 2] == nextchar:
                return parse_string(string, idx + 3, strict, delimiter=nextchar * 3)  # 三引号
            else:
                return parse_string(string, idx + 1, strict, delimiter=nextchar)  # 普通引号
        elif nextchar == "{":
            # 处理对象
            return parse_object((string, idx + 1), strict, _scan_once, object_hook, object_pairs_hook, memo)
        elif nextchar == "[":
            # 处理数组
            return parse_array((string, idx + 1), _scan_once)
        elif nextchar == "n" and string[idx : idx + 4] == "null":
            # 处理null
            return None, idx + 4
        elif nextchar == "t" and string[idx : idx + 4] == "true":
            # 处理true
            return True, idx + 4
        elif nextchar == "f" and string[idx : idx + 5] == "false":
            # 处理false
            return False, idx + 5

        m = match_number(string, idx)
        if m is not None:
            # 处理数字
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or "") + (exp or ""))
            else:
                res = parse_int(integer)
            return res, m.end()
        elif nextchar == "N" and string[idx : idx + 3] == "NaN":
            return parse_constant("NaN"), idx + 3
        elif nextchar == "I" and string[idx : idx + 8] == "Infinity":
            return parse_constant("Infinity"), idx + 8
        elif nextchar == "-" and string[idx : idx + 9] == "-Infinity":
            return parse_constant("-Infinity"), idx + 9
        else:
            raise StopIteration(idx)

    # 执行扫描并清除缓存
    def scan_once(string, idx):
        try:
            return _scan_once(string, idx)
        finally:
            memo.clear()

    return scan_once


# 正则表达式标志
FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL

# 匹配字符串内容的正则表达式
STRINGCHUNK = re.compile(r'(.*?)(["\\\x00-\x1f])', FLAGS)
STRINGCHUNK_SINGLEQUOTE = re.compile(r"(.*?)([\'\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_DOUBLE_QUOTE = re.compile(r"(.*?)(\"\"\"|[\\\x00-\x1f])", FLAGS)
STRINGCHUNK_TRIPLE_SINGLEQUOTE = re.compile(r"(.*?)('''|[\\\x00-\x1f])", FLAGS)

# 转义字符映射
BACKSLASH = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

# 匹配空白字符的正则表达式
WHITESPACE = re.compile(r"[ \t\n\r]*", FLAGS)
WHITESPACE_STR = " \t\n\r"


# 解析JSON对象的函数
def JSONObject(
        s_and_end, strict, scan_once, object_hook, object_pairs_hook, memo=None, _w=WHITESPACE.match, _ws=WHITESPACE_STR
):
    """解析JSON对象并返回解析后的对象。

    参数：
        s_and_end (tuple): 包含输入字符串和当前索引的元组。
        strict (bool): 如果为True，表示严格的JSON解码规则；如果为False，则允许字符串中的控制字符。默认为True。
        scan_once (callable): 用于扫描并解析JSON值的函数。
        object_hook (callable): 如果指定，将使用该函数处理解析后的对象。
        object_pairs_hook (callable): 如果指定，将使用该函数处理解析后的键值对。
        memo (dict, optional): 用于优化的内存字典，默认为None。
        _w (function): 用于匹配空白字符的正则表达式函数，默认为WHITESPACE.match。
        _ws (str): 包含空白字符的字符串，默认为WHITESPACE_STR。

    返回：
        tuple或dict: 返回解析后的对象和输入字符串中对象结束后的字符索引。
    """

    s, end = s_and_end
    pairs = []
    pairs_append = pairs.append
    if memo is None:
        memo = {}
    memo_get = memo.setdefault

    nextchar = s[end: end + 1]
    if nextchar != '"' and nextchar != "'":
        if nextchar in _ws:
            end = _w(s, end).end()
            nextchar = s[end: end + 1]
        if nextchar == "}":
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                return result, end + 1
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return pairs, end + 1
        elif nextchar != '"':
            raise JSONDecodeError("Expecting property name enclosed in double quotes", s, end)
    end += 1
    while True:
        if end + 1 < len(s) and s[end] == nextchar and s[end + 1] == nextchar:
            key, end = scanstring(s, end + 2, strict, delimiter=nextchar * 3)
        else:
            key, end = scanstring(s, end, strict, delimiter=nextchar)
        key = memo_get(key, key)

        if s[end: end + 1] != ":":
            end = _w(s, end).end()
            if s[end: end + 1] != ":":
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        end += 1

        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass

        try:
            value, end = scan_once(s, end)
        except StopIteration as err:
            raise JSONDecodeError("Expecting value", s, err.value) from None
        pairs_append((key, value))
        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _w(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ""
        end += 1

        if nextchar == "}":
            break
        elif nextchar != ",":
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        end = _w(s, end).end()
        nextchar = s[end: end + 1]
        end += 1
        if nextchar != '"':
            raise JSONDecodeError("Expecting property name enclosed in double quotes", s, end - 1)
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return result, end
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return pairs, end


def py_scanstring(s, end, strict=True, _b=BACKSLASH, _m=STRINGCHUNK.match, delimiter='"'):
    """Scan the string s for a JSON string.

    Args:
        s (str): 需要扫描的输入字符串。
        end (int): 传入字符串 s 中，开始 JSON 字符串的引号后的字符索引。
        strict (bool): 如果为 True，则强制执行严格的 JSON 字符串解码规则。
            如果为 False，则允许字符串中的控制字符。默认为 True。
        _b (dict): 包含转义序列映射的字典。
        _m (function): 用于匹配字符串块的正则表达式函数。
        delimiter (str): The string delimiter used to define the start and end of the JSON string.
            Can be one of: '"', "'", '\"""', or "'''". Defaults to '"'.

    返回值：
        tuple: 一个包含解码后的字符串和 s 中字符串结束引号后的字符索引的元组。
    """

    chunks = []  # 用于存储解析出的字符串片段
    _append = chunks.append  # 定义一个简化的追加操作
    begin = end - 1  # 获取字符串开始位置
    # 根据传入的定界符，选择相应的正则表达式进行匹配
    if delimiter == '"':
        _m = STRINGCHUNK.match
    elif delimiter == "'":
        _m = STRINGCHUNK_SINGLEQUOTE.match
    elif delimiter == '"""':
        _m = STRINGCHUNK_TRIPLE_DOUBLE_QUOTE.match
    else:
        _m = STRINGCHUNK_TRIPLE_SINGLEQUOTE.match

    while 1:
        # 尝试匹配字符串的下一个块
        chunk = _m(s, end)
        if chunk is None:
            raise JSONDecodeError("Unterminated string starting at", s, begin)

        end = chunk.end()  # 更新结束位置
        content, terminator = chunk.groups()  # 获取内容和终止符
        # 如果有内容，加入到结果列表中
        if content:
            _append(content)
        # 终止符为引号，表示字符串解析完成
        if terminator == delimiter:
            break
        # 如果遇到的终止符不是反斜杠，且严格模式下报错
        elif terminator != "\\":
            if strict:
                # msg = "Invalid control character %r at" % (terminator,)
                msg = "Invalid control character {0!r} at".format(terminator)
                raise JSONDecodeError(msg, s, end)
            else:
                _append(terminator)
                continue
        try:
            esc = s[end]  # 获取转义字符后的下一个字符
        except IndexError:
            raise JSONDecodeError("Unterminated string starting at", s, begin) from None

        # 如果不是 Unicode 转义序列，则必须在转义序列查找表中
        if esc != "u":
            try:
                char = _b[esc]
            except KeyError:
                msg = "Invalid \\escape: {0!r}".format(esc)
                raise JSONDecodeError(msg, s, end)
            end += 1
        else:
            # 解析 Unicode 转义字符
            uni = _decode_uXXXX(s, end)
            end += 5
            # 如果是高代理区，尝试解析后续的低代理区字符
            if 0xD800 <= uni <= 0xDBFF and s[end: end + 2] == "\\u":
                uni2 = _decode_uXXXX(s, end + 1)
                if 0xDC00 <= uni2 <= 0xDFFF:
                    uni = 0x10000 + (((uni - 0xD800) << 10) | (uni2 - 0xDC00))
                    end += 6
            char = chr(uni)  # 获取字符对应的 Unicode 字符
        _append(char)  # 将字符追加到结果中

    return "".join(chunks), end  # 返回解码后的字符串和结束位置


scanstring = py_scanstring  # 将 py_scanstring 函数赋值给 scanstring 变量


class CustomDecoder(json.JSONDecoder):
    """自定义 JSON 解码器类，继承自 json.JSONDecoder。

    可以自定义解析过程，例如解析字符串时使用 py_scanstring。
    """

    def __init__(
            self,
            *,
            object_hook=None,
            parse_float=None,
            parse_int=None,
            parse_constant=None,
            strict=True,
            object_pairs_hook=None
    ):
        # 调用父类的构造函数，初始化自定义解码器
        super().__init__(
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )
        self.parse_object = JSONObject  # 解析对象
        self.parse_string = py_scanstring  # 解析字符串，使用自定义的 py_scanstring
        self.scan_once = py_make_scanner(self)  # 使用自定义扫描器

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        """重写 decode 方法，进行 JSON 解码。"""
        return super().decode(s)  # 调用父类的 decode 方法解码 JSON 字符串
