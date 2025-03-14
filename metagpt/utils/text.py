from typing import Generator, Sequence

from metagpt.utils.token_counter import TOKEN_MAX, count_output_tokens


def reduce_message_length(
    msgs: Generator[str, None, None],
    model_name: str,
    system_text: str,
    reserved: int = 0,
) -> str:
    """减少拼接的消息段长度，以适应最大令牌大小。

    参数:
        msgs: 字符串生成器，表示逐渐缩短的有效提示。
        model_name: 用于编码的模型名称（例如 "gpt-3.5-turbo"）。
        system_text: 系统提示文本。
        reserved: 保留的令牌数。

    返回:
        拼接后的消息段，已减少到适应最大令牌大小。

    异常:
        RuntimeError: 如果未能成功减少消息长度，则抛出此异常。
    """
    max_token = TOKEN_MAX.get(model_name, 2048) - count_output_tokens(system_text, model_name) - reserved
    for msg in msgs:
        if count_output_tokens(msg, model_name) < max_token or model_name not in TOKEN_MAX:
            return msg

    raise RuntimeError("fail to reduce message length")


def generate_prompt_chunk(
    text: str,
    prompt_template: str,
    model_name: str,
    system_text: str,
    reserved: int = 0,
) -> Generator[str, None, None]:
    """将文本拆分成最大令牌大小的块。

    参数:
        text: 需要拆分的文本。
        prompt_template: 提示的模板，包含一个 `{}` 占位符，例如 "### Reference\n{}"。
        model_name: 用于编码的模型名称（例如 "gpt-3.5-turbo"）。
        system_text: 系统提示文本。
        reserved: 保留的令牌数。

    返回:
        生成每个文本块。
    """
    paragraphs = text.splitlines(keepends=True)
    current_token = 0
    current_lines = []

    reserved = reserved + count_output_tokens(prompt_template + system_text, model_name)
    # 100 是一个“魔法数字”，确保不会超出最大上下文长度
    max_token = TOKEN_MAX.get(model_name, 2048) - reserved - 100

    while paragraphs:
        paragraph = paragraphs.pop(0)
        token = count_output_tokens(paragraph, model_name)
        if current_token + token <= max_token:
            current_lines.append(paragraph)
            current_token += token
        elif token > max_token:
            paragraphs = split_paragraph(paragraph) + paragraphs
            continue
        else:
            yield prompt_template.format("".join(current_lines))
            current_lines = [paragraph]
            current_token = token

    if current_lines:
        yield prompt_template.format("".join(current_lines))


def split_paragraph(paragraph: str, sep: str = ".,", count: int = 2) -> list[str]:
    """将段落分割成多个部分。

    参数:
        paragraph: 需要分割的段落。
        sep: 用于分割的分隔符字符。
        count: 将段落分割成的部分数。

    返回:
        一个包含分割后的段落部分的列表。
    """
    for i in sep:
        sentences = list(_split_text_with_ends(paragraph, i))
        if len(sentences) <= 1:
            continue
        ret = ["".join(j) for j in _split_by_count(sentences, count)]
        return ret
    return list(_split_by_count(paragraph, count))


def decode_unicode_escape(text: str) -> str:
    """解码带有 Unicode 转义序列的文本。

    参数:
        text: 需要解码的文本。

    返回:
        解码后的文本。
    """
    return text.encode("utf-8").decode("unicode_escape", "ignore")


def _split_by_count(lst: Sequence, count: int):
    """将列表按指定的数量分割。

    参数:
        lst: 需要分割的列表。
        count: 将列表分割成的部分数。

    返回:
        生成分割后的部分。
    """
    avg = len(lst) // count
    remainder = len(lst) % count
    start = 0
    for i in range(count):
        end = start + avg + (1 if i < remainder else 0)
        yield lst[start:end]
        start = end


def _split_text_with_ends(text: str, sep: str = "."):
    """按分隔符将文本分割成多个部分。

    参数:
        text: 需要分割的文本。
        sep: 用于分割的分隔符。

    返回:
        生成分割后的部分。
    """
    parts = []
    for i in text:
        parts.append(i)
        if i == sep:
            yield "".join(parts)
            parts = []
    if parts:
        yield "".join(parts)
