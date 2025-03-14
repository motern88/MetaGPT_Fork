#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : repair llm raw output with particular conditions

import copy
from enum import Enum
from typing import Callable, Optional, Union

import regex as re
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.utils.custom_decoder import CustomDecoder


class RepairType(Enum):
    CS = "case sensitivity"  # 大小写敏感
    RKPM = "required key pair missing"  # 缺少必要的键对，例如 `[key] xx` 但缺少 `[/key]`
    SCM = "special character missing"  # 缺少特殊字符，通常要求成对出现，如 `[key] xx [/key]`
    JSON = "json format"  # JSON 格式问题


def repair_case_sensitivity(output: str, req_key: str) -> str:
    """
    修复大小写敏感问题，通常 req_key 是期望 JSON 或 markdown 内容的键名，它不会出现在值部分。
    例如修复目标字符串 `"Shared Knowledge": ""`，但实际为 `"Shared knowledge": ""`
    """
    if req_key in output:
        return output

    output_lower = output.lower()  # 转换输出为小写
    req_key_lower = req_key.lower()  # 转换目标键名为小写
    if req_key_lower in output_lower:
        # 找到子字符串的索引并将其替换为原始的 req_key
        lidx = output_lower.find(req_key_lower)
        source = output[lidx : lidx + len(req_key_lower)]
        output = output.replace(source, req_key)
        logger.info(f"repair_case_sensitivity: {req_key}")

    return output


def repair_special_character_missing(output: str, req_key: str = "[/CONTENT]") -> str:
    """
    修复缺少特殊字符的情况
        1. 目标字符串 `[CONTENT] xx [CONTENT] xxx [CONTENT]` 缺少最后一个 `[CONTENT]` 的 `/`
        2. 目标字符串 `xx [CONTENT] xxx [CONTENT] xxxx` 缺少最后一个 `[CONTENT]` 的 `/`
    """
    sc_arr = ["/"]

    if req_key in output:
        return output

    for sc in sc_arr:
        req_key_pure = req_key.replace(sc, "")  # 去除斜杠
        appear_cnt = output.count(req_key_pure)  # 计算纯键的出现次数
        if req_key_pure in output and appear_cnt > 1:
            # req_key 通常出现在尾部
            ridx = output.rfind(req_key_pure)  # 找到最后一个出现的索引
            output = f"{output[:ridx]}{req_key}{output[ridx + len(req_key_pure):]}"  # 插入斜杠
            logger.info(f"repair_special_character_missing: {sc} in {req_key_pure} as position {ridx}")

    return output


def repair_required_key_pair_missing(output: str, req_key: str = "[/CONTENT]") -> str:
    """
    修复 req_key 配对缺失的情况
        req_key 格式：
            1. `[req_key]` 和其配对 `[/req_key]`
            2. `[/req_key]` 和其配对 `[req_key]`
    """
    sc = "/"  # 特殊字符
    if req_key.startswith("[") and req_key.endswith("]"):
        if sc in req_key:
            left_key = req_key.replace(sc, "")  # `[/req_key]` -> `[req_key]`
            right_key = req_key
        else:
            left_key = req_key
            right_key = f"{req_key[0]}{sc}{req_key[1:]}"  # `[req_key]` -> `[/req_key]`

        if left_key not in output:
            output = left_key + "\n" + output  # 将左键插入输出的开头
        if right_key not in output:
            # 判断是否潜在是 JSON 格式并处理
            def judge_potential_json(routput: str, left_key: str) -> Union[str, None]:
                ridx = routput.rfind(left_key)
                if ridx < 0:
                    return None
                sub_output = routput[ridx:]
                idx1 = sub_output.rfind("}")
                idx2 = sub_output.rindex("]")
                idx = idx1 if idx1 >= idx2 else idx2
                sub_output = sub_output[: idx + 1]
                return sub_output

            if output.strip().endswith("}") or (output.strip().endswith("]") and not output.strip().endswith(left_key)):
                # 避免 `[req_key]xx[req_key]` 这种情况，将 `[/req_key]` 插入到结尾
                output = output + "\n" + right_key
            elif judge_potential_json(output, left_key) and (not output.strip().endswith(left_key)):
                sub_content = judge_potential_json(output, left_key)
                output = sub_content + "\n" + right_key

    return output


def repair_json_format(output: str) -> str:
    """
    修复 JSON 格式问题，如去掉尾部的多余的 `[` 或 `}`。
    """
    output = output.strip()

    if output.startswith("[{"):
        output = output[1:]  # 去掉开始的 `[`
        logger.info(f"repair_json_format: {'[{'}")
    elif output.endswith("}]"):
        output = output[:-1]  # 去掉结束的 `]`
        logger.info(f"repair_json_format: {'}]'}")
    elif output.startswith("{") and output.endswith("]"):
        output = output[:-1] + "}"  # 修复从 `{` 到 `]` 的不匹配问题

    # 去掉 JSON 字符串中的注释
    arr = output.split("\n")
    new_arr = []
    for json_line in arr:
        # 查找注释并删除它们
        comment_index = -1
        for match in re.finditer(r"(\".*?\"|\'.*?\')|(#|//)", json_line):
            if match.group(1):  # 如果是字符串值
                continue
            if match.group(2):  # 如果是注释
                comment_index = match.start(2)
                break
        # 删除注释部分
        if comment_index != -1:
            json_line = json_line[:comment_index].rstrip()
        new_arr.append(json_line)
    output = "\n".join(new_arr)
    return output


def _repair_llm_raw_output(output: str, req_key: str, repair_type: RepairType = None) -> str:
    repair_types = [repair_type] if repair_type else [item for item in RepairType if item not in [RepairType.JSON]]
    for repair_type in repair_types:
        if repair_type == RepairType.CS:
            output = repair_case_sensitivity(output, req_key)
        elif repair_type == RepairType.RKPM:
            output = repair_required_key_pair_missing(output, req_key)
        elif repair_type == RepairType.SCM:
            output = repair_special_character_missing(output, req_key)
        elif repair_type == RepairType.JSON:
            output = repair_json_format(output)
    return output


def repair_llm_raw_output(
    output: str, req_keys: list[str], repair_type: RepairType = None, config: Optional[Config] = None
) -> str:
    """
    处理开放源代码的 LLM 模型输出，通常模型可能无法完全遵循指令，输出可能不完整，
    所以这里会尝试修复输出，默认使用所有修复方法。
    """
    config = config if config else Config.default()
    if not config.repair_llm_output:
        return output

    # 对每个 req_key 进行修复
    for req_key in req_keys:
        output = _repair_llm_raw_output(output=output, req_key=req_key, repair_type=repair_type)
    return output


def repair_invalid_json(output: str, error: str) -> str:
    """
    修复无效的 JSON 格式问题，通常是错误提示中存在多余的字符。
    错误示例：
        例 1. json.decoder.JSONDecodeError: Expecting ',' delimiter: line 154 column 1 (char 2765)
        例 2. xxx.JSONDecodeError: Expecting property name enclosed in double quotes: line 14 column 1 (char 266)
    """
    pattern = r"line ([0-9]+) column ([0-9]+)"

    matches = re.findall(pattern, error, re.DOTALL)
    if len(matches) > 0:
        # 从错误信息中提取出行号和列号，并减去1来适应0基索引
        line_no = int(matches[0][0]) - 1
        col_no = int(matches[0][1]) - 1

        # 因为 CustomDecoder 可以处理 `"": ''` 或 `'': ""` 这样的情况，所以将 `"""` 替换为 `"`，`'''` 替换为 `'`
        output = output.replace('"""', '"').replace("'''", '"')
        arr = output.split("\n")  # 将输出按行拆分成数组
        rline = arr[line_no]  # 获取原始的行
        line = arr[line_no].strip()  # 去除行两端的空格

        # 常见的几种问题修复
        if line.endswith("],"):
            # 问题：冗余字符 `]`
            new_line = line.replace("]", "")
        elif line.endswith("},") and not output.endswith("},"):
            # 问题：冗余字符 `}`
            new_line = line.replace("}", "")
        elif line.endswith("},") and output.endswith("},"):
            # 问题：末尾多了一个逗号
            new_line = line[:-1]
        elif (rline[col_no] in ["'", '"']) and (line.startswith('"') or line.startswith("'")) and "," not in line:
            # 问题：`"""` 或 `'''` 没有 `,` 后续字符
            new_line = f",{line}"
        elif col_no - 1 >= 0 and rline[col_no - 1] in ['"', "'"]:
            # 问题：输出中存在转义字符，如 `\"`
            char = rline[col_no - 1]
            nearest_char_idx = rline[col_no:].find(char)
            new_line = (
                rline[: col_no - 1]
                + "\\"  # 添加转义符
                + rline[col_no - 1 : col_no + nearest_char_idx]
                + "\\"  # 添加转义符
                + rline[col_no + nearest_char_idx :]
            )
        elif '",' not in line and "," not in line and '"' not in line:
            # 问题：缺少逗号，补充逗号
            new_line = f'{line}",'
        elif not line.endswith(","):
            # 问题：行尾缺少逗号
            new_line = f"{line},"
        elif "," in line and len(line) == 1:
            # 问题：格式不正确，只有一个字符时加上引号
            new_line = f'"{line}'
        elif '",' in line:
            # 问题：需要调整逗号和引号的顺序
            new_line = line[:-2] + "',"
        else:
            # 如果没有问题，保持行不变
            new_line = line

        arr[line_no] = new_line  # 更新该行
        output = "\n".join(arr)  # 重新拼接所有行
        logger.info(f"repair_invalid_json, raw error: {error}")  # 记录修复信息

    return output


def run_after_exp_and_passon_next_retry(logger: "loguru.Logger") -> Callable[["RetryCallState"], None]:
    def run_and_passon(retry_state: RetryCallState) -> None:
        """
        RetryCallState 示例：
            {
                "start_time":143.098322024,
                "retry_object":"<Retrying object at 0x7fabcaca25e0 (stop=<tenacity.stop.stop_after_attempt ... >)>",
                "fn":"<function retry_parse_json_text_v2 at 0x7fabcac80ee0>",  # 被重试的函数
                "args":"(\"tag:[/CONTENT]\",)",  # 函数的输入参数
                "kwargs":{},  # 函数的输入关键词参数
                "attempt_number":1,  # 重试次数
                "outcome":"<Future at xxx>",  # 类型（outcome.result() = "str", outcome.exception() = "class"）
                "outcome_timestamp":143.098416904,
                "idle_for":0,
                "next_action":"None"
            }
        """
        config = Config.default()
        if retry_state.outcome.failed:
            if retry_state.args:
                # 如果 args 存在，提取第一个参数
                func_param_output = retry_state.args[0]
            elif retry_state.kwargs:
                # 如果 kwargs 存在，提取 output 参数
                func_param_output = retry_state.kwargs.get("output", "")
            exp_str = str(retry_state.outcome.exception())

            fix_str = "try to fix it, " if config.repair_llm_output else ""
            logger.warning(
                f"解析[CONTENT][/CONTENT]中的JSON失败，在重试 "
                f"{retry_state.attempt_number} 次，{fix_str}异常：{exp_str}"
            )

            # 修复无效的JSON输出
            repaired_output = repair_invalid_json(func_param_output, exp_str)
            retry_state.kwargs["output"] = repaired_output

    return run_and_passon


def repair_stop_after_attempt(retry_state):
    # 根据配置决定最大重试次数
    return stop_after_attempt(3 if Config.default().repair_llm_output else 0)(retry_state)


@retry(
    stop=repair_stop_after_attempt,
    wait=wait_fixed(1),
    after=run_after_exp_and_passon_next_retry(logger),
)
def retry_parse_json_text(output: str) -> Union[list, dict]:
    """
    修复JSON文本的错误，例如额外的字符，如 [']', '}']。

    警告：
        如果 CONFIG.repair_llm_output 为 False，那么重试 _aask_v1 {x=3} 次，retry_parse_json_text 的重试无效；
        如果 CONFIG.repair_llm_output 为 True，_aask_v1 和 retry_parse_json_text 会循环执行 {x=3*3} 次。
        这是一个双重重试循环。
    """
    # logger.debug(f"输出用于JSON解码:\n{output}")

    # 如果 CONFIG.repair_llm_output 为 True，则会尝试修复输出直到重试停止
    parsed_data = CustomDecoder(strict=False).decode(output)

    return parsed_data


def extract_content_from_output(content: str, right_key: str = "[/CONTENT]"):
    """使用正则模式从 [CONTENT](xxx)[/CONTENT] 中提取 xxx 内容"""

    def re_extract_content(cont: str, pattern: str) -> str:
        matches = re.findall(pattern, cont, re.DOTALL)
        for match in matches:
            if match:
                cont = match
                break
        return cont.strip()

    # TODO 构建带有 right_key 的提取模式
    raw_content = copy.deepcopy(content)
    pattern = r"\[CONTENT\]([\s\S]*)\[/CONTENT\]"
    new_content = re_extract_content(raw_content, pattern)

    if not new_content.startswith("{"):
        # TODO 找到更通用的模式
        # 对于 `[CONTENT]xxx[CONTENT]xxxx[/CONTENT]` 的情况
        logger.warning(f"extract_content 尝试使用另一个模式: {pattern}")
        if right_key not in new_content:
            raw_content = copy.deepcopy(new_content + "\n" + right_key)
        # # 模式 = r"\[CONTENT\](\s*\{.*?\}\s*)\[/CONTENT\]"
        new_content = re_extract_content(raw_content, pattern)
    else:
        if right_key in new_content:
            idx = new_content.find(right_key)
            new_content = new_content[:idx]
            new_content = new_content.strip()

    return new_content


def extract_state_value_from_output(content: str) -> str:
    """
    对于 OpenAI 模型，它们会始终返回状态数。但对于开放 LLM 模型，指令结果可能是一个
    长文本，包含目标数字，因此这里添加了提取来提高成功率。

    参数：
        content (str): LLM 输出来自 `Role._think` 的结果
    """
    content = content.strip()  # 处理类似 " 0"、"0\n" 等输出情况
    pattern = (
        r"(?<!-)[0-9]"  # TODO 寻找更合适的方式而不是通过正则模式从内容中提取数字
    )
    matches = re.findall(pattern, content, re.DOTALL)
    matches = list(set(matches))
    state = matches[0] if len(matches) > 0 else "-1"
    return state


def repair_escape_error(commands):
    """
    修复命令响应中的转义错误。
    当 RoleZero 解析命令时，命令可能包含未知的转义字符。

    该函数有两个步骤：
    1. 将像 "\d" 和 "\(" 这样的未转义子字符串转换为 "\\\\d" 和 "\\\\("。
    2. 将转义字符像 '\f' 转换为类似 "\\\\f" 的子字符串。

    示例：
        当原始 JSON 字符串是 " {"content":"\\\\( \\\\frac{1}{2} \\\\)"} ",
        "content" 将正确解析为 "\( \frac{1}{2} \)"。

        然而，如果原始 JSON 字符串是 " {"content":"\( \frac{1}{2} \)"}" 直接解析，
        将会导致解析错误。

        为了修复错误的 JSON 字符串，以下转换将被应用：
        "\("   --->  "\\\\("
        '\f'   --->  "\\\\f"
        "\)"   --->  "\\\\)"
    """
    escape_repair_map = {
        "\a": "\\\\a",
        "\b": "\\\\b",
        "\f": "\\\\f",
        "\r": "\\\\r",
        "\t": "\\\\t",
        "\v": "\\\\v",
    }
    new_command = ""
    for index, ch in enumerate(commands):
        if ch == "\\" and index + 1 < len(commands):
            if commands[index + 1] not in ["n", '"', " "] :
                new_command += "\\"  # 不需要转义的字符，直接加回反斜杠
        elif ch in escape_repair_map:
            ch = escape_repair_map[ch]  # 替换转义字符
        new_command += ch
    return new_command
