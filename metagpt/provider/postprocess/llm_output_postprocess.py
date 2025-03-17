#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the entry of choosing which PostProcessPlugin to deal particular LLM model's output

from typing import Union

from metagpt.provider.postprocess.base_postprocess_plugin import BasePostProcessPlugin


def llm_output_postprocess(
        output: str, schema: dict, req_key: str = "[/CONTENT]", model_name: str = None
) -> Union[dict, str]:
    """
    默认使用 BasePostProcessPlugin，如果没有匹配的插件。

    参数：
        output (str): LLM 的原始输出结果
        schema (dict): 输出的 JSON 模式
        req_key (str): 外部键对的右键，通常为 `[/REQ_KEY]` 格式，默认为 "[/CONTENT]"
        model_name (str): 模型名称，用于选择不同的模型插件（当前未使用）

    返回：
        Union[dict, str]: 处理后的结果，返回一个字典或字符串
    """
    # TODO 根据模型选择不同的插件
    postprocess_plugin = BasePostProcessPlugin()  # 默认使用 BasePostProcessPlugin

    # 使用后处理插件进行处理
    result = postprocess_plugin.run(output=output, schema=schema, req_key=req_key)
    return result
