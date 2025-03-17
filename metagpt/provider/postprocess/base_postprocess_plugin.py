#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : base llm postprocess plugin to do the operations like repair the raw llm output

from typing import Union

from metagpt.utils.repair_llm_raw_output import (
    RepairType,
    extract_content_from_output,
    repair_llm_raw_output,
    retry_parse_json_text,
)


class BasePostProcessPlugin(object):
    model = None  # 用于判断的 `model` 插件，在 `llm_postprocess` 中使用

    def run_repair_llm_output(self, output: str, schema: dict, req_key: str = "[/CONTENT]") -> Union[dict, list]:
        """
        修复步骤：
            1. 使用 schema 的字段修复大小写问题
            2. 从 req_key 对应的内容中提取有效的 JSON 数据（如 xx[REQ_KEY]xxx[/REQ_KEY]xx）
            3. 修复内容中的无效 JSON 格式
            4. 解析 JSON 文本，并根据异常情况进行重试修复
        """
        output_class_fields = list(schema["properties"].keys())  # Custom ActionOutput 的字段

        content = self.run_repair_llm_raw_output(output, req_keys=output_class_fields + [req_key])
        content = self.run_extract_content_from_output(content, right_key=req_key)
        # req_keys 模拟
        content = self.run_repair_llm_raw_output(content, req_keys=[None], repair_type=RepairType.JSON)
        parsed_data = self.run_retry_parse_json_text(content)

        return parsed_data

    def run_repair_llm_raw_output(self, content: str, req_keys: list[str], repair_type: str = None) -> str:
        """继承类可以重写此函数"""
        return repair_llm_raw_output(content, req_keys=req_keys, repair_type=repair_type)

    def run_extract_content_from_output(self, content: str, right_key: str) -> str:
        """继承类可以重写此函数"""
        return extract_content_from_output(content, right_key=right_key)

    def run_retry_parse_json_text(self, content: str) -> Union[dict, list]:
        """继承类可以重写此函数"""
        # logger.info(f"从输出中提取的 JSON CONTENT：\n{content}")
        parsed_data = retry_parse_json_text(output=content)  # 应该使用 output=content
        return parsed_data

    def run(self, output: str, schema: dict, req_key: str = "[/CONTENT]") -> Union[dict, list]:
        """
        用于处理需要 JSON 格式输出的提示，且带有外部键对（例如 [REQ_KEY] 包含 JSON 数据 [/REQ_KEY]）

        参数：
            output (str): LLM 的原始输出
            schema (dict): 输出 JSON 模式
            req_key (str): 外部键对的右键，通常为 `[/REQ_KEY]` 格式
        """
        assert len(schema.get("properties")) > 0  # 确保 schema 中有 properties 字段
        assert "/" in req_key  # 确保 req_key 中包含分隔符 "/"

        # 当前，后处理仅处理修复 LLM 原始输出
        new_output = self.run_repair_llm_output(output=output, schema=schema, req_key=req_key)
        return new_output
