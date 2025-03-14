#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the implement of serialization and deserialization

import copy
import pickle

from metagpt.utils.common import import_class


def actionoutout_schema_to_mapping(schema: dict) -> dict:
    """
    直接遍历 schema 的第一层 `properties`。
    schema 的结构如下所示：
    ```
    {
        "title":"prd",
        "type":"object",
        "properties":{
            "Original Requirements":{
                "title":"Original Requirements",
                "type":"string"
            },
        },
        "required":[
            "Original Requirements",
        ]
    }
    ```
    参数：
    - schema (dict): 输入的 schema 字典。

    返回值：
    - mapping (dict): 转换后的映射字典，其中键是字段名，值是类型元组。
    """
    mapping = dict()
    for field, property in schema["properties"].items():
        if property["type"] == "string":
            mapping[field] = (str, ...)
        elif property["type"] == "array" and property["items"]["type"] == "string":
            mapping[field] = (list[str], ...)
        elif property["type"] == "array" and property["items"]["type"] == "array":
            # 这里仅考虑 `list[list[str]]` 的情况
            mapping[field] = (list[list[str]], ...)
    return mapping


def actionoutput_mapping_to_str(mapping: dict) -> dict:
    """
    将映射字典中的类型转换为字符串表示。

    参数：
    - mapping (dict): 输入的映射字典。

    返回值：
    - new_mapping (dict): 转换后的字典，其中值是类型的字符串表示。
    """
    new_mapping = {}
    for key, value in mapping.items():
        new_mapping[key] = str(value)
    return new_mapping


def actionoutput_str_to_mapping(mapping: dict) -> dict:
    """
    将字符串类型的映射字典转换回原始的类型映射字典。

    参数：
    - mapping (dict): 输入的字符串类型的映射字典。

    返回值：
    - new_mapping (dict): 转换回原始类型的映射字典。
    """
    new_mapping = {}
    for key, value in mapping.items():
        if value == "(<class 'str'>, Ellipsis)":
            new_mapping[key] = (str, ...)
        else:
            new_mapping[key] = eval(value)  # 将 `"'(list[str], Ellipsis)"` 转换为 `(list[str], ...)`
    return new_mapping


def serialize_message(message: "Message"):
    """
    序列化消息对象，防止 `instruct_content` 值通过引用更新。

    参数：
    - message (Message): 要序列化的消息对象。

    返回值：
    - msg_ser (str): 序列化后的消息对象。
    """
    message_cp = copy.deepcopy(message)  # 避免通过引用更新 `instruct_content` 的值
    ic = message_cp.instruct_content
    if ic:
        # 模型通过 pydantic 创建，例如 `pydantic.main.prd`，不能直接用 pickle.dump
        schema = ic.model_json_schema()
        mapping = actionoutout_schema_to_mapping(schema)

        message_cp.instruct_content = {"class": schema["title"], "mapping": mapping, "value": ic.model_dump()}
    msg_ser = pickle.dumps(message_cp)

    return msg_ser


def deserialize_message(message_ser: str) -> "Message":
    """
    反序列化消息对象，恢复原始的消息内容。

    参数：
    - message_ser (str): 序列化后的消息字符串。

    返回值：
    - message (Message): 反序列化后的消息对象。
    """
    message = pickle.loads(message_ser)
    if message.instruct_content:
        ic = message.instruct_content
        # 动态导入 `ActionNode` 类，避免循环导入问题
        actionnode_class = import_class("ActionNode", "metagpt.actions.action_node")
        ic_obj = actionnode_class.create_model_class(class_name=ic["class"], mapping=ic["mapping"])
        ic_new = ic_obj(**ic["value"])
        message.instruct_content = ic_new

    return message
