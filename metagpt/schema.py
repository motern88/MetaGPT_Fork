#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/8 22:12
@Author  : alexanderwu
@File    : schema.py
@Modified By: mashenquan, 2023-10-31.
        根据 RFC 116 的第 2.2.1 章：重新规划了 Message 类属性的职责分工和功能定位。
@Modified By: mashenquan, 2023/11/22.
        1. 在 RFC 135 的第 2.2.3.4 章中，为 FileRepository 添加了 Document 和 Documents。
        2. 将常见的键值集合封装为 pydantic 结构，以标准化和统一操作之间的参数传递。
        3. 根据 RFC 135 的第 2.2.3.1.1 章，为 Message 添加了 id。
"""

from __future__ import annotations

import asyncio
import json
import os.path
import time
import uuid
from abc import ABC
from asyncio import Queue, QueueEmpty, wait_for
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    create_model,
    field_serializer,
    field_validator,
)

from metagpt.base.base_serialization import BaseSerialization
from metagpt.const import (
    AGENT,
    MESSAGE_ROUTE_CAUSE_BY,
    MESSAGE_ROUTE_FROM,
    MESSAGE_ROUTE_TO,
    MESSAGE_ROUTE_TO_ALL,
    SERDESER_PATH,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.repo_parser import DotClassInfo
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import (
    CodeParser,
    any_to_str,
    any_to_str_set,
    aread,
    import_class,
    read_json_file,
    write_json_file,
)
from metagpt.utils.exceptions import handle_exception
from metagpt.utils.report import TaskReporter
from metagpt.utils.serialize import (
    actionoutout_schema_to_mapping,
    actionoutput_mapping_to_str,
    actionoutput_str_to_mapping,
)


class SerializationMixin(BaseSerialization):
    @handle_exception
    def serialize(self, file_path: str = None) -> str:
        """将当前实例序列化为 JSON 文件。

        如果发生异常，`handle_exception` 将捕获异常并返回 `None`。

        参数:
            file_path (str, 可选): JSON 文件的路径，用于保存实例数据。默认为 None。

        返回:
            str: 保存实例数据的 JSON 文件路径。
        """

        file_path = file_path or self.get_serialization_path()

        serialized_data = self.model_dump()

        write_json_file(file_path, serialized_data, use_fallback=True)
        logger.debug(f"{self.__class__.__qualname__} 序列化成功，文件保存路径: {file_path}")

        return file_path

    @classmethod
    @handle_exception
    def deserialize(cls, file_path: str = None) -> BaseModel:
        """从 JSON 文件反序列化为类的实例。

        如果发生异常，`handle_exception` 将捕获异常并返回 `None`。

        参数:
            file_path (str, 可选): 需要读取的 JSON 文件路径。默认为 None。

        返回:
            cls 的实例。
        """

        file_path = file_path or cls.get_serialization_path()

        data: dict = read_json_file(file_path)

        model = cls(**data)
        logger.debug(f"{cls.__qualname__} 反序列化成功，实例由文件 {file_path} 创建")

        return model

    @classmethod
    def get_serialization_path(cls) -> str:
        """获取类的默认序列化路径。

        该方法基于类名构造一个文件路径，默认路径格式为 `./workspace/storage/ClassName.json`，
        其中 `ClassName` 为类的名称。

        返回:
            str: 序列化文件的路径。
        """

        return str(SERDESER_PATH / f"{cls.__qualname__}.json")


class SimpleMessage(BaseModel):
    """表示一个简单的消息对象。"""

    content: str
    role: str


class Document(BaseModel):
    """
    表示一个文档。
    """

    root_path: str = ""  # 文档的根路径
    filename: str = ""  # 文档文件名
    content: str = ""  # 文档内容

    def get_meta(self) -> Document:
        """获取文档的元数据。

        返回:
            Document: 一个新的 `Document` 实例，包含相同的 `root_path` 和 `filename`，但不包含 `content`。
        """

        return Document(root_path=self.root_path, filename=self.filename)

    @property
    def root_relative_path(self):
        """获取文档相对于 Git 仓库根目录的相对路径。

        返回:
            str: 文档的相对路径。
        """
        return os.path.join(self.root_path, self.filename)

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content

    @classmethod
    async def load(
            cls, filename: Union[str, Path], project_path: Optional[Union[str, Path]] = None
    ) -> Optional["Document"]:
        """
        从文件加载文档。

        参数:
            filename (Union[str, Path]): 要加载的文件路径。
            project_path (Optional[Union[str, Path]], 可选): 项目根路径，默认为 None。

        返回:
            Optional[Document]: 加载的文档实例，如果文件不存在则返回 None。
        """
        if not filename or not Path(filename).exists():
            return None
        content = await aread(filename=filename)
        doc = cls(content=content, filename=str(filename))
        if project_path and Path(filename).is_relative_to(project_path):
            doc.root_path = Path(filename).relative_to(project_path).parent
            doc.filename = Path(filename).name
        return doc


class Documents(BaseModel):
    """表示一组文档的类。

    属性:
        docs (Dict[str, Document]): 一个字典，将文档名称映射到 Document 实例。
    """

    docs: Dict[str, Document] = Field(default_factory=dict)

    @classmethod
    def from_iterable(cls, documents: Iterable[Document]) -> Documents:
        """从可迭代的 Document 实例列表创建一个 Documents 实例。

        参数:
            documents (Iterable[Document]): 一个 Document 实例的可迭代对象。

        返回:
            Documents: 一个包含所有文档的 Documents 实例。
        """

        docs = {doc.filename: doc for doc in documents}
        return Documents(docs=docs)

    def to_action_output(self) -> "ActionOutput":
        """转换为操作输出格式。

        返回:
            ActionOutput: 一个包含文档内容的操作输出实例。
        """
        from metagpt.actions.action_output import ActionOutput

        return ActionOutput(content=self.model_dump_json(), instruct_content=self)


class Resource(BaseModel):
    """用于 `Message`.`parse_resources` 方法的资源类。

    属性:
        resource_type (str): 资源的类型。
        value (str): 资源的内容（字符串类型）。
        description (str): 资源的描述信息。
    """

    resource_type: str  # 资源类型
    value: str  # 资源内容
    description: str  # 资源描述


class Message(BaseModel):
    """表示对话中的消息，格式为 list[<role>: <content>]。

    属性:
        id (str): 消息的唯一标识符，默认自动生成。
        content (str): 用户或代理的自然语言内容。
        instruct_content (Optional[BaseModel]): 结构化的指令内容，可选。
        role (str): 消息的角色，默认为 "user"（系统 / 用户 / 助手）。
        cause_by (str): 触发消息的原因。
        sent_from (str): 发送消息的来源。
        send_to (set[str]): 发送目标，默认为全体广播。
        metadata (Dict[str, Any]): 存储 `content` 和 `instruct_content` 相关的元数据。
    """

    id: str = Field(default="", validate_default=True)  # RFC 135 标准唯一 ID
    content: str  # 自然语言内容
    instruct_content: Optional[BaseModel] = Field(default=None, validate_default=True)  # 结构化指令内容
    role: str = "user"  # 角色: system / user / assistant
    cause_by: str = Field(default="", validate_default=True)  # 消息触发原因
    sent_from: str = Field(default="", validate_default=True)  # 消息发送来源
    send_to: set[str] = Field(default={MESSAGE_ROUTE_TO_ALL}, validate_default=True)  # 消息接收者，默认广播
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 存储内容相关的元数据

    @field_validator("id", mode="before")
    @classmethod
    def check_id(cls, id: str) -> str:
        """确保 ID 不能为空，否则自动生成唯一 ID。"""
        return id if id else uuid.uuid4().hex

    @field_validator("instruct_content", mode="before")
    @classmethod
    def check_instruct_content(cls, ic: Any) -> BaseModel:
        """检查 instruct_content 是否符合格式，并将字典转换为 BaseModel 实例。

        兼容:
            - 自定义 `ActionOutput`
            - 继承自 `BaseModel` 的类
        """

        if ic and isinstance(ic, dict) and "class" in ic:
            if "mapping" in ic:
                # 兼容 `ActionOutput`
                mapping = actionoutput_str_to_mapping(ic["mapping"])
                actionnode_class = import_class("ActionNode", "metagpt.actions.action_node")  # 避免循环导入
                ic_obj = actionnode_class.create_model_class(class_name=ic["class"], mapping=mapping)
            elif "module" in ic:
                # 解析 BaseModel 子类
                ic_obj = import_class(ic["class"], ic["module"])
            else:
                raise KeyError("缺少初始化 Message.instruct_content 所需的键")
            ic = ic_obj(**ic["value"])
        return ic

    @field_validator("cause_by", mode="before")
    @classmethod
    def check_cause_by(cls, cause_by: Any) -> str:
        """确保 cause_by 不能为空，若为空则默认使用 'UserRequirement'。"""
        return any_to_str(cause_by if cause_by else import_class("UserRequirement", "metagpt.actions.add_requirement"))

    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        """确保 sent_from 不能为空，若为空则默认为空字符串。"""
        return any_to_str(sent_from if sent_from else "")

    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> set:
        """确保 send_to 不能为空，若为空则默认广播消息。"""
        return any_to_str_set(send_to if send_to else {MESSAGE_ROUTE_TO_ALL})

    @field_serializer("send_to", mode="plain")
    def ser_send_to(self, send_to: set) -> list:
        """将 send_to 转换为列表，以便序列化存储。"""
        return list(send_to)

    @field_serializer("instruct_content", mode="plain")
    def ser_instruct_content(self, ic: BaseModel) -> Union[dict, None]:
        """将 instruct_content 序列化为字典格式。

        兼容:
            - `ActionOutput`
            - 继承自 `BaseModel` 的类
        """

        ic_dict = None
        if ic:
            schema = ic.model_json_schema()
            ic_type = str(type(ic))
            if "<class 'metagpt.actions.action_node" in ic_type:
                # instruct_content 由 AutoNode.create_model_class 生成
                mapping = actionoutout_schema_to_mapping(schema)
                mapping = actionoutput_mapping_to_str(mapping)
                ic_dict = {"class": schema["title"], "mapping": mapping, "value": ic.model_dump()}
            else:
                # 直接继承自 BaseModel
                ic_dict = {"class": schema["title"], "module": ic.__module__, "value": ic.model_dump()}
        return ic_dict

    def __init__(self, content: str = "", **data: Any):
        """
        初始化 Message 对象，允许传入 `content` 以及其他字段参数。

        :param content: 消息的文本内容，默认为空字符串。
        :param data: 其他可选参数，以键值对的形式传入。
        """
        data["content"] = data.get("content", content)
        super().__init__(**data)

    def __setattr__(self, key, val):
        """
        重写 `__setattr__` 方法，以确保某些特定属性在赋值时被转换为字符串格式。

        - `MESSAGE_ROUTE_CAUSE_BY` 和 `MESSAGE_ROUTE_FROM` 的值会转换为字符串。
        - `MESSAGE_ROUTE_TO` 会转换为 `set[str]` 类型。
        - 其他属性保持原样赋值。

        :param key: 要设置的属性名。
        :param val: 要设置的值。
        """
        if key == MESSAGE_ROUTE_CAUSE_BY:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_FROM:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_TO:
            new_val = any_to_str_set(val)
        else:
            new_val = val
        super().__setattr__(key, new_val)

    def __str__(self):
        """
        返回 `Message` 对象的字符串表示形式。
        如果 `instruct_content` 存在，则返回指令内容的 JSON 格式；
        否则返回 `role: content` 形式的字符串。
        """
        if self.instruct_content:
            return f"{self.role}: {self.instruct_content.model_dump()}"
        return f"{self.role}: {self.content}"

    def __repr__(self):
        """
        返回 `Message` 对象的可读字符串表示，等同于 `__str__` 方法。
        """
        return self.__str__()

    def rag_key(self) -> str:
        """
        用于检索增强生成（RAG）的关键字提取方法，返回消息的文本内容。

        :return: 消息文本内容，用于搜索或检索。
        """
        return self.content

    def to_dict(self) -> dict:
        """
        将 `Message` 对象转换为字典格式，适用于 LLM（大语言模型）调用。

        :return: 包含 `role` 和 `content` 的字典。
        """
        return {"role": self.role, "content": self.content}

    def dump(self) -> str:
        """
        将 `Message` 对象转换为 JSON 字符串，排除 `None` 值，并禁止警告信息。

        :return: JSON 格式的字符串表示。
        """
        return self.model_dump_json(exclude_none=True, warnings=False)

    @staticmethod
    @handle_exception(exception_type=JSONDecodeError, default_return=None)
    def load(val):
        """
        从 JSON 字符串解析 `Message` 对象。

        :param val: JSON 格式的字符串。
        :return: 解析成功返回 `Message` 对象，解析失败返回 None。
        """
        try:
            m = json.loads(val)  # 解析 JSON 字符串
            id = m.get("id")  # 获取 `id` 字段（如果存在）
            if "id" in m:
                del m["id"]  # 删除 `id`，避免重复赋值
            msg = Message(**m)  # 使用解析后的数据创建 `Message` 对象
            if id:
                msg.id = id  # 重新赋值 `id`
            return msg
        except JSONDecodeError as err:
            logger.error(f"解析 JSON 失败: {val}, 错误信息: {err}")
        return None

    async def parse_resources(self, llm: "BaseLLM", key_descriptions: Dict[str, str] = None) -> Dict:
        """
        解析消息内容中的资源信息（资源可能包括数据、文档、链接等）。

        `parse_resources` 主要用于 LLM（大语言模型）分析输入文本并提取其中的资源，
        未来可能会迁移到 `context builder` 中。

        :param llm: `BaseLLM` 类的实例，用于执行语言模型的推理。
        :param key_descriptions: （可选）字典，包含资源键及其描述信息。
        :return: 解析出的资源信息，以字典形式返回。

        返回格式：
        ```json
        {
            "resources": [
                {
                    "resource_type": "数据集",
                    "value": "https://example.com/dataset.csv",
                    "description": "用于训练机器学习模型的数据集"
                }
            ],
            "reason": "从文本中提取出的相关资源信息"
        }
        ```
        """
        if not self.content:
            return {}

        # 生成用于 LLM 解析的文本
        content = f"## 原始需求\n```text\n{self.content}\n```\n"

        # 定义返回格式的描述
        return_format = (
            "请以 markdown JSON 格式返回以下内容:\n"
            '- 一个 "resources" 键，包含一个对象列表，每个对象包括:\n'
            '  - 一个 "resource_type" 键，说明资源的类型;\n'
            '  - 一个 "value" 键，包含资源的字符串内容;\n'
            '  - 一个 "description" 键，说明资源的用途;\n'
        )

        # 追加自定义的键描述
        key_descriptions = key_descriptions or {}
        for k, v in key_descriptions.items():
            return_format += f'- 一个 "{k}" 键，包含 {v};\n'
        return_format += '- 一个 "reason" 键，解释资源提取的理由;\n'

        # 组装指令内容
        instructions = ['列出 "原始需求" 中包含的所有资源。', return_format]

        # 调用 LLM 进行解析
        rsp = await llm.aask(msg=content, system_msgs=instructions)

        # 解析 LLM 返回的 JSON 格式的资源信息
        json_data = CodeParser.parse_code(text=rsp, lang="json")
        m = json.loads(json_data)

        # 将解析出的资源转换为 `Resource` 对象
        m["resources"] = [Resource(**i) for i in m.get("resources", [])]

        return m

    def add_metadata(self, key: str, value: str):
        """
        添加元数据（metadata）。

        参数：
        - key (str): 元数据的键。
        - value (str): 元数据的值。

        说明：
        该方法用于给消息对象添加额外的元数据，存储在 `self.metadata` 字典中。
        """
        self.metadata[key] = value

    @staticmethod
    def create_instruct_value(kvs: Dict[str, Any], class_name: str = "") -> BaseModel:
        """
        动态创建一个基于 Pydantic BaseModel 的子类，并使用指定的字典进行初始化。

        参数：
        - kvs (Dict[str, Any]): 用于创建 BaseModel 子类的字典，键为字段名，值为字段对应的数据。
        - class_name (str, 可选): 生成的类的名称，默认为随机生成的唯一标识符。

        返回：
        - BaseModel: 生成的 Pydantic BaseModel 子类实例，包含传入字典的字段和值。

        说明：
        该方法会基于 `kvs` 字典动态创建 Pydantic 模型，并使用 `model_validate` 进行初始化。
        """
        if not class_name:
            class_name = "DM" + uuid.uuid4().hex[0:8]  # 生成随机类名
        dynamic_class = create_model(class_name, **{key: (value.__class__, ...) for key, value in kvs.items()})
        return dynamic_class.model_validate(kvs)

    def is_user_message(self) -> bool:
        """
        判断当前消息是否为用户发送的消息。

        返回：
        - bool: 如果 `role` 等于 `"user"`，返回 `True`，否则返回 `False`。
        """
        return self.role == "user"

    def is_ai_message(self) -> bool:
        """
        判断当前消息是否为 AI 发送的消息。

        返回：
        - bool: 如果 `role` 等于 `"assistant"`，返回 `True`，否则返回 `False`。
        """
        return self.role == "assistant"


class UserMessage(Message):
    """便于支持OpenAI的消息
    Facilitate support for OpenAI messages
    """

    def __init__(self, content: str, **kwargs):
        kwargs.pop("role", None)
        super().__init__(content=content, role="user", **kwargs)


class SystemMessage(Message):
    """便于支持OpenAI的消息
    Facilitate support for OpenAI messages
    """

    def __init__(self, content: str, **kwargs):
        kwargs.pop("role", None)
        super().__init__(content=content, role="system", **kwargs)


class AIMessage(Message):
    """便于支持OpenAI的消息
    Facilitate support for OpenAI messages
    """

    def __init__(self, content: str, **kwargs):
        kwargs.pop("role", None)  # 移除可能存在的 role 参数
        super().__init__(content=content, role="assistant", **kwargs)

    def with_agent(self, name: str):
        """
        设置该消息的代理（Agent）。

        参数：
        - name (str): 代理的名称。

        返回：
        - AIMessage: 返回当前对象，以支持链式调用。
        """
        self.add_metadata(key=AGENT, value=name)
        return self

    @property
    def agent(self) -> str:
        """
        获取该消息所属的代理名称。

        返回：
        - str: 代理的名称，若不存在则返回空字符串。
        """
        return self.metadata.get(AGENT, "")


class Task(BaseModel):
    """
    任务类，表示一个可执行任务。

    属性：
    - task_id (str): 任务 ID，默认为空字符串。
    - dependent_task_ids (list[str]): 该任务的前置任务列表。
    - instruction (str): 任务的具体指令。
    - task_type (str): 任务类型。
    - code (str): 任务相关的代码，默认为空字符串。
    - result (str): 任务执行的结果，默认为空字符串。
    - is_success (bool): 任务是否成功完成，默认为 False。
    - is_finished (bool): 任务是否已经完成，默认为 False。
    - assignee (str): 任务的分配对象，默认为空字符串。

    方法：
    - reset(): 重置任务状态，将 `code`、`result` 清空，并将 `is_success` 和 `is_finished` 设为 `False`。
    - update_task_result(task_result: TaskResult): 更新任务的执行结果，并累加代码和结果内容。
    """
    task_id: str = ""
    dependent_task_ids: list[str] = []  # Tasks prerequisite to this Task
    instruction: str = ""
    task_type: str = ""
    code: str = ""
    result: str = ""
    is_success: bool = False
    is_finished: bool = False
    assignee: str = ""

    def reset(self):
        """
        重置任务状态：
        - 清空 `code` 和 `result`。
        - 将 `is_success` 和 `is_finished` 设为 `False`。
        """
        self.code = ""
        self.result = ""
        self.is_success = False
        self.is_finished = False

    def update_task_result(self, task_result: TaskResult):
        """
        更新任务的执行结果，并累加代码和结果内容。

        参数：
        - task_result (TaskResult): 任务执行的结果对象。

        说明：
        - 任务的 `code` 和 `result` 将累加 `task_result` 中的对应内容。
        - 任务的 `is_success` 将继承 `task_result.is_success` 的值。
        """
        self.code = self.code + "\n" + task_result.code
        self.result = self.result + "\n" + task_result.result
        self.is_success = task_result.is_success


class TaskResult(BaseModel):
    """
    任务执行结果类，表示一个任务的执行结果。

    属性：
    - code (str): 任务执行过程中涉及的代码，默认为空字符串。
    - result (str): 任务执行的最终结果（必填）。
    - is_success (bool): 任务是否成功执行（必填）。
    """

    code: str = ""
    result: str
    is_success: bool


@register_tool(
    include_functions=[
        "append_task",
        "reset_task",
        "replace_task",
        "finish_current_task",
    ]
)

class Plan(BaseModel):
    """Plan 表示朝向目标的一系列任务。"""

    goal: str  # 计划的目标
    context: str = ""  # 计划的上下文信息
    tasks: list[Task] = []  # 任务列表
    task_map: dict[str, Task] = {}  # 任务 ID 到任务对象的映射
    current_task_id: str = ""  # 当前执行的任务 ID

    def _topological_sort(self, tasks: list[Task]):
        """
        对任务进行拓扑排序，确保任务执行顺序满足依赖关系。

        Args:
            tasks (list[Task]): 需要排序的任务列表。

        Returns:
            list[Task]: 排序后的任务列表，符合依赖顺序。
        """
        task_map = {task.task_id: task for task in tasks}
        dependencies = {task.task_id: set(task.dependent_task_ids) for task in tasks}
        sorted_tasks = []
        visited = set()

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dependent_id in dependencies.get(task_id, []):
                visit(dependent_id)
            sorted_tasks.append(task_map[task_id])

        for task in tasks:
            visit(task.task_id)

        return sorted_tasks

    def add_tasks(self, tasks: list[Task]):
        """
        将新任务合并到当前任务计划中，并确保依赖顺序。

        1. 如果当前计划没有任务，则对新任务进行拓扑排序，并直接设置为当前任务列表。
        2. 如果已有任务，则会尽量保留原有任务的前缀部分，并将新任务追加到后面，确保任务的依赖关系。
        3. 更新 `current_task_id` 以指向第一个未完成的任务。

        Args:
            tasks (list[Task]): 需要添加的任务列表（可能是无序的）。
        """
        if not tasks:
            return

        # 对新任务进行拓扑排序，确保正确的依赖顺序
        new_tasks = self._topological_sort(tasks)

        if not self.tasks:
            # 如果当前计划没有任务，则直接设定任务列表
            self.tasks = new_tasks
        else:
            # 计算原任务和新任务的公共前缀长度
            prefix_length = 0
            for old_task, new_task in zip(self.tasks, new_tasks):
                if old_task.task_id != new_task.task_id or old_task.instruction != new_task.instruction:
                    break
                prefix_length += 1

            # 合并公共前缀和新的任务部分
            final_tasks = self.tasks[:prefix_length] + new_tasks[prefix_length:]
            self.tasks = final_tasks

        # 更新当前任务 ID 并刷新任务映射
        self._update_current_task()
        self.task_map = {task.task_id: task for task in self.tasks}


    def reset_task(self, task_id: str):
        """
        重置指定任务，并递归重置所有依赖该任务的任务。

        Args:
            task_id (str): 需要重置的任务 ID。
        """
        if task_id in self.task_map:
            task = self.task_map[task_id]
            task.reset()
            # 递归重置所有依赖于该任务的下游任务
            for dep_task in self.tasks:
                if task_id in dep_task.dependent_task_ids:
                    # FIXME: if LLM generates cyclic tasks, this will result in infinite recursion
                    self.reset_task(dep_task.task_id)

        self._update_current_task()

    def _replace_task(self, new_task: Task):
        """
        替换现有任务，并重置所有依赖该任务的任务。

        Args:
            new_task (Task): 用于替换的任务对象。
        """
        assert new_task.task_id in self.task_map
        # 替换任务
        self.task_map[new_task.task_id] = new_task
        for i, task in enumerate(self.tasks):
            if task.task_id == new_task.task_id:
                self.tasks[i] = new_task
                break

        # 递归重置依赖任务
        for task in self.tasks:
            if new_task.task_id in task.dependent_task_ids:
                self.reset_task(task.task_id)

        self._update_current_task()

    def _append_task(self, new_task: Task):
        """
        追加新任务到当前任务序列的末尾。

        Args:
            new_task (Task): 需要追加的任务对象。
        """
        # 先检查是否已经存在相同的任务 ID
        if self.has_task_id(new_task.task_id):
            logger.warning(
                "任务已存在当前计划中，应该使用 replace_task 进行替换。将覆盖现有任务。"
            )

        # 确保新任务的依赖任务都已存在
        assert all(
            [self.has_task_id(dep_id) for dep_id in new_task.dependent_task_ids]
        ), "新任务包含未知的依赖任务"

        # 由于现有任务不依赖新任务，直接添加到任务序列的末尾
        self.tasks.append(new_task)
        self.task_map[new_task.task_id] = new_task
        self._update_current_task()

    def has_task_id(self, task_id: str) -> bool:
        """
        判断任务 ID 是否已存在于当前任务计划中。

        Args:
            task_id (str): 需要检查的任务 ID。

        Returns:
            bool: 如果任务 ID 存在，则返回 True，否则返回 False。
        """
        return task_id in self.task_map

    def _update_current_task(self):
        """
        更新当前任务 ID，使其指向第一个未完成的任务，并重新整理任务顺序。
        """
        self.tasks = self._topological_sort(self.tasks)
        self.task_map = {task.task_id: task for task in self.tasks}

        current_task_id = ""
        for task in self.tasks:
            if not task.is_finished:
                current_task_id = task.task_id
                break
        self.current_task_id = current_task_id

        # 任务更新后，报告最新的任务列表和当前任务 ID
        TaskReporter().report({"tasks": [i.model_dump() for i in self.tasks], "current_task_id": current_task_id})

    @property
    def current_task(self) -> Task:
        """获取当前需要执行的任务

        Returns:
            Task: 当前需要执行的任务对象，如果没有任务，则返回 None
        """
        return self.task_map.get(self.current_task_id, None)

    def finish_current_task(self):
        """标记当前任务为完成，并自动更新当前任务 ID 为下一个待执行任务"""
        if self.current_task_id:
            self.current_task.is_finished = True
            self._update_current_task()  # 更新当前任务为下一个待执行任务

    def finish_all_tasks(self):
        """标记所有任务为完成状态"""
        while self.current_task:
            self.finish_current_task()

    def is_plan_finished(self) -> bool:
        """检查所有任务是否已完成

        Returns:
            bool: 如果所有任务均已完成，则返回 True，否则返回 False
        """
        return all(task.is_finished for task in self.tasks)

    def get_finished_tasks(self) -> list[Task]:
        """获取所有已完成的任务，按照拓扑排序后的顺序返回

        Returns:
            list[Task]: 已完成任务的列表
        """
        return [task for task in self.tasks if task.is_finished]

    def append_task(
        self, task_id: str, dependent_task_ids: list[str], instruction: str, assignee: str, task_type: str = ""
    ):
        """
        向任务序列的末尾追加一个新任务。
        如果 `dependent_task_ids` 非空，则新任务会依赖于列表中的任务。
        任务的 `assignee` 代表该任务分配的角色名称。

        Args:
            task_id (str): 任务的唯一 ID
            dependent_task_ids (list[str]): 任务依赖的其他任务 ID 列表
            instruction (str): 任务指令或描述
            assignee (str): 任务分配的执行者（角色名称）
            task_type (str, optional): 任务类型，默认为空字符串

        Returns:
            None
        """
        new_task = Task(
            task_id=task_id,
            dependent_task_ids=dependent_task_ids,
            instruction=instruction,
            assignee=assignee,
            task_type=task_type,
        )
        return self._append_task(new_task)

    def replace_task(self, task_id: str, new_dependent_task_ids: list[str], new_instruction: str, new_assignee: str):
        """
        替换已有的任务（可以是当前任务）。
        该方法会用新的任务替换原任务，并重置所有依赖于该任务的后续任务。

        Args:
            task_id (str): 需要被替换的任务 ID
            new_dependent_task_ids (list[str]): 新任务的依赖任务 ID 列表
            new_instruction (str): 新任务的指令或描述
            new_assignee (str): 新任务的分配执行者（角色名称）

        Returns:
            None
        """
        new_task = Task(
            task_id=task_id,
            dependent_task_ids=new_dependent_task_ids,
            instruction=new_instruction,
            assignee=new_assignee,
        )
        return self._replace_task(new_task)


class MessageQueue(BaseModel):
    """支持异步更新的消息队列。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _queue: Queue = PrivateAttr(default_factory=Queue)  # 内部消息队列

    def pop(self) -> Message | None:
        """从队列中取出一个消息。

        Returns:
            Message | None: 若队列非空，则返回队首消息；否则返回 None。
        """
        try:
            item = self._queue.get_nowait()
            if item:
                self._queue.task_done()
            return item
        except QueueEmpty:
            return None

    def pop_all(self) -> List[Message]:
        """取出队列中的所有消息。

        Returns:
            List[Message]: 队列中的所有消息列表。
        """
        ret = []
        while True:
            msg = self.pop()
            if not msg:
                break
            ret.append(msg)
        return ret

    def push(self, msg: Message):
        """将消息加入队列。

        Args:
            msg (Message): 要加入队列的消息对象。
        """
        self._queue.put_nowait(msg)

    def empty(self) -> bool:
        """检查队列是否为空。

        Returns:
            bool: 如果队列为空，则返回 True，否则返回 False。
        """
        return self._queue.empty()


    async def dump(self) -> str:
        """将 `MessageQueue` 对象转换为 JSON 字符串。

        Returns:
            str: 序列化后的 JSON 字符串，如果队列为空，则返回 "[]"
        """
        if self.empty():
            return "[]"

        lst = []  # 用于存储消息的 JSON 数据
        msgs = []  # 临时存储已取出的消息
        try:
            while True:
                item = await wait_for(self._queue.get(), timeout=1.0)  # 异步等待消息
                if item is None:
                    break
                msgs.append(item)
                lst.append(item.dump())
                self._queue.task_done()
        except asyncio.TimeoutError:
            logger.debug("队列为空，退出读取...")
        finally:
            # 由于 dump 过程中取出了消息，这里需要重新放回队列，确保队列内容不变
            for m in msgs:
                self._queue.put_nowait(m)

        return json.dumps(lst, ensure_ascii=False)

    @staticmethod
    def load(data) -> "MessageQueue":
        """从 JSON 字符串恢复 `MessageQueue` 对象。

        Args:
            data (str): JSON 格式的消息队列数据。

        Returns:
            MessageQueue: 解析后的消息队列对象。
        """
        queue = MessageQueue()
        try:
            lst = json.loads(data)
            for i in lst:
                msg = Message.load(i)  # 将 JSON 数据转换回 Message 对象
                queue.push(msg)
        except JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {data}, 错误: {e}")

        return queue


# 定义一个泛型类型变量，用于限制泛型类型必须是 BaseModel 的子类
T = TypeVar("T", bound="BaseModel")


class BaseContext(BaseModel, ABC):
    """上下文基类，所有具体的上下文类都继承自该类。"""

    @classmethod
    @handle_exception  # 异常处理装饰器，确保解析异常时不会影响程序运行
    def loads(cls: Type[T], val: str) -> Optional[T]:
        """从 JSON 字符串加载对象实例。

        Args:
            val (str): JSON 格式的字符串。

        Returns:
            Optional[T]: 解析后的对象实例，如果解析失败返回 None。
        """
        i = json.loads(val)
        return cls(**i)


class CodingContext(BaseContext):
    """代码开发相关的上下文，包括设计文档、任务文档、代码文档等。"""

    filename: str  # 代码文件名
    design_doc: Optional[Document] = None  # 设计文档
    task_doc: Optional[Document] = None  # 任务文档
    code_doc: Optional[Document] = None  # 代码文档
    code_plan_and_change_doc: Optional[Document] = None  # 代码计划与变更文档


class TestingContext(BaseContext):
    """测试相关的上下文，包括代码文件和测试文档。"""

    filename: str  # 代码文件名
    code_doc: Document  # 代码文档
    test_doc: Optional[Document] = None  # 测试文档（可选）


class RunCodeContext(BaseContext):
    """代码运行的上下文信息，包括运行模式、代码内容、命令参数等。"""

    mode: str = "script"  # 运行模式（默认为脚本模式）
    code: Optional[str] = None  # 代码内容
    code_filename: str = ""  # 代码文件名
    test_code: Optional[str] = None  # 测试代码内容
    test_filename: str = ""  # 测试文件名
    command: List[str] = Field(default_factory=list)  # 运行命令参数列表
    working_directory: str = ""  # 运行目录
    additional_python_paths: List[str] = Field(default_factory=list)  # 额外的 Python 路径
    output_filename: Optional[str] = None  # 运行结果输出文件名
    output: Optional[str] = None  # 运行结果输出内容

class RunCodeResult(BaseContext):
    """代码运行结果，包括摘要、标准输出和错误输出。"""

    summary: str  # 运行结果摘要
    stdout: str  # 标准输出内容
    stderr: str  # 标准错误输出内容

class CodeSummarizeContext(BaseModel):
    """代码总结的上下文，包括设计文档、任务文档和相关代码文件。"""

    design_filename: str = ""  # 设计文档文件名
    task_filename: str = ""  # 任务文档文件名
    codes_filenames: List[str] = Field(default_factory=list)  # 代码文件列表
    reason: str = ""  # 代码总结的原因

    @staticmethod
    def loads(filenames: List) -> "CodeSummarizeContext":
        """根据文件路径列表加载代码总结上下文。

        Args:
            filenames (List): 文件路径列表。

        Returns:
            CodeSummarizeContext: 解析后的代码总结上下文。
        """
        ctx = CodeSummarizeContext()
        for filename in filenames:
            if Path(filename).is_relative_to(SYSTEM_DESIGN_FILE_REPO):
                ctx.design_filename = str(filename)
                continue
            if Path(filename).is_relative_to(TASK_FILE_REPO):
                ctx.task_filename = str(filename)
                continue
        return ctx

    def __hash__(self):
        """计算对象的哈希值，使其可用于集合或字典键。"""
        return hash((self.design_filename, self.task_filename))


class CodePlanAndChangeContext(BaseModel):
    """代码计划与变更的上下文，包括需求、问题、相关文档等。"""

    requirement: str = ""  # 需求描述
    issue: str = ""  # 相关问题描述
    prd_filename: str = ""  # PRD 文档文件名
    design_filename: str = ""  # 设计文档文件名
    task_filename: str = ""  # 任务文档文件名



# mermaid class view
class UMLClassMeta(BaseModel):
    """UML 类的基本元数据，包括名称和可见性。"""

    name: str = ""  # 类名或属性/方法名
    visibility: str = ""  # 可见性（public/private/protected）

    @staticmethod
    def name_to_visibility(name: str) -> str:
        """根据命名规则推导 UML 可见性符号。

        Args:
            name (str): 方法或属性名。

        Returns:
            str: UML 可见性符号（+ 表示 public，- 表示 private，# 表示 protected）。
        """
        if name == "__init__":
            return "+"
        if name.startswith("__"):
            return "-"
        elif name.startswith("_"):
            return "#"
        return "+"


class UMLClassAttribute(UMLClassMeta):
    """UML 类的属性信息，包括类型和默认值。"""

    value_type: str = ""  # 属性类型
    default_value: str = ""  # 默认值

    def get_mermaid(self, align=1) -> str:
        """生成 Mermaid.js 兼容的 UML 属性描述。

        Args:
            align (int, optional): 缩进级别，默认为 1。

        Returns:
            str: Mermaid.js 兼容的 UML 属性描述字符串。
        """
        content = "".join(["\t" for _ in range(align)]) + self.visibility
        if self.value_type:
            content += self.value_type.replace(" ", "") + " "
        name = self.name.split(":", 1)[1] if ":" in self.name else self.name
        content += name
        if self.default_value:
            content += "="
            if self.value_type not in ["str", "string", "String"]:
                content += self.default_value
            else:
                content += '"' + self.default_value.replace('"', "") + '"'
        return content


class UMLClassMethod(UMLClassMeta):
    """UML 类的方法信息，包括参数和返回值类型。"""

    args: List[UMLClassAttribute] = Field(default_factory=list)  # 方法参数列表
    return_type: str = ""  # 返回值类型

    def get_mermaid(self, align=1) -> str:
        """生成 Mermaid.js 兼容的 UML 方法描述。

        Args:
            align (int, optional): 缩进级别，默认为 1。

        Returns:
            str: Mermaid.js 兼容的 UML 方法描述字符串。
        """
        content = "".join(["\t" for _ in range(align)]) + self.visibility
        name = self.name.split(":", 1)[1] if ":" in self.name else self.name
        content += name + "(" + ",".join([v.get_mermaid(align=0) for v in self.args]) + ")"
        if self.return_type:
            content += " " + self.return_type.replace(" ", "")
        return content


class UMLClassView(UMLClassMeta):
    attributes: List[UMLClassAttribute] = Field(default_factory=list)  # 类的属性列表
    methods: List[UMLClassMethod] = Field(default_factory=list)  # 类的方法列表

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + "class " + self.name + "{\n"  # 构建 mermaid 格式的类图
        for v in self.attributes:
            content += v.get_mermaid(align=align + 1) + "\n"  # 添加属性到类图
        for v in self.methods:
            content += v.get_mermaid(align=align + 1) + "\n"  # 添加方法到类图
        content += "".join(["\t" for i in range(align)]) + "}\n"  # 关闭类定义
        return content

    @classmethod
    def load_dot_class_info(cls, dot_class_info: DotClassInfo) -> UMLClassView:
        """
        从 DotClassInfo 加载类信息并返回 UMLClassView 实例。

        Args:
            dot_class_info: 包含类信息的 DotClassInfo 实例。

        Returns:
            UMLClassView: 包含类信息的 UMLClassView 实例。
        """
        visibility = UMLClassView.name_to_visibility(dot_class_info.name)  # 获取可见性
        class_view = cls(name=dot_class_info.name, visibility=visibility)  # 创建 UML 类视图
        for i in dot_class_info.attributes.values():
            visibility = UMLClassAttribute.name_to_visibility(i.name)  # 获取属性的可见性
            attr = UMLClassAttribute(name=i.name, visibility=visibility, value_type=i.type_,
                                     default_value=i.default_)  # 创建属性
            class_view.attributes.append(attr)  # 将属性添加到类视图
        for i in dot_class_info.methods.values():
            visibility = UMLClassMethod.name_to_visibility(i.name)  # 获取方法的可见性
            method = UMLClassMethod(name=i.name, visibility=visibility, return_type=i.return_args.type_)  # 创建方法
            for j in i.args:
                arg = UMLClassAttribute(name=j.name, value_type=j.type_, default_value=j.default_)  # 创建方法参数
                method.args.append(arg)  # 将参数添加到方法中
            method.return_type = i.return_args.type_  # 设置方法返回类型
            class_view.methods.append(method)  # 将方法添加到类视图
        return class_view


class BaseEnum(Enum):
    """枚举的基类。"""

    def __new__(cls, value, desc=None):
        """
        构造枚举成员实例。

        Args:
            cls: 类。
            value: 枚举成员的值。
            desc: 枚举成员的描述，默认为 None。
        """
        if issubclass(cls, str):
            obj = str.__new__(cls, value)  # 创建字符串类型的枚举实例
        elif issubclass(cls, int):
            obj = int.__new__(cls, value)  # 创建整数类型的枚举实例
        else:
            obj = object.__new__(cls)  # 创建其他类型的枚举实例
        obj._value_ = value  # 设置枚举成员的值
        obj.desc = desc  # 设置枚举成员的描述
        return obj


class LongTermMemoryItem(BaseModel):
    message: Message  # 消息内容
    created_at: Optional[float] = Field(default_factory=time.time)  # 创建时间（默认是当前时间）

    def rag_key(self) -> str:
        """生成 RAG（检索增强生成）键"""
        return self.message.content  # 返回消息内容作为 RAG 键
