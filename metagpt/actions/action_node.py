#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/11 18:45
@Author  : alexanderwu
@File    : action_node.py

NOTE: You should use typing.List instead of list to do type annotation. Because in the markdown extraction process,
  we can use typing to extract the type of the node, but we cannot use built-in list to extract.
"""
import json
import re
import typing
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model, model_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions.action_outcls_registry import register_action_outcls
from metagpt.const import MARKDOWN_TITLE_PREFIX, USE_CONFIG_TIMEOUT
from metagpt.exp_pool import exp_cache
from metagpt.exp_pool.serializers import ActionNodeSerializer
from metagpt.llm import BaseLLM
from metagpt.logs import logger
from metagpt.provider.postprocess.llm_output_postprocess import llm_output_postprocess
from metagpt.utils.common import OutputParser, general_after_log
from metagpt.utils.human_interaction import HumanInteraction
from metagpt.utils.sanitize import sanitize


class ReviewMode(Enum):
    HUMAN = "human"
    AUTO = "auto"


class ReviseMode(Enum):
    HUMAN = "human"  # human revise
    HUMAN_REVIEW = "human_review"  # human-review and auto-revise
    AUTO = "auto"  # auto-review and auto-revise


TAG = "CONTENT"


class FillMode(Enum):
    CODE_FILL = "code_fill"
    XML_FILL = "xml_fill"
    SINGLE_FILL = "single_fill"


LANGUAGE_CONSTRAINT = "Language: Please use the same language as Human INPUT."
FORMAT_CONSTRAINT = f"Format: output wrapped inside [{TAG}][/{TAG}] like format example, nothing else."


SIMPLE_TEMPLATE = """
## context
{context}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow instructions of nodes, generate output and make sure it follows the format example.
"""

REVIEW_TEMPLATE = """
## context
Compare the key's value of nodes_output and the corresponding requirements one by one. If a key's value that does not match the requirement is found, provide the comment content on how to modify it. No output is required for matching keys.

### nodes_output
{nodes_output}

-----

## format example
[{tag}]
{{
    "key1": "comment1",
    "key2": "comment2",
    "keyn": "commentn"
}}
[/{tag}]

## nodes: "<node>: <type>  # <instruction>"
- key1: <class \'str\'> # the first key name of mismatch key
- key2: <class \'str\'> # the second key name of mismatch key
- keyn: <class \'str\'> # the last key name of mismatch key

## constraint
{constraint}

## action
Follow format example's {prompt_schema} format, generate output and make sure it follows the format example.
"""

REVISE_TEMPLATE = """
## context
change the nodes_output key's value to meet its comment and no need to add extra comment.

### nodes_output
{nodes_output}

-----

## format example
{example}

## nodes: "<node>: <type>  # <instruction>"
{instruction}

## constraint
{constraint}

## action
Follow format example's {prompt_schema} format, generate output and make sure it follows the format example.
"""


def dict_to_markdown(d, prefix=MARKDOWN_TITLE_PREFIX, kv_sep="\n", postfix="\n"):
    """将字典转换为Markdown格式"""
    markdown_str = ""
    for key, value in d.items():
        markdown_str += f"{prefix}{key}{kv_sep}{value}{postfix}"
    return markdown_str


class ActionNode:
    """ActionNode：表示一个动作节点，是一个树结构的节点。"""

    schema: str  # 存储原始格式（json/markdown等）

    # 动作上下文
    context: str  # 所有上下文信息，包含所有必要的信息
    llm: BaseLLM  # 与 LLM 交互的接口
    children: dict[str, "ActionNode"]  # 孩子节点的字典

    # 动作输入
    key: str  # 产品需求 / 文件列表 / 代码等
    func: typing.Callable  # 与节点相关联的函数或 LLM 调用
    params: Dict[str, Type]  # 输入参数字典，键为参数名，值为参数类型
    expected_type: Type  # 预期的类型，例如 str / int / float 等
    instruction: str  # 需要遵循的指令
    example: Any  # 供上下文学习的示例

    # 动作输出
    content: str  # 动作的内容输出
    instruct_content: BaseModel  # 指令内容

    # ActionGraph 的相关字段
    prevs: List["ActionNode"]  # 前置节点
    nexts: List["ActionNode"]  # 后置节点

    def __init__(
        self,
        key: str,
        expected_type: Type,
        instruction: str,
        example: Any,
        content: str = "",
        children: dict[str, "ActionNode"] = None,
        schema: str = "",
    ):
        """初始化一个动作节点"""
        self.key = key
        self.expected_type = expected_type
        self.instruction = instruction
        self.example = example
        self.content = content
        self.children = children if children is not None else {}
        self.schema = schema
        self.prevs = []
        self.nexts = []

    def __str__(self):
        """返回节点的字符串表示"""
        return (
            f"{self.key}, {repr(self.expected_type)}, {self.instruction}, {self.example} "
            f", {self.content}, {self.children}"
        )

    def __repr__(self):
        """返回节点的字符串表示"""
        return self.__str__()

    def add_prev(self, node: "ActionNode"):
        """增加前置节点"""
        self.prevs.append(node)

    def add_next(self, node: "ActionNode"):
        """增加后置节点"""
        self.nexts.append(node)

    def add_child(self, node: "ActionNode"):
        """增加子节点"""
        self.children[node.key] = node

    def get_child(self, key: str) -> Union["ActionNode", None]:
        """根据键获取子节点"""
        return self.children.get(key, None)

    def add_children(self, nodes: List["ActionNode"]):
        """批量增加子节点"""
        for node in nodes:
            self.add_child(node)

    @classmethod
    def from_children(cls, key, nodes: List["ActionNode"]):
        """直接从一系列子节点初始化"""
        obj = cls(key, str, "", "")
        obj.add_children(nodes)
        return obj

    def _get_children_mapping(self, exclude=None) -> Dict[str, Any]:
        """获取子节点的映射字典，支持多级结构"""
        exclude = exclude or []

        def _get_mapping(node: "ActionNode") -> Dict[str, Any]:
            mapping = {}
            for key, child in node.children.items():
                if key in exclude:
                    continue
                # 对于嵌套的子节点，递归获取映射
                if child.children:
                    mapping[key] = _get_mapping(child)
                else:
                    mapping[key] = (child.expected_type, Field(default=child.example, description=child.instruction))
            return mapping

        return _get_mapping(self)

    def _get_self_mapping(self) -> Dict[str, Tuple[Type, Any]]:
        """获取自身节点的映射"""
        return {self.key: (self.expected_type, ...)}

    def get_mapping(self, mode="children", exclude=None) -> Dict[str, Tuple[Type, Any]]:
        """根据模式获取映射字典"""
        if mode == "children" or (mode == "auto" and self.children):
            return self._get_children_mapping(exclude=exclude)
        return {} if exclude and self.key in exclude else self._get_self_mapping()

    @classmethod
    @register_action_outcls
    def create_model_class(cls, class_name: str, mapping: Dict[str, Tuple[Type, Any]]):
        """基于 pydantic v2 的模型动态生成，用来检验结果类型的正确性"""

        def check_fields(cls, values):
            """检查字段是否完整和正确"""
            all_fields = set(mapping.keys())
            required_fields = set()
            for k, v in mapping.items():
                type_v, field_info = v
                if ActionNode.is_optional_type(type_v):
                    continue
                required_fields.add(k)

            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f"Missing fields: {missing_fields}")

            unrecognized_fields = set(values.keys()) - all_fields
            if unrecognized_fields:
                logger.warning(f"Unrecognized fields: {unrecognized_fields}")
            return values

        validators = {"check_missing_fields_validator": model_validator(mode="before")(check_fields)}

        new_fields = {}
        for field_name, field_value in mapping.items():
            if isinstance(field_value, dict):
                # 对于嵌套结构，递归创建模型类
                nested_class_name = f"{class_name}_{field_name}"
                nested_class = cls.create_model_class(nested_class_name, field_value)
                new_fields[field_name] = (nested_class, ...)
            else:
                new_fields[field_name] = field_value

        new_class = create_model(class_name, __validators__=validators, **new_fields)
        return new_class

    def create_class(self, mode: str = "auto", class_name: str = None, exclude=None):
        """根据节点生成对应的模型类"""
        class_name = class_name if class_name else f"{self.key}_AN"
        mapping = self.get_mapping(mode=mode, exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def _create_children_class(self, exclude=None):
        """批量生成子节点对应的模型类"""
        class_name = f"{self.key}_AN"
        mapping = self.get_mapping(mode="children", exclude=exclude)
        return self.create_model_class(class_name, mapping)

    def to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:
        """将当前节点与子节点都按照node: format的格式组织成字典"""
        nodes = self._to_dict(format_func=format_func, mode=mode, exclude=exclude)
        if not isinstance(nodes, dict):
            nodes = {self.key: nodes}
        return nodes

    def _to_dict(self, format_func=None, mode="auto", exclude=None) -> Dict:
        """将当前节点与子节点都按照node: format的格式组织成字典"""

        # 如果没有提供格式化函数，则使用默认的格式化函数
        if format_func is None:
            format_func = lambda node: node.instruction

        # 使用提供的格式化函数来格式化当前节点的值
        formatted_value = format_func(self)

        # 创建当前节点的键值对
        if (mode == "children" or mode == "auto") and self.children:
            node_value = {}
        else:
            node_value = formatted_value

        if mode == "root":
            return {self.key: node_value}

        # 递归处理子节点
        exclude = exclude or []
        for child_key, child_node in self.children.items():
            if child_key in exclude:
                continue
            # 递归调用 to_dict 方法并更新节点字典
            child_dict = child_node._to_dict(format_func, mode, exclude)
            node_value[child_key] = child_dict

        return node_value

    def update_instruct_content(self, incre_data: dict[str, Any]):
        """
        更新指令内容，将增量数据合并到原始指令内容中。

        参数:
            incre_data (dict[str, Any]): 要合并到原始指令内容的增量数据。

        更新后的指令内容将被重新创建并赋值给 self.instruct_content。
        """
        assert self.instruct_content
        origin_sc_dict = self.instruct_content.model_dump()
        origin_sc_dict.update(incre_data)
        output_class = self.create_class()
        self.instruct_content = output_class(**origin_sc_dict)

    def keys(self, mode: str = "auto") -> list:
        """
        获取当前节点的键，根据模式返回不同的键列表。

        参数:
            mode (str): 模式，默认为 "auto"。可选值有 "children", "root"。

        返回:
            list: 当前节点的键列表。
        """
        if mode == "children" or (mode == "auto" and self.children):
            keys = []
        else:
            keys = [self.key]
        if mode == "root":
            return keys

        for _, child_node in self.children.items():
            keys.append(child_node.key)
        return keys

    def compile_to(self, i: Dict, schema, kv_sep) -> str:
        """
        将字典编译为不同格式的字符串（JSON，Markdown等）。

        参数:
            i (Dict): 要编译的字典。
            schema (str): 编译格式，支持 "json" 或 "markdown"。
            kv_sep (str): 键值分隔符。

        返回:
            str: 编译后的字符串。
        """
        if schema == "json":
            return json.dumps(i, indent=4, ensure_ascii=False)
        elif schema == "markdown":
            return dict_to_markdown(i, kv_sep=kv_sep)
        else:
            return str(i)

    def tagging(self, text, schema, tag="") -> str:
        """
        给文本添加标签。

        参数:
            text (str): 要添加标签的文本。
            schema (str): 使用的模式。
            tag (str): 要添加的标签，默认为空。

        返回:
            str: 添加标签后的文本。
        """
        if not tag:
            return text
        return f"[{tag}]\n{text}\n[/{tag}]"

    def _compile_f(self, schema, mode, tag, format_func, kv_sep, exclude=None) -> str:
        """
        编译格式化内容，并为其添加标签。

        参数:
            schema (str): 编译格式，支持 "json", "markdown"。
            mode (str): 模式，决定哪些节点会被编译。
            tag (str): 标签。
            format_func (callable): 格式化函数。
            kv_sep (str): 键值分隔符。
            exclude (list): 要排除的字段列表。

        返回:
            str: 编译后的文本。
        """
        nodes = self.to_dict(format_func=format_func, mode=mode, exclude=exclude)
        text = self.compile_to(nodes, schema, kv_sep)
        return self.tagging(text, schema, tag)

    def compile_instruction(self, schema="markdown", mode="children", tag="", exclude=None) -> str:
        """
        编译指令部分，支持不同格式（JSON，Markdown等）。

        参数:
            schema (str): 编译格式，支持 "json" 或 "markdown"。
            mode (str): 模式，决定编译哪些节点。
            tag (str): 标签。
            exclude (list): 要排除的字段列表。

        返回:
            str: 编译后的指令部分。
        """
        format_func = lambda i: f"{i.expected_type}  # {i.instruction}"
        return self._compile_f(schema, mode, tag, format_func, kv_sep=": ", exclude=exclude)

    def compile_example(self, schema="json", mode="children", tag="", exclude=None) -> str:
        """
        编译示例部分，支持不同格式（JSON，Markdown等）。

        参数:
            schema (str): 编译格式，支持 "json" 或 "markdown"。
            mode (str): 模式，决定编译哪些节点。
            tag (str): 标签。
            exclude (list): 要排除的字段列表。

        返回:
            str: 编译后的示例部分。
        """
        format_func = lambda i: i.example
        return self._compile_f(schema, mode, tag, format_func, kv_sep="\n", exclude=exclude)

    def compile(self, context, schema="json", mode="children", template=SIMPLE_TEMPLATE, exclude=[]) -> str:
        """
        编译整个内容，根据模式和格式返回不同的编译结果。

        参数:
            context (str): 上下文信息。
            schema (str): 编译格式，支持 "raw", "json", "markdown"。
            mode (str): 模式，决定编译哪些节点（"all", "root", "children"）。
            template (str): 用于格式化的模板。
            exclude (list): 要排除的字段列表。

        返回:
            str: 编译后的完整内容。
        """
        if schema == "raw":
            return f"{context}\n\n## Actions\n{LANGUAGE_CONSTRAINT}\n{self.instruction}"

        instruction = self.compile_instruction(schema="markdown", mode=mode, exclude=exclude)
        example = self.compile_example(schema=schema, tag=TAG, mode=mode, exclude=exclude)
        constraints = [LANGUAGE_CONSTRAINT, FORMAT_CONSTRAINT]
        constraint = "\n".join(constraints)

        prompt = template.format(
            context=context,
            example=example,
            instruction=instruction,
            constraint=constraint,
        )
        return prompt

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _aask_v1(
        self,
        prompt: str,
        output_class_name: str,
        output_data_mapping: dict,
        images: Optional[Union[str, list[str]]] = None,
        system_msgs: Optional[list[str]] = None,
        schema="markdown",  # 兼容原始格式
        timeout=USE_CONFIG_TIMEOUT,
    ) -> (str, BaseModel):
        """
        使用 ActionOutput 包装 _aask 的输出。

        参数:
            prompt (str): 提示信息。
            output_class_name (str): 输出类的名称。
            output_data_mapping (dict): 输出数据的映射。
            images (Optional[Union[str, list[str]]]): 可选的图像数据。
            system_msgs (Optional[list[str]]): 系统消息。
            schema (str): 格式（支持 "json" 或 "markdown"）。
            timeout (int): 超时时间。

        返回:
            (str, BaseModel): 返回原始内容和解析后的指令内容。
        """
        content = await self.llm.aask(prompt, system_msgs, images=images, timeout=timeout)
        logger.debug(f"llm raw output:\n{content}")
        output_class = self.create_model_class(output_class_name, output_data_mapping)

        if schema == "json":
            parsed_data = llm_output_postprocess(
                output=content, schema=output_class.model_json_schema(), req_key=f"[/{TAG}]"
            )
        else:  # 使用 markdown 解析器
            parsed_data = OutputParser.parse_data_with_mapping(content, output_data_mapping)

        logger.debug(f"parsed_data:\n{parsed_data}")
        instruct_content = output_class(**parsed_data)
        return content, instruct_content

    def get(self, key):
        """
        获取指定字段的值。

        参数:
            key (str): 要获取的字段名。

        返回:
            Any: 字段的值。
        """
        return self.instruct_content.model_dump()[key]

    def set_recursive(self, name, value):
        """
        递归设置指定字段的值。

        参数:
            name (str): 字段名。
            value (Any): 要设置的值。
        """
        setattr(self, name, value)
        for _, i in self.children.items():
            i.set_recursive(name, value)

    def set_llm(self, llm):
        """
        设置 LLM（大语言模型）实例。

        参数:
            llm: LLM 实例。
        """
        self.set_recursive("llm", llm)

    def set_context(self, context):
        """
        设置上下文。

        参数:
            context: 上下文信息。
        """
        self.set_recursive("context", context)

    async def simple_fill(
        self, schema, mode, images: Optional[Union[str, list[str]]] = None, timeout=USE_CONFIG_TIMEOUT, exclude=None
    ):
        """
        简单填充，使用 LLM 填充内容。

        参数:
            schema (str): 格式，支持 "json", "markdown"。
            mode (str): 模式，决定填充哪些字段。
            images (Optional[Union[str, list[str]]]): 可选的图像数据。
            timeout (int): 超时时间。
            exclude (list): 要排除的字段列表。

        返回:
            self: 填充后的当前对象。
        """
        prompt = self.compile(context=self.context, schema=schema, mode=mode, exclude=exclude)
        if schema != "raw":
            mapping = self.get_mapping(mode, exclude=exclude)
            class_name = f"{self.key}_AN"
            content, scontent = await self._aask_v1(
                prompt, class_name, mapping, images=images, schema=schema, timeout=timeout
            )
            self.content = content
            self.instruct_content = scontent
        else:
            self.content = await self.llm.aask(prompt)
            self.instruct_content = None

        return self

    def get_field_name(self):
        """
        获取与此 ActionNode 关联的 Pydantic 模型的字段名称。
        """
        model_class = self.create_class()
        fields = model_class.model_fields

        # 假设模型中只有一个字段
        if len(fields) == 1:
            return next(iter(fields))

        # 如果有多个字段，可能需要使用 self.key 来找到正确的字段
        return self.key

    def get_field_names(self):
        """
        获取与此 ActionNode 的 Pydantic 模型关联的所有字段名称。
        """
        model_class = self.create_class()
        return model_class.model_fields.keys()

    def get_field_types(self):
        """
        获取与此 ActionNode 的 Pydantic 模型关联的字段类型。
        """
        model_class = self.create_class()
        return {field_name: field.annotation for field_name, field in model_class.model_fields.items()}

    def xml_compile(self, context):
        """
        编译提示词，将其转换为更易于模型理解的 XML 格式。
        """
        field_names = self.get_field_names()
        # 使用字段名称构建示例
        examples = []
        for field_name in field_names:
            examples.append(f"<{field_name}>content</{field_name}>")

        # 将所有示例连接成一个字符串
        example_str = "\n".join(examples)
        # 将示例添加到上下文中
        context += f"""
### 响应格式（必须严格遵循）：所有内容必须包含在给定的 XML 标签中，确保每个开始标签 <tag> 都有对应的结束标签 </tag>，没有不完整或自闭合的标签。\n
{example_str}
"""
        return context

    async def code_fill(
            self, context: str, function_name: Optional[str] = None, timeout: int = USE_CONFIG_TIMEOUT
    ) -> Dict[str, str]:
        """
        使用 ``` ``` 填充代码块
        """
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, timeout=timeout)
        extracted_code = sanitize(code=content, entrypoint=function_name)
        result = {field_name: extracted_code}
        return result

    async def single_fill(self, context: str, images: Optional[Union[str, list[str]]] = None) -> Dict[str, str]:
        field_name = self.get_field_name()
        prompt = context
        content = await self.llm.aask(prompt, images=images)
        result = {field_name: content}
        return result

    async def xml_fill(self, context: str, images: Optional[Union[str, list[str]]] = None) -> Dict[str, Any]:
        """
        使用 XML 标签填充上下文，并根据字段类型进行转换，包括字符串、整数、布尔值、列表和字典类型
        """
        field_names = self.get_field_names()
        field_types = self.get_field_types()

        extracted_data: Dict[str, Any] = {}
        content = await self.llm.aask(context, images=images)

        for field_name in field_names:
            pattern = rf"<{field_name}>(.*?)</{field_name}>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                raw_value = match.group(1).strip()
                field_type = field_types.get(field_name)

                if field_type == str:
                    extracted_data[field_name] = raw_value
                elif field_type == int:
                    try:
                        extracted_data[field_name] = int(raw_value)
                    except ValueError:
                        extracted_data[field_name] = 0  # 或者其他默认值
                elif field_type == bool:
                    extracted_data[field_name] = raw_value.lower() in ("true", "yes", "1", "on", "True")
                elif field_type == list:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], list):
                            raise ValueError
                    except:
                        extracted_data[field_name] = []  # 默认空列表
                elif field_type == dict:
                    try:
                        extracted_data[field_name] = eval(raw_value)
                        if not isinstance(extracted_data[field_name], dict):
                            raise ValueError
                    except:
                        extracted_data[field_name] = {}  # 默认空字典

        return extracted_data

    @exp_cache(serializer=ActionNodeSerializer())  # 使用缓存装饰器，序列化使用 ActionNodeSerializer
    async def fill(
            self,# 异步方法，填充节点内容
            *,
            req,  # 请求参数，包含填充节点所需的所有信息
            llm,  # 预定义的语言模型（LLM）
            schema="json",  # 输出格式，默认为json
            mode="auto",  # 填充模式，默认为auto
            strgy="simple",  # 填充策略，默认为simple
            images: Optional[Union[str, list[str]]] = None,  # 图片的URL或base64编码，可选参数
            timeout=USE_CONFIG_TIMEOUT,  # 请求超时设置
            exclude=[],  # 排除的ActionNode键
            function_name: str = None,  # 函数名称
    ):
        """ 填充节点内容的函数

        :param req: 填充节点所需的请求参数
        :param llm: 大型语言模型
        :param schema: 输出格式，可以是json, markdown等
        :param mode: 填充模式（auto、children、root）
        :param strgy: 填充策略（simple、complex）
        :param images: 图片的URL或base64
        :param timeout: 超时时间
        :param exclude: 排除的ActionNode键
        :return: 填充后的节点内容
        """
        self.set_llm(llm)  # 设置LLM
        self.set_context(req)  # 设置上下文

        if self.schema:  # 如果类中有指定schema，则使用类中的schema
            schema = self.schema

        # 根据mode的不同选择不同的填充方式
        if mode == FillMode.CODE_FILL.value:  # 如果是代码填充
            result = await self.code_fill(context, function_name, timeout)  # 填充代码
            self.instruct_content = self.create_class()(**result)  # 创建类并设置指令内容
            return self

        elif mode == FillMode.XML_FILL.value:  # 如果是XML填充
            context = self.xml_compile(context=self.context)  # 编译XML上下文
            result = await self.xml_fill(context, images=images)  # 填充XML内容
            self.instruct_content = self.create_class()(**result)  # 创建类并设置指令内容
            return self

        elif mode == FillMode.SINGLE_FILL.value:  # 如果是单次填充
            result = await self.single_fill(context, images=images)  # 单次填充
            self.instruct_content = self.create_class()(**result)  # 创建类并设置指令内容
            return self

        if strgy == "simple":  # 如果策略是simple
            return await self.simple_fill(schema=schema, mode=mode, images=images, timeout=timeout, exclude=exclude)
        elif strgy == "complex":  # 如果策略是complex
            tmp = {}
            for _, i in self.children.items():  # 遍历所有子节点
                if exclude and i.key in exclude:  # 如果当前子节点在排除列表中，跳过
                    continue
                child = await i.simple_fill(schema=schema, mode=mode, images=images, timeout=timeout,
                                            exclude=exclude)  # 对子节点进行填充
                tmp.update(child.instruct_content.model_dump())  # 将子节点的填充结果更新到临时字典中
            cls = self._create_children_class()  # 创建子类
            self.instruct_content = cls(**tmp)  # 创建并设置子类的指令内容
            return self

    # 以下是审查和修改相关的方法

    async def human_review(self) -> dict[str, str]:  # 人工审查
        review_comments = HumanInteraction().interact_with_instruct_content(  # 与用户交互并获取审查评论
            instruct_content=self.instruct_content, interact_type="review"
        )
        return review_comments

    def _makeup_nodes_output_with_req(self) -> dict[str, str]:  # 生成带有要求的节点输出
        instruct_content_dict = self.instruct_content.model_dump()  # 获取当前指令内容的字典
        nodes_output = {}
        for key, value in instruct_content_dict.items():  # 遍历指令内容
            child = self.get_child(key)  # 获取子节点
            nodes_output[key] = {"value": value,
                                 "requirement": child.instruction if child else self.instruction}  # 如果有子节点，则使用其指令，否则使用当前指令
        return nodes_output

    async def auto_review(self, template: str = REVIEW_TEMPLATE) -> dict[str, str]:  # 自动审查
        nodes_output = self._makeup_nodes_output_with_req()  # 获取带要求的节点输出
        if not nodes_output:
            return dict()

        prompt = template.format(
            nodes_output=json.dumps(nodes_output, ensure_ascii=False),  # 格式化审查模板
            tag=TAG,  # 标签
            constraint=FORMAT_CONSTRAINT,  # 格式约束
            prompt_schema="json",  # 使用JSON格式的模式
        )

        content = await self.llm.aask(prompt)  # 请求LLM生成审查内容
        keys = self.keys()  # 获取所有节点的键
        include_keys = []
        for key in keys:
            if f'"{key}":' in content:  # 如果内容中包含该键，则包括它
                include_keys.append(key)
        if not include_keys:
            return dict()

        exclude_keys = list(set(keys).difference(include_keys))  # 获取排除的键
        output_class_name = f"{self.key}_AN_REVIEW"  # 设置输出类名
        output_class = self.create_class(class_name=output_class_name, exclude=exclude_keys)  # 创建输出类
        parsed_data = llm_output_postprocess(
            output=content, schema=output_class.model_json_schema(), req_key=f"[/{TAG}]"
        )  # 后处理LLM输出
        instruct_content = output_class(**parsed_data)  # 创建指令内容
        return instruct_content.model_dump()  # 返回审查结果

    async def simple_review(self, review_mode: ReviewMode = ReviewMode.AUTO):  # 简单审查
        if review_mode == ReviewMode.HUMAN:  # 如果是人工审查
            review_comments = await self.human_review()  # 获取人工审查评论
        else:
            review_comments = await self.auto_review()  # 获取自动审查评论

        if not review_comments:
            logger.warning("There are no review comments")  # 如果没有评论，则发出警告
        return review_comments

    async def review(self, strgy: str = "simple", review_mode: ReviewMode = ReviewMode.AUTO) -> dict[str, str]:  # 审查方法
        if not hasattr(self, "llm"):
            raise RuntimeError("use `review` after `fill`")  # 如果未调用fill方法，抛出错误
        assert review_mode in ReviewMode  # 确保审查模式有效
        assert self.instruct_content, 'review only support with `schema != "raw"`'  # 确保有指令内容

        if strgy == "simple":  # 如果策略是simple
            review_comments = await self.simple_review(review_mode)  # 获取简单审查评论
        elif strgy == "complex":  # 如果策略是complex
            review_comments = {}
            for _, child in self.children.items():  # 对每个子节点进行审查
                child_review_comment = await child.simple_review(review_mode)
                review_comments.update(child_review_comment)

        return review_comments

    # 以下是修订相关的方法

    async def human_revise(self) -> dict[str, str]:  # 人工修订
        review_contents = HumanInteraction().interact_with_instruct_content(  # 与用户交互并获取修订内容
            instruct_content=self.instruct_content, mapping=self.get_mapping(mode="auto"), interact_type="revise"
        )
        # 重新填充ActionNode
        self.update_instruct_content(review_contents)
        return review_contents

    def _makeup_nodes_output_with_comment(self, review_comments: dict[str, str]) -> dict[str, str]:  # 生成带评论的节点输出
        instruct_content_dict = self.instruct_content.model_dump()  # 获取当前指令内容
        nodes_output = {}
        for key, value in instruct_content_dict.items():
            if key in review_comments:  # 如果有修订评论，添加到节点输出
                nodes_output[key] = {"value": value, "comment": review_comments[key]}
        return nodes_output

    async def auto_revise(
            self, revise_mode: ReviseMode = ReviseMode.AUTO, template: str = REVISE_TEMPLATE
    ) -> dict[str, str]:  # 自动修订
        # 生成修订评论
        if revise_mode == ReviseMode.AUTO:
            review_comments: dict = await self.auto_review()  # 自动审查评论
        elif revise_mode == ReviseMode.HUMAN_REVIEW:
            review_comments: dict = await self.human_review()  # 人工审查评论

        include_keys = list(review_comments.keys())  # 获取需要修订的键

        # 生成修订内容
        nodes_output = self._makeup_nodes_output_with_comment(review_comments)  # 生成带评论的节点输出
        keys = self.keys()  # 获取所有节点的键
        exclude_keys = list(set(keys).difference(include_keys))  # 获取排除的键
        example = self.compile_example(schema="json", mode="auto", tag=TAG, exclude=exclude_keys)  # 编译示例
        instruction = self.compile_instruction(schema="markdown", mode="auto", exclude=exclude_keys)  # 编译指令

        prompt = template.format(  # 格式化修订模板
            nodes_output=json.dumps(nodes_output, ensure_ascii=False),
            example=example,
            instruction=instruction,
            constraint=FORMAT_CONSTRAINT,
            prompt_schema="json",
        )

        # 步骤2，使用_aask_v1获取修订结果
        output_mapping = self.get_mapping(mode="auto", exclude=exclude_keys)
        output_class_name = f"{self.key}_AN_REVISE"  # 设置修订类名
        content, scontent = await self._aask_v1(
            prompt=prompt, output_class_name=output_class_name, output_data_mapping=output_mapping, schema="json"
        )

        # 重新填充ActionNode
        sc_dict = scontent.model_dump()
        self.update_instruct_content(sc_dict)
        return sc_dict

    async def simple_revise(self, revise_mode: ReviseMode = ReviseMode.AUTO) -> dict[str, str]:  # 简单修订
        if revise_mode == ReviseMode.HUMAN:  # 如果是人工修订
            revise_contents = await self.human_revise()  # 获取人工修订内容
        else:
            revise_contents = await self.auto_revise()  # 获取自动修订内容
        return revise_contents

    async def revise(self, strgy: str = "simple", revise_mode: ReviseMode = ReviseMode.AUTO) -> dict[str, str]:  # 修订方法
        if not hasattr(self, "llm"):
            raise RuntimeError("use `revise` after `fill`")  # 如果未调用fill方法，抛出错误
        assert revise_mode in ReviseMode  # 确保修订模式有效
        assert self.instruct_content, 'revise only support with `schema != "raw"`'  # 确保有指令内容

        if strgy == "simple":  # 如果策略是simple
            revise_contents = await self.simple_revise(revise_mode)  # 获取简单修订内容
        elif strgy == "complex":  # 如果策略是complex
            revise_contents = {}
            for _, child in self.children.items():  # 对每个子节点进行修订
                child_revise_content = await child.simple_revise(revise_mode)
                revise_contents.update(child_revise_content)

        return revise_contents

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], key: str = None):
        """
        从 Pydantic 模型创建一个 ActionNode 树。

        参数：
            model (Type[BaseModel]): 要转换的 Pydantic 模型。

        返回：
            ActionNode: 创建的 ActionNode 树的根节点。
        """
        key = key or model.__name__  # 如果没有提供 key，则使用模型的类名作为 key
        root_node = cls(key=key, expected_type=Type[model], instruction="", example="")

        # 遍历模型的字段，处理每个字段
        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation  # 字段类型
            description = field_info.description  # 字段描述
            default = field_info.default  # 字段默认值

            # 如果字段类型是嵌套的 Pydantic 模型，则递归处理
            if not isinstance(field_type, typing._GenericAlias) and issubclass(field_type, BaseModel):
                child_node = cls.from_pydantic(field_type, key=field_name)  # 递归创建子节点
            else:
                # 否则，创建一个普通的字段节点
                child_node = cls(key=field_name, expected_type=field_type, instruction=description, example=default)

            root_node.add_child(child_node)  # 将子节点添加到根节点

        return root_node  # 返回根节点

    @staticmethod
    def is_optional_type(tp) -> bool:
        """如果 `tp` 是 `typing.Optional[...]` 类型，返回 True"""
        if typing.get_origin(tp) is Union:  # 判断类型是否为 Union
            args = typing.get_args(tp)  # 获取 Union 的类型参数
            non_none_types = [arg for arg in args if arg is not type(None)]  # 筛选出非 None 的类型
            return len(non_none_types) == 1 and len(args) == 2  # 如果只有一个非 None 类型且包含 None，则是 Optional
        return False
