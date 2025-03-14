#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from metagpt.config2 import config
from metagpt.provider.openai_api import OpenAILLM as GPTAPI
from metagpt.utils.common import awrite

ICL_SAMPLE = """Interface definition:
```text
Interface Name: Element Tagging
Interface Path: /projects/{project_key}/node-tags
Method: POST

Request parameters:
Path parameters:
project_key

Body parameters:
Name	Type	Required	Default Value	Remarks
nodes	array	Yes		Nodes
	node_key	string	No		Node key
	tags	array	No		Original node tag list
	node_type	string	No		Node type DATASET / RECIPE
operations	array	Yes		
	tags	array	No		Operation tag list
	mode	string	No		Operation type ADD / DELETE

Return data:
Name	Type	Required	Default Value	Remarks
code	integer	Yes		Status code
msg	string	Yes		Prompt message
data	object	Yes		Returned data
list	array	No		Node list true / false
node_type	string	No		Node type DATASET / RECIPE
node_key	string	No		Node key
```

Unit test：
```python
@pytest.mark.parametrize(
"project_key, nodes, operations, expected_msg",
[
("project_key", [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "success"),
("project_key", [{"node_key": "dataset_002", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["tag1"], "mode": "DELETE"}], "success"),
("", [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Missing the required parameter project_key"),
(123, [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Incorrect parameter type"),
("project_key", [{"node_key": "a"*201, "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Request parameter exceeds field boundary")
]
)
def test_node_tags(project_key, nodes, operations, expected_msg):
    pass

# The above is an interface definition and a unit test example.
# Next, please play the role of an expert test manager with 20 years of experience at Google. When I give the interface definition, 
# reply to me with a unit test. There are several requirements:
# 1. Only output one `@pytest.mark.parametrize` and the corresponding test_<interface name> function (inside pass, do not implement).
# -- The function parameter contains expected_msg for result verification.
# 2. The generated test cases use shorter text or numbers and are as compact as possible.
# 3. If comments are needed, use Chinese.

# If you understand, please wait for me to give the interface definition and just answer "Understood" to save tokens.
"""

ACT_PROMPT_PREFIX = """Refer to the test types: such as missing request parameters, field boundary verification, incorrect field type.
Please output 10 test cases within one `@pytest.mark.parametrize` scope.
```text
"""

YFT_PROMPT_PREFIX = """Refer to the test types: such as SQL injection, cross-site scripting (XSS), unauthorized access and privilege escalation, 
authentication and authorization, parameter verification, exception handling, file upload and download.
Please output 10 test cases within one `@pytest.mark.parametrize` scope.
```text
"""

OCR_API_DOC = """```text
Interface Name: OCR recognition
Interface Path: /api/v1/contract/treaty/task/ocr
Method: POST

Request Parameters:
Path Parameters:

Body Parameters:
Name	Type	Required	Default Value	Remarks
file_id	string	Yes		
box	array	Yes		
contract_id	number	Yes		Contract id
start_time	string	No		yyyy-mm-dd
end_time	string	No		yyyy-mm-dd
extract_type	number	No		Recognition type 1- During import 2- After import Default 1

Response Data:
Name	Type	Required	Default Value	Remarks
code	integer	Yes		
message	string	Yes		
data	object	Yes		
```
"""


class UTGenerator:
    """UT生成器：通过API文档构建单元测试"""

    def __init__(
        self,
        swagger_file: str,
        ut_py_path: str,
        questions_path: str,
        chatgpt_method: str = "API",
        template_prefix=YFT_PROMPT_PREFIX,
    ) -> None:
        """初始化UT生成器

        参数：
            swagger_file: Swagger文件路径
            ut_py_path: 存储测试用例的路径
            questions_path: 存储模板的路径，方便后续检查
            chatgpt_method: 使用的ChatGPT方法，默认为API
            template_prefix: 使用的模板，默认为YFT_UT_PROMPT
        """
        self.swagger_file = swagger_file
        self.ut_py_path = ut_py_path
        self.questions_path = questions_path
        assert chatgpt_method in ["API"], "无效的chatgpt_method"
        self.chatgpt_method = chatgpt_method

        # ICL：上下文学习，提供一个示例供GPT模仿
        self.icl_sample = ICL_SAMPLE
        self.template_prefix = template_prefix

    def get_swagger_json(self) -> dict:
        """从本地文件加载Swagger JSON"""
        with open(self.swagger_file, "r", encoding="utf-8") as file:
            swagger_json = json.load(file)
        return swagger_json

    def __para_to_str(self, prop, required, name=""):
        """将参数转换为字符串格式

        参数：
            prop: 参数的属性字典
            required: 是否是必需的
            name: 参数的名称
        """
        name = name or prop["name"]
        ptype = prop["type"]
        title = prop.get("title", "")
        desc = prop.get("description", "")
        return f'{name}\t{ptype}\t{"Yes" if required else "No"}\t{title}\t{desc}'

    def _para_to_str(self, prop):
        """将参数转换为字符串格式，默认参数为非必需的"""
        required = prop.get("required", False)
        return self.__para_to_str(prop, required)

    def para_to_str(self, name, prop, prop_object_required):
        """处理特定参数的字符串转换

        参数：
            name: 参数名称
            prop: 参数属性
            prop_object_required: 是否是必需的
        """
        required = name in prop_object_required
        return self.__para_to_str(prop, required, name)

    def build_object_properties(self, node, prop_object_required, level: int = 0) -> str:
        """递归输出对象和数组[对象]类型的属性

        参数：
            node: 子项的值
            prop_object_required: 是否为必需字段
            level: 当前递归深度
        """
        doc = ""

        def dive_into_object(node):
            """如果是对象类型，递归输出其属性"""
            if node.get("type") == "object":
                sub_properties = node.get("properties", {})
                return self.build_object_properties(sub_properties, prop_object_required, level=level + 1)
            return ""

        if node.get("in", "") in ["query", "header", "formData"]:
            doc += f'{"	" * level}{self._para_to_str(node)}\n'
            doc += dive_into_object(node)
            return doc

        for name, prop in node.items():
            if not isinstance(prop, dict):
                doc += f'{"	" * level}{self._para_to_str(node)}\n'
                break
            doc += f'{"	" * level}{self.para_to_str(name, prop, prop_object_required)}\n'
            doc += dive_into_object(prop)
            if prop["type"] == "array":
                items = prop.get("items", {})
                doc += dive_into_object(items)
        return doc

    def get_tags_mapping(self) -> dict:
        """处理标签和路径的映射关系

        返回：
            dict: 标签到路径的映射
        """
        swagger_data = self.get_swagger_json()
        paths = swagger_data["paths"]
        tags = {}

        for path, path_obj in paths.items():
            for method, method_obj in path_obj.items():
                for tag in method_obj["tags"]:
                    if tag not in tags:
                        tags[tag] = {}
                    if path not in tags[tag]:
                        tags[tag][path] = {}
                    tags[tag][path][method] = method_obj

        return tags

    async def generate_ut(self, include_tags) -> bool:
        """生成测试用例文件"""
        tags = self.get_tags_mapping()  # 获取标签与路径的映射
        # 遍历所有标签及其对应的路径
        for tag, paths in tags.items():
            # 如果未指定标签或当前标签在include_tags中，则生成测试用例
            if include_tags is None or tag in include_tags:
                await self._generate_ut(tag, paths)  # 生成指定标签下的测试用例
        return True

    def build_api_doc(self, node: dict, path: str, method: str) -> str:
        """构建API文档"""
        summary = node["summary"]  # 获取API的摘要

        # 创建API文档的基本信息
        doc = f"API 名称: {summary}\nAPI 路径: {path}\n方法: {method.upper()}\n"
        doc += "\n请求参数:\n"
        if "parameters" in node:  # 如果有请求参数
            parameters = node["parameters"]
            doc += "路径参数:\n"

            # param["in"]: path / formData / body / query / header
            for param in parameters:
                if param["in"] == "path":  # 如果是路径参数
                    doc += f'{param["name"]} \n'

            doc += "\n请求体参数:\n"
            doc += "名称\t类型\t是否必需\t默认值\t备注\n"
            for param in parameters:
                if param["in"] == "body":  # 如果是请求体参数
                    schema = param.get("schema", {})
                    prop_properties = schema.get("properties", {})
                    prop_required = schema.get("required", [])
                    doc += self.build_object_properties(prop_properties, prop_required)  # 递归处理对象属性
                else:
                    doc += self.build_object_properties(param, [])  # 处理其他类型的参数

        # 展示响应数据的信息
        doc += "\n响应数据:\n"
        doc += "名称\t类型\t是否必需\t默认值\t备注\n"
        responses = node["responses"]
        response = responses.get("200", {})  # 获取200响应的schema
        schema = response.get("schema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", {})

        doc += self.build_object_properties(properties, required)  # 处理响应数据的属性
        doc += "\n"
        doc += "```"

        return doc

    async def ask_gpt_and_save(self, question: str, tag: str, fname: str):
        """生成问题并保存问题和答案"""
        messages = [self.icl_sample, question]  # 将示例和问题拼接成消息
        result = await self.gpt_msgs_to_code(messages=messages)  # 获取GPT生成的代码

        # 将问题保存到指定路径
        await awrite(Path(self.questions_path) / tag / f"{fname}.txt", question)
        data = result.get("code", "") if result else ""  # 获取生成的代码
        # 将生成的代码保存到指定路径
        await awrite(Path(self.ut_py_path) / tag / f"{fname}.py", data)

    async def _generate_ut(self, tag, paths):
        """处理路径下的结构

        参数：
            tag (_type_): 模块名称
            paths (_type_): 路径对象
        """
        # 遍历每个路径和对应的操作方法
        for path, path_obj in paths.items():
            for method, node in path_obj.items():
                summary = node["summary"]  # 获取API的摘要
                question = self.template_prefix  # 获取模板前缀
                question += self.build_api_doc(node, path, method)  # 构建API文档
                await self.ask_gpt_and_save(question, tag, summary)  # 生成并保存测试用例

    async def gpt_msgs_to_code(self, messages: list) -> str:
        """根据不同调用方式选择处理方法"""
        result = ""
        if self.chatgpt_method == "API":
            result = await GPTAPI(config.get_openai_llm()).aask_code(messages=messages)  # 调用API获取代码

        return result
