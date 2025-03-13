#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : write_test.py
@Modified By: mashenquan, 2023-11-27. Following the think-act principle, solidify the task parameters when creating the
        WriteTest object, rather than passing them in when calling the run function.
"""

from typing import Optional

from metagpt.actions.action import Action
from metagpt.const import TEST_CODES_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import Document, TestingContext
from metagpt.utils.common import CodeParser

PROMPT_TEMPLATE = """
NOTICE
1. Role: You are a QA engineer; the main goal is to design, develop, and execute PEP8 compliant, well-structured, maintainable test cases and scripts for Python 3.9. Your focus should be on ensuring the product quality of the entire project through systematic testing.
2. Requirement: Based on the context, develop a comprehensive test suite that adequately covers all relevant aspects of the code file under review. Your test suite will be part of the overall project QA, so please develop complete, robust, and reusable test cases.
3. Attention1: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script.
4. Attention2: If there are any settings in your tests, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
5. Attention3: YOU MUST FOLLOW "Data structures and interfaces". DO NOT CHANGE ANY DESIGN. Make sure your tests respect the existing design and ensure its validity.
6. Think before writing: What should be tested and validated in this document? What edge cases could exist? What might fail?
7. CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES/SCRIPTS IN THIS FILE.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
-----
## Given the following code, please write appropriate test cases using Python's unittest framework to verify the correctness and robustness of this code:
```python
{code_to_test}
```
Note that the code to test is at {source_file_path}, we will put your test code at {workspace}/tests/{test_file_name}, and run your test code from {workspace},
you should correctly import the necessary classes based on these file locations!
## {test_file_name}: Write test code with triple quote. Do your best to implement THIS ONLY ONE FILE.
"""


class WriteTest(Action):
    # 类名：WriteTest，表示一个编写测试的动作
    name: str = "WriteTest"  # 动作名称
    i_context: Optional[TestingContext] = None  # 测试上下文，可选

    async def write_code(self, prompt):
        # 异步方法，用于生成代码
        code_rsp = await self._aask(prompt)  # 向语言模型请求生成代码

        try:
            # 尝试解析代码
            code = CodeParser.parse_code(text=code_rsp)
        except Exception:
            # 如果代码解析失败，记录错误并返回原始的代码响应
            logger.error(f"Can't parse the code: {code_rsp}")
            code = code_rsp  # 在解析失败的情况下，直接返回原始代码响应
        return code

    async def run(self, *args, **kwargs) -> TestingContext:
        # 异步方法，用于运行测试生成逻辑
        if not self.i_context.test_doc:
            # 如果没有测试文档，则创建一个新的测试文档
            self.i_context.test_doc = Document(
                filename="test_" + self.i_context.code_doc.filename,  # 使用源代码文件名创建测试文件名
                root_path=TEST_CODES_FILE_REPO  # 测试代码文件的根路径
            )
        fake_root = "/data"  # 设置一个假根目录路径
        prompt = PROMPT_TEMPLATE.format(
            code_to_test=self.i_context.code_doc.content,  # 要测试的代码内容
            test_file_name=self.i_context.test_doc.filename,  # 测试文件的名称
            source_file_path=fake_root + "/" + self.i_context.code_doc.root_relative_path,  # 源代码文件的路径
            workspace=fake_root,  # 工作空间
        )
        # 使用模板生成提示并生成代码
        self.i_context.test_doc.content = await self.write_code(prompt)
        # 返回更新后的测试上下文
        return self.i_context
