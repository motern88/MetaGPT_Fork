#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : debug_error.py
@Modified By: mashenquan, 2023/11/27.
        1. Divide the context into three components: legacy code, unit test code, and console log.
        2. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
"""
import re
from typing import Optional

from pydantic import BaseModel, Field

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.schema import RunCodeContext, RunCodeResult
from metagpt.utils.common import CodeParser
from metagpt.utils.project_repo import ProjectRepo

PROMPT_TEMPLATE = """
NOTICE
1. 角色: 你是一个开发工程师或QA工程师；
2. 任务: 你收到来自另一位开发工程师或QA工程师的消息，消息中包含他们运行或测试你的代码后的结果。根据这个消息，首先判断你自己扮演的角色，是开发工程师（Engineer）还是QA工程师（QaEngineer），
然后根据你的角色、错误信息和总结，重写开发代码或测试代码，修复所有错误并确保代码正常工作。
注意：使用 '##' 来分隔各部分内容，不要使用 '#'，并且 '## <SECTION_NAME>' 应该写在测试用例或脚本之前，并且用三引号包裹代码。
以下是消息内容：
# 旧代码
```python
{code}
```
---
# Unit Test Code
```python
{test_code}
```
---
# Console logs
```text
{logs}
```
---
Now you should start rewriting the code:
## file name of the code to rewrite: Write code with triple quote. Do your best to implement THIS IN ONLY ONE FILE.
"""


class DebugError(Action):
    # RunCodeContext：存储与运行代码相关的上下文信息（例如文件名、执行状态等）
    i_context: RunCodeContext = Field(default_factory=RunCodeContext)

    # ProjectRepo：项目仓库，包含源代码、测试代码等内容
    repo: Optional[ProjectRepo] = Field(default=None, exclude=True)

    # BaseModel：用于存储输入参数的基类
    input_args: Optional[BaseModel] = Field(default=None, exclude=True)

    # 异步运行方法，执行错误调试和代码重写
    async def run(self, *args, **kwargs) -> str:
        # 从仓库获取测试输出文档，检查是否存在输出
        output_doc = await self.repo.test_outputs.get(filename=self.i_context.output_filename)
        if not output_doc:
            return ""  # 如果没有输出文档，返回空字符串

        # 解析输出文档的内容
        output_detail = RunCodeResult.loads(output_doc.content)

        # 使用正则表达式检查测试是否通过（即所有测试通过时显示 "OK"）
        pattern = r"Ran (\d+) tests in ([\d.]+)s\n\nOK"
        matches = re.search(pattern, output_detail.stderr)

        # 如果匹配成功，说明所有测试都通过了，直接返回空字符串
        if matches:
            return ""

        # 如果测试未通过，记录日志并开始调试
        logger.info(f"Debug and rewrite {self.i_context.test_filename}")

        # 获取源代码文档
        code_doc = await self.repo.srcs.get(filename=self.i_context.code_filename)
        if not code_doc:
            return ""  # 如果没有源代码文档，返回空字符串

        # 获取测试代码文档
        test_doc = await self.repo.tests.get(filename=self.i_context.test_filename)
        if not test_doc:
            return ""  # 如果没有测试代码文档，返回空字符串

        # 构建提示信息，包含源代码、测试代码和控制台日志
        prompt = PROMPT_TEMPLATE.format(code=code_doc.content, test_code=test_doc.content, logs=output_detail.stderr)

        # 向模型请求重写代码的建议
        rsp = await self._aask(prompt)

        # 解析模型返回的重写代码
        code = CodeParser.parse_code(text=rsp)

        # 返回重写后的代码
        return code
