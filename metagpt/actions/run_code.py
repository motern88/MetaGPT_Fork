#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : run_code.py
@Modified By: mashenquan, 2023/11/27.
            1.在 PROMPT_TEMPLATE 中使用 Markdown 代码块格式化标记控制台日志的位置，以增强 LLM 对日志的理解。
            修复 bug：添加“安装依赖”操作。
            2.将 RunCode 的输入封装为 RunCodeContext，将 RunCode 的输出封装为 RunCodeResult，以规范化和统一 WriteCode、RunCode 和 DebugError 之间的参数传递。
            3.根据 RFC 135 第 2.2.3.5.7 节，修改传递文件内容（代码文件、单元测试文件、日志文件）的方法，从使用消息改为使用文件名。
            4.合并 send18:dev 分支的 Config 类，接管 Environment 类的 set/get 操作。
"""
import subprocess
from pathlib import Path
from typing import Tuple

from pydantic import Field

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.schema import RunCodeContext, RunCodeResult
from metagpt.utils.exceptions import handle_exception

PROMPT_TEMPLATE = """
角色：你是一个资深的开发和 QA 工程师，你的角色是总结代码运行结果。
如果运行结果没有包含错误，你应该明确批准该结果。
另一方面，如果运行结果显示存在错误，你应该指出是开发代码还是测试代码产生了错误，
并给出具体的修正指示。以下是代码信息：
{context}
现在你应该开始分析
---
## 指令：
请总结错误的原因并给出修正指示
## 需要重写的文件：
确定需要重写的文件以修复错误，例如 xyz.py 或 test_xyz.py
## 状态：
确定所有代码是否正常工作，如果是，请写 PASS，否则写 FAIL，
此部分只写一个单词，PASS 或 FAIL
## 发送给：
如果没有错误，请写 NoOne；如果错误是由于开发代码问题引起的，请写 Engineer；如果是 QA 代码问题，请写 QaEngineer，
此部分只写一个单词，NoOne 或 Engineer 或 QaEngineer。
---
你应该填写必要的指令、状态、发送给谁，并最终返回 --- 分隔线之间的所有内容。
"""

TEMPLATE_CONTEXT = """
## 开发代码文件名
{code_file_name}
## 开发代码
```python
{code}
```
## 测试文件名
{test_file_name}
## 测试代码
```python
{test_code}
```
## 运行命令
{command}
## 运行输出
标准输出: 
```text
{outs}
```
标准错误: 
```text
{errs}
```
"""


class RunCode(Action):
    name: str = "RunCode"  # 定义类名，用于标识该动作的名称
    i_context: RunCodeContext = Field(default_factory=RunCodeContext)  # 定义类的上下文，默认使用 RunCodeContext

    @classmethod
    async def run_text(cls, code) -> Tuple[str, str]:
        """运行代码并返回结果和错误信息"""
        try:
            # 我们将在这个字典中存储结果
            namespace = {}  # 用来存储执行代码的环境变量
            exec(code, namespace)  # 执行代码，将结果存储在 namespace 中
        except Exception as e:
            return "", str(e)  # 如果发生错误，返回空字符串和错误信息
        return namespace.get("result", ""), ""  # 返回 'result' 变量的值，如果没有，则返回空字符串

    async def run_script(self, working_directory, additional_python_paths=[], command=[]) -> Tuple[str, str]:
        """运行脚本并返回标准输出和标准错误"""
        working_directory = str(working_directory)  # 转换工作目录为字符串
        additional_python_paths = [str(path) for path in additional_python_paths]  # 确保所有路径是字符串

        # 复制当前的环境变量
        env = self.context.new_environ()  # 获取新的环境变量

        # 修改 PYTHONPATH 环境变量
        additional_python_paths = [working_directory] + additional_python_paths  # 将工作目录加入到额外的 Python 路径中
        additional_python_paths = ":".join(additional_python_paths)  # 将路径连接成一个字符串
        env["PYTHONPATH"] = additional_python_paths + ":" + env.get("PYTHONPATH", "")  # 更新环境变量中的 PYTHONPATH
        RunCode._install_dependencies(working_directory=working_directory, env=env)  # 安装依赖

        # 启动子进程运行脚本
        process = subprocess.Popen(
            command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        logger.info(" ".join(command))  # 记录运行的命令

        try:
            # 等待子进程完成，并设置超时
            stdout, stderr = process.communicate(timeout=10)  # 获取标准输出和标准错误
        except subprocess.TimeoutExpired:
            logger.info("The command did not complete within the given timeout.")  # 如果超时，记录超时信息
            process.kill()  # 如果超时，终止进程
            stdout, stderr = process.communicate()  # 获取子进程的输出
        return stdout.decode("utf-8"), stderr.decode("utf-8")  # 返回解码后的输出和错误信息

    async def run(self, *args, **kwargs) -> RunCodeResult:
        """执行运行代码的任务"""
        logger.info(f"Running {' '.join(self.i_context.command)}")  # 记录运行的命令
        if self.i_context.mode == "script":  # 如果模式是脚本
            outs, errs = await self.run_script(
                command=self.i_context.command,
                working_directory=self.i_context.working_directory,
                additional_python_paths=self.i_context.additional_python_paths,
            )
        elif self.i_context.mode == "text":  # 如果模式是文本
            outs, errs = await self.run_text(code=self.i_context.code)

        logger.info(f"{outs=}")  # 记录标准输出
        logger.info(f"{errs=}")  # 记录标准错误

        # 格式化模板上下文
        context = TEMPLATE_CONTEXT.format(
            code=self.i_context.code,
            code_file_name=self.i_context.code_filename,
            test_code=self.i_context.test_code,
            test_file_name=self.i_context.test_filename,
            command=" ".join(self.i_context.command),
            outs=outs[:500],  # 将输出限制为前500个字符，防止token溢出
            errs=errs[:10000],  # 将错误信息限制为前10000个字符，防止token溢出
        )

        # 格式化最终的提示
        prompt = PROMPT_TEMPLATE.format(context=context)
        rsp = await self._aask(prompt)  # 获取响应
        return RunCodeResult(summary=rsp, stdout=outs, stderr=errs)  # 返回运行结果

    @staticmethod
    @handle_exception(exception_type=subprocess.CalledProcessError)  # 处理 CalledProcessError 异常
    def _install_via_subprocess(cmd, check, cwd, env):
        """通过子进程安装依赖"""
        return subprocess.run(cmd, check=check, cwd=cwd, env=env)  # 运行安装命令并检查错误

    @staticmethod
    def _install_requirements(working_directory, env):
        """安装 requirements.txt 中列出的依赖"""
        file_path = Path(working_directory) / "requirements.txt"  # 获取 requirements.txt 文件路径
        if not file_path.exists():  # 如果文件不存在，返回
            return
        if file_path.stat().st_size == 0:  # 如果文件为空，返回
            return
        install_command = ["python", "-m", "pip", "install", "-r", "requirements.txt"]  # 安装命令
        logger.info(" ".join(install_command))  # 记录安装命令
        RunCode._install_via_subprocess(install_command, check=True, cwd=working_directory, env=env)  # 通过子进程安装依赖

    @staticmethod
    def _install_pytest(working_directory, env):
        """安装 pytest 测试框架"""
        install_pytest_command = ["python", "-m", "pip", "install", "pytest"]  # 安装 pytest 的命令
        logger.info(" ".join(install_pytest_command))  # 记录安装命令
        RunCode._install_via_subprocess(install_pytest_command, check=True, cwd=working_directory, env=env)  # 通过子进程安装 pytest

    @staticmethod
    def _install_dependencies(working_directory, env):
        """安装所有依赖"""
        RunCode._install_requirements(working_directory, env)  # 安装 requirements.txt 中的依赖
        RunCode._install_pytest(working_directory, env)  # 安装 pytest
