#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : qa_engineer.py
@Modified By: mashenquan, 2023-11-1. 根据 RFC 116 的第 2.2.1 和 2.2.2 节，
        将 `Message` 中 `cause_by` 值的数据类型修改为字符串，并利用新的消息过滤功能。

@Modified By: mashenquan, 2023-11-27.
        1. 根据思考-行动原则，在创建 WriteTest/RunCode/DebugError 对象时，固化任务参数，而不是在调用 run 函数时传递它们。
        2. 根据 RFC 135 第 2.2.3.5.7 节，修改文件传输的方法，从使用 Message 传输改为使用文件引用。

@Modified By: mashenquan, 2023-12-5. 增强工作流，根据 SummarizeCode 的结果导航到 WriteCode 或 QaEngineer。
"""
from typing import Optional

from pydantic import BaseModel, Field

from metagpt.actions import DebugError, RunCode, UserRequirement, WriteTest
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.actions.summarize_code import SummarizeCode
from metagpt.const import MESSAGE_ROUTE_TO_NONE, MESSAGE_ROUTE_TO_SELF
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import AIMessage, Document, Message, RunCodeContext, TestingContext
from metagpt.utils.common import (
    any_to_str,
    any_to_str_set,
    get_project_srcs_path,
    init_python_folder,
    parse_recipient,
)
from metagpt.utils.project_repo import ProjectRepo
from metagpt.utils.report import EditorReporter


class QaEngineer(Role):
    name: str = "Edward"  # 姓名
    profile: str = "QaEngineer"  # 角色描述
    goal: str = "Write comprehensive and robust tests to ensure codes will work as expected without bugs"  # 目标：编写全面且健壮的测试，以确保代码按预期工作且无漏洞
    constraints: str = (
        "The test code you write should conform to code standard like PEP8, be modular, easy to read and maintain."
        "Use same language as user requirement"
    )  # 限制：编写的测试代码应符合 PEP8 等代码标准，模块化，易于阅读和维护；使用与用户需求相同的语言
    test_round_allowed: int = 5  # 允许的最大测试轮数
    test_round: int = 0  # 当前测试轮数
    repo: Optional[ProjectRepo] = Field(default=None, exclude=True)  # 可选的项目仓库
    input_args: Optional[BaseModel] = Field(default=None, exclude=True)  # 可选的输入参数

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_memory = False  # 禁用内存

        # FIXME: 这里有点小技巧，初始化一个动作来绕过 _think() 逻辑，
        # 将在未来更新中覆盖 _think()
        self.set_actions([WriteTest])  # 设置动作
        self._watch([SummarizeCode, WriteTest, RunCode, DebugError])  # 监听相关消息
        self.test_round = 0  # 初始化测试轮数

    # 编写测试代码
    async def _write_test(self, message: Message) -> None:
        reqa_file = self.context.kwargs.reqa_file or self.config.reqa_file
        changed_files = {reqa_file} if reqa_file else set(self.repo.srcs.changed_files.keys())
        for filename in changed_files:
            # 编写测试
            if not filename or "test" in filename:
                continue
            code_doc = await self.repo.srcs.get(filename)
            if not code_doc or not code_doc.content:
                continue
            if not code_doc.filename.endswith(".py"):
                continue
            test_doc = await self.repo.tests.get("test_" + code_doc.filename)
            if not test_doc:
                test_doc = Document(
                    root_path=str(self.repo.tests.root_path), filename="test_" + code_doc.filename, content=""
                )
            logger.info(f"Writing {test_doc.filename}..")
            context = TestingContext(filename=test_doc.filename, test_doc=test_doc, code_doc=code_doc)

            context = await WriteTest(i_context=context, context=self.context, llm=self.llm).run()
            async with EditorReporter(enable_llm_stream=True) as reporter:
                await reporter.async_report({"type": "test", "filename": test_doc.filename}, "meta")

                doc = await self.repo.tests.save_doc(
                    doc=context.test_doc, dependencies={context.code_doc.root_relative_path}
                )
                await reporter.async_report(self.repo.workdir / doc.root_relative_path, "path")

            # 为下一轮运行测试准备上下文
            run_code_context = RunCodeContext(
                command=["python", context.test_doc.root_relative_path],
                code_filename=context.code_doc.filename,
                test_filename=context.test_doc.filename,
                working_directory=str(self.repo.workdir),
                additional_python_paths=[str(self.repo.srcs.workdir)],
            )
            self.publish_message(
                AIMessage(content=run_code_context.model_dump_json(), cause_by=WriteTest, send_to=MESSAGE_ROUTE_TO_SELF)
            )

        logger.info(f"Done {str(self.repo.tests.workdir)} generating.")  # 完成测试代码生成

    # 运行代码
    async def _run_code(self, msg):
        run_code_context = RunCodeContext.loads(msg.content)
        src_doc = await self.repo.srcs.get(run_code_context.code_filename)
        if not src_doc:
            return
        test_doc = await self.repo.tests.get(run_code_context.test_filename)
        if not test_doc:
            return
        run_code_context.code = src_doc.content
        run_code_context.test_code = test_doc.content
        result = await RunCode(i_context=run_code_context, context=self.context, llm=self.llm).run()
        run_code_context.output_filename = run_code_context.test_filename + ".json"
        await self.repo.test_outputs.save(
            filename=run_code_context.output_filename,
            content=result.model_dump_json(),
            dependencies={src_doc.root_relative_path, test_doc.root_relative_path},
        )
        run_code_context.code = None
        run_code_context.test_code = None
        # 收件人可能是工程师或我自己
        recipient = parse_recipient(result.summary)
        mappings = {"Engineer": "Alex", "QaEngineer": "Edward"}
        if recipient != "Engineer":
            self.publish_message(
                AIMessage(
                    content=run_code_context.model_dump_json(),
                    cause_by=RunCode,
                    instruct_content=self.input_args,
                    send_to=MESSAGE_ROUTE_TO_SELF,
                )
            )
        else:
            kvs = self.input_args.model_dump()
            kvs["changed_test_filenames"] = [
                str(self.repo.tests.workdir / i) for i in list(self.repo.tests.changed_files.keys())
            ]
            self.publish_message(
                AIMessage(
                    content=run_code_context.model_dump_json(),
                    cause_by=RunCode,
                    instruct_content=self.input_args,
                    send_to=mappings.get(recipient, MESSAGE_ROUTE_TO_NONE),
                )
            )

    # 调试错误
    async def _debug_error(self, msg):
        run_code_context = RunCodeContext.loads(msg.content)
        code = await DebugError(
            i_context=run_code_context, repo=self.repo, input_args=self.input_args, context=self.context, llm=self.llm
        ).run()
        await self.repo.tests.save(filename=run_code_context.test_filename, content=code)
        run_code_context.output = None
        self.publish_message(
            AIMessage(content=run_code_context.model_dump_json(), cause_by=DebugError, send_to=MESSAGE_ROUTE_TO_SELF)
        )

    # 行动函数，根据条件执行不同操作
    async def _act(self) -> Message:
        if self.input_args.project_path:
            await init_python_folder(self.repo.tests.workdir)
        if self.test_round > self.test_round_allowed:
            kvs = self.input_args.model_dump()
            kvs["changed_test_filenames"] = [
                str(self.repo.tests.workdir / i) for i in list(self.repo.tests.changed_files.keys())
            ]
            result_msg = AIMessage(
                content=f"Exceeding {self.test_round_allowed} rounds of tests, stop. "
                + "\n".join(list(self.repo.tests.changed_files.keys())),
                cause_by=WriteTest,
                instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="WriteTestOutput"),
                send_to=MESSAGE_ROUTE_TO_NONE,
            )
            return result_msg

        code_filters = any_to_str_set({PrepareDocuments, SummarizeCode})
        test_filters = any_to_str_set({WriteTest, DebugError})
        run_filters = any_to_str_set({RunCode})
        for msg in self.rc.news:
            # 根据观察到的消息类型决定执行什么操作，目前由人工定义，
            # 未来可能将其移动到 _think 函数，由代理自动决定
            if msg.cause_by in code_filters:
                # 工程师写了代码，是时候为其编写测试了
                await self._write_test(msg)
            elif msg.cause_by in test_filters:
                # 我编写或调试了测试代码，是时候运行它了
                await self._run_code(msg)
            elif msg.cause_by in run_filters:
                # 我运行了测试代码，是时候修复错误了
                await self._debug_error(msg)
            elif msg.cause_by == any_to_str(UserRequirement):
                return await self._parse_user_requirement(msg)
        self.test_round += 1
        kvs = self.input_args.model_dump()
        kvs["changed_test_filenames"] = [
            str(self.repo.tests.workdir / i) for i in list(self.repo.tests.changed_files.keys())
        ]
        return AIMessage(
            content=f"Round {self.test_round} of tests done",
            instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="WriteTestOutput"),
            cause_by=WriteTest,
            send_to=MESSAGE_ROUTE_TO_NONE,
        )

    # 解析用户需求并准备文档
    async def _parse_user_requirement(self, msg: Message) -> AIMessage:
        action = PrepareDocuments(
            send_to=any_to_str(self),
            key_descriptions={
                "project_path": '如果 "Original Requirement" 中存在，提供项目路径',
                "reqa_file": '如果 "Original Requirement" 中存在，提供需要重写单元测试的文件名',
            },
            context=self.context,
        )
        rsp = await action.run([msg])
        if not self.src_workspace:
            self.src_workspace = self.git_repo.workdir / self.git_repo.workdir.name
        return rsp

    # 思考函数，决定下一步的行动
    async def _think(self) -> bool:
        if not self.rc.news:
            return False
        msg = self.rc.news[0]
        if msg.cause_by == any_to_str(SummarizeCode):
            self.input_args = msg.instruct_content
            self.repo = ProjectRepo(self.input_args.project_path)
            if self.repo.src_relative_path is None:
                path = get_project_srcs_path(self.repo.workdir)
                self.repo.with_src_path(path)
        return True
