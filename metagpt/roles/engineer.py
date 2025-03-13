#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : engineer.py
@Modified By: mashenquan, 2023-11-1. 根据 RFC 116 的第 2.2.1 和 2.2.2 节：
    1. 修改 `Message` 中 `cause_by` 值的数据类型为字符串，并利用新的消息分发功能进行消息过滤。
    2. 将消息接收和处理逻辑合并到 `_observe` 中。
    3. 修复 bug：增加处理异步消息的逻辑，当消息尚未准备好时进行处理。
    4. 补充了内部消息的外部传输。
@Modified By: mashenquan, 2023-11-27.
    1. 根据 RFC 135 第 2.2.3.1 节的要求，将消息中的文件数据替换为文件名。
    2. 根据 RFC 135 第 2.2.3.5.5 节的设计，添加增量迭代功能。
@Modified By: mashenquan, 2023-12-5. 增强工作流，根据 SummarizeCode 的结果导航至 WriteCode 或 QaEngineer。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set

from pydantic import BaseModel, Field

from metagpt.actions import WriteCode, WriteCodeReview, WriteTasks
from metagpt.actions.fix_bug import FixBug
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.actions.project_management_an import REFINED_TASK_LIST, TASK_LIST
from metagpt.actions.summarize_code import SummarizeCode
from metagpt.actions.write_code_plan_and_change_an import WriteCodePlanAndChange
from metagpt.const import (
    CODE_PLAN_AND_CHANGE_FILE_REPO,
    MESSAGE_ROUTE_TO_SELF,
    REQUIREMENT_FILENAME,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import (
    AIMessage,
    CodePlanAndChangeContext,
    CodeSummarizeContext,
    CodingContext,
    Document,
    Documents,
    Message,
)
from metagpt.utils.common import (
    any_to_name,
    any_to_str,
    any_to_str_set,
    get_project_srcs_path,
    init_python_folder,
)
from metagpt.utils.git_repository import ChangeType
from metagpt.utils.project_repo import ProjectRepo

IS_PASS_PROMPT = """
{context}

----
Does the above log indicate anything that needs to be done?
If there are any tasks to be completed, please answer 'NO' along with the to-do list in JSON format;
otherwise, answer 'YES' in JSON format.
"""


class Engineer(Role):
    """
    表示一个工程师角色，负责编写和可能的代码审查。

    属性：
        name (str): 工程师的名字。
        profile (str): 角色简介，默认值为 'Engineer'。
        goal (str): 工程师的目标。
        constraints (str): 工程师的约束条件。
        n_borg (int): borg 的数量。
        use_code_review (bool): 是否使用代码审查。
    """

    name: str = "Alex"
    profile: str = "Engineer"
    goal: str = "编写优雅、可读、可扩展、高效的代码"
    constraints: str = (
        "代码应符合如 Google 风格的标准，并且要模块化和可维护。 "
        "使用与用户要求相同的语言。"
    )
    n_borg: int = 1
    use_code_review: bool = False
    code_todos: list = []  # 代码待办事项
    summarize_todos: list = []  # 总结待办事项
    next_todo_action: str = ""  # 下一个待办动作
    n_summarize: int = 0  # 总结次数
    input_args: Optional[BaseModel] = Field(default=None, exclude=True)  # 输入参数
    repo: Optional[ProjectRepo] = Field(default=None, exclude=True)  # 项目仓库

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.enable_memory = False  # 是否启用内存
        self.set_actions([WriteCode])  # 设置动作为写代码
        self._watch([WriteTasks, SummarizeCode, WriteCode, WriteCodeReview, FixBug, WriteCodePlanAndChange])  # 监听这些动作
        self.code_todos = []  # 初始化代码待办事项
        self.summarize_todos = []  # 初始化总结待办事项
        self.next_todo_action = any_to_name(WriteCode)  # 下一个动作为写代码

    @staticmethod
    def _parse_tasks(task_msg: Document) -> list[str]:
        """解析任务消息并返回任务列表"""
        m = json.loads(task_msg.content)
        return m.get(TASK_LIST.key) or m.get(REFINED_TASK_LIST.key)

    async def _act_sp_with_cr(self, review=False) -> Set[str]:
        """处理代码任务并执行代码审查（如果需要）"""
        changed_files = set()  # 存储改变的文件
        for todo in self.code_todos:
            # 从历史数据中选择关键信息，以减少提示长度（根据人类经验总结）
            coding_context = await todo.run()  # 执行代码任务
            # 如果需要代码审查
            if review:
                action = WriteCodeReview(
                    i_context=coding_context,
                    repo=self.repo,
                    input_args=self.input_args,
                    context=self.context,
                    llm=self.llm,
                )
                self._init_action(action)  # 初始化代码审查动作
                coding_context = await action.run()  # 执行代码审查

            # 记录代码的依赖关系
            dependencies = {coding_context.design_doc.root_relative_path, coding_context.task_doc.root_relative_path}
            if self.config.inc:
                dependencies.add(coding_context.code_plan_and_change_doc.root_relative_path)
            await self.repo.srcs.save(
                filename=coding_context.filename,
                dependencies=list(dependencies),
                content=coding_context.code_doc.content,
            )
            changed_files.add(coding_context.code_doc.filename)  # 将修改的文件添加到集合中
        if not changed_files:
            logger.info("Nothing has changed.")  # 如果没有文件更改
        return changed_files

    async def _act(self) -> Message | None:
        """根据是否使用代码审查来决定执行的动作"""
        if self.rc.todo is None:
            return None
        if isinstance(self.rc.todo, WriteCodePlanAndChange):
            self.next_todo_action = any_to_name(WriteCode)
            return await self._act_code_plan_and_change()
        if isinstance(self.rc.todo, WriteCode):
            self.next_todo_action = any_to_name(SummarizeCode)
            return await self._act_write_code()
        if isinstance(self.rc.todo, SummarizeCode):
            self.next_todo_action = any_to_name(WriteCode)
            return await self._act_summarize()
        return await self.rc.todo.run(self.rc.history)

    async def _act_write_code(self):
        """执行写代码动作"""
        await self._act_sp_with_cr(review=self.use_code_review)
        return AIMessage(
            content="", cause_by=WriteCodeReview if self.use_code_review else WriteCode, send_to=MESSAGE_ROUTE_TO_SELF
        )

    async def _act_summarize(self):
        """执行总结代码动作"""
        tasks = []
        for todo in self.summarize_todos:
            if self.n_summarize >= self.config.max_auto_summarize_code:
                break
            summary = await todo.run()  # 执行总结任务
            summary_filename = Path(todo.i_context.design_filename).with_suffix(".md").name
            dependencies = {todo.i_context.design_filename, todo.i_context.task_filename}
            for filename in todo.i_context.codes_filenames:
                rpath = self.repo.src_relative_path / filename
                dependencies.add(str(rpath))
            await self.repo.resources.code_summary.save(
                filename=summary_filename, content=summary, dependencies=dependencies
            )
            is_pass, reason = await self._is_pass(summary)
            if not is_pass:
                todo.i_context.reason = reason
                tasks.append(todo.i_context.model_dump())  # 将不合格的任务添加到任务列表

                await self.repo.docs.code_summary.save(
                    filename=Path(todo.i_context.design_filename).name,
                    content=todo.i_context.model_dump_json(),
                    dependencies=dependencies,
                )
            else:
                await self.repo.docs.code_summary.delete(filename=Path(todo.i_context.design_filename).name)
        self.summarize_todos = []  # 清空总结待办事项
        logger.info(f"--max-auto-summarize-code={self.config.max_auto_summarize_code}")
        if not tasks or self.config.max_auto_summarize_code == 0:
            self.n_summarize = 0  # 如果没有任务或者自动总结次数已达上限，重置总结次数
            kvs = self.input_args.model_dump()
            kvs["changed_src_filenames"] = [
                str(self.repo.srcs.workdir / i) for i in list(self.repo.srcs.changed_files.keys())
            ]
            if self.repo.docs.code_plan_and_change.changed_files:
                kvs["changed_code_plan_and_change_filenames"] = [
                    str(self.repo.docs.code_plan_and_change.workdir / i)
                    for i in list(self.repo.docs.code_plan_and_change.changed_files.keys())
                ]
            if self.repo.docs.code_summary.changed_files:
                kvs["changed_code_summary_filenames"] = [
                    str(self.repo.docs.code_summary.workdir / i)
                    for i in list(self.repo.docs.code_summary.changed_files.keys())
                ]
            return AIMessage(
                content=f"Coding is complete. The source code is at {self.repo.workdir.name}/{self.repo.srcs.root_path}, containing: "
                + "\n".join(
                    list(self.repo.resources.code_summary.changed_files.keys())
                    + list(self.repo.srcs.changed_files.keys())
                    + list(self.repo.resources.code_plan_and_change.changed_files.keys())
                ),
                instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="SummarizeCodeOutput"),
                cause_by=SummarizeCode,
                send_to="Edward",  # 发送给 QaEngineer
            )
        self.n_summarize += 1 if self.config.max_auto_summarize_code > self.n_summarize else 0
        return AIMessage(content="", cause_by=SummarizeCode, send_to=MESSAGE_ROUTE_TO_SELF)

    async def _act_code_plan_and_change(self):
        """编写代码计划和变更，指导后续的写代码和代码审查"""
        node = await self.rc.todo.run()
        code_plan_and_change = node.instruct_content.model_dump_json()
        dependencies = {
            REQUIREMENT_FILENAME,
            str(Path(self.rc.todo.i_context.prd_filename).relative_to(self.repo.workdir)),
            str(Path(self.rc.todo.i_context.design_filename).relative_to(self.repo.workdir)),
            str(Path(self.rc.todo.i_context.task_filename).relative_to(self.repo.workdir)),
        }
        code_plan_and_change_filepath = Path(self.rc.todo.i_context.design_filename)
        await self.repo.docs.code_plan_and_change.save(
            filename=code_plan_and_change_filepath.name, content=code_plan_and_change, dependencies=dependencies
        )
        await self.repo.resources.code_plan_and_change.save(
            filename=code_plan_and_change_filepath.with_suffix(".md").name,
            content=node.content,
            dependencies=dependencies,
        )

        return AIMessage(content="", cause_by=WriteCodePlanAndChange, send_to=MESSAGE_ROUTE_TO_SELF)

    async def _is_pass(self, summary) -> (str, str):
        # 调用LLM进行评估，判断是否通过
        rsp = await self.llm.aask(msg=IS_PASS_PROMPT.format(context=summary), stream=False)
        logger.info(rsp)  # 记录响应内容
        if "YES" in rsp:
            return True, rsp  # 如果响应包含 "YES"，表示通过
        return False, rsp  # 否则，返回不通过

    async def _think(self) -> bool:
        # 判断是否有新闻
        if not self.rc.news:
            return False  # 如果没有新闻，返回False
        msg = self.rc.news[0]  # 获取最新的消息
        input_args = msg.instruct_content  # 获取消息中的指令内容
        if msg.cause_by in {any_to_str(WriteTasks), any_to_str(FixBug)}:
            # 如果消息来源于 "WriteTasks" 或 "FixBug"，则进行处理
            self.input_args = input_args
            self.repo = ProjectRepo(input_args.project_path)  # 获取项目路径
            if self.repo.src_relative_path is None:
                path = get_project_srcs_path(self.repo.workdir)  # 获取源代码路径
                self.repo.with_src_path(path)  # 设置源代码路径
        # 定义各种任务的过滤器
        write_plan_and_change_filters = any_to_str_set([PrepareDocuments, WriteTasks, FixBug])
        write_code_filters = any_to_str_set([WriteTasks, WriteCodePlanAndChange, SummarizeCode])
        summarize_code_filters = any_to_str_set([WriteCode, WriteCodeReview])
        if self.config.inc and msg.cause_by in write_plan_and_change_filters:
            # 如果配置为增量更新并且消息类型为 "WriteCodePlanAndChange"，则执行相关操作
            logger.debug(f"TODO WriteCodePlanAndChange:{msg.model_dump_json()}")
            await self._new_code_plan_and_change_action(cause_by=msg.cause_by)
            return bool(self.rc.todo)
        if msg.cause_by in write_code_filters:
            # 如果消息类型为 "WriteCode"，则执行相关操作
            logger.debug(f"TODO WriteCode:{msg.model_dump_json()}")
            await self._new_code_actions()
            return bool(self.rc.todo)
        if msg.cause_by in summarize_code_filters and msg.sent_from == any_to_str(self):
            # 如果消息类型为 "SummarizeCode" 且消息来源于当前实例，则执行相关操作
            logger.debug(f"TODO SummarizeCode:{msg.model_dump_json()}")
            await self._new_summarize_actions()
            return bool(self.rc.todo)
        return False  # 如果没有匹配的条件，返回False

    async def _new_coding_context(self, filename, dependency) -> Optional[CodingContext]:
        # 获取旧的代码文档
        old_code_doc = await self.repo.srcs.get(filename)
        if not old_code_doc:
            old_code_doc = Document(root_path=str(self.repo.src_relative_path), filename=filename, content="")
        dependencies = {Path(i) for i in await dependency.get(old_code_doc.root_relative_path)}  # 获取依赖项
        task_doc = None
        design_doc = None
        code_plan_and_change_doc = await self._get_any_code_plan_and_change() if await self._is_fixbug() else None
        # 遍历依赖项，根据不同的父路径加载相关文档
        for i in dependencies:
            if str(i.parent) == TASK_FILE_REPO:
                task_doc = await self.repo.docs.task.get(i.name)
            elif str(i.parent) == SYSTEM_DESIGN_FILE_REPO:
                design_doc = await self.repo.docs.system_design.get(i.name)
            elif str(i.parent) == CODE_PLAN_AND_CHANGE_FILE_REPO:
                code_plan_and_change_doc = await self.repo.docs.code_plan_and_change.get(i.name)
        if not task_doc or not design_doc:
            if filename == "__init__.py":  # 特殊文件不处理
                return None
            logger.error(f'Detected source code "{filename}" from an unknown origin.')  # 如果未找到相关文档，记录错误
            raise ValueError(f'Detected source code "{filename}" from an unknown origin.')
        # 创建并返回新的CodingContext对象
        context = CodingContext(
            filename=filename,
            design_doc=design_doc,
            task_doc=task_doc,
            code_doc=old_code_doc,
            code_plan_and_change_doc=code_plan_and_change_doc,
        )
        return context

    async def _new_coding_doc(self, filename, dependency) -> Optional[Document]:
        # 获取并创建新的代码上下文
        context = await self._new_coding_context(filename, dependency)
        if not context:
            return None  # 如果是 "__init__.py" 文件，直接返回 None
        # 创建并返回新的Document对象
        coding_doc = Document(
            root_path=str(self.repo.src_relative_path), filename=filename, content=context.model_dump_json()
        )
        return coding_doc

    async def _new_code_actions(self):
        # 检查是否是bug修复
        bug_fix = await self._is_fixbug()
        # 获取更改的源代码文件
        changed_src_files = self.repo.srcs.changed_files
        if self.context.kwargs.src_filename:
            changed_src_files = {self.context.kwargs.src_filename: ChangeType.UNTRACTED}
        if bug_fix:
            changed_src_files = self.repo.srcs.all_files
        changed_files = Documents()
        # 获取更改的任务文件
        if hasattr(self.input_args, "changed_task_filenames"):
            changed_task_filenames = self.input_args.changed_task_filenames
        else:
            changed_task_filenames = [
                str(self.repo.docs.task.workdir / i) for i in list(self.repo.docs.task.changed_files.keys())
            ]
        for filename in changed_task_filenames:
            task_filename = Path(filename)
            design_filename = None
            # 获取更改的系统设计文件
            if hasattr(self.input_args, "changed_system_design_filenames"):
                changed_system_design_filenames = self.input_args.changed_system_design_filenames
            else:
                changed_system_design_filenames = [
                    str(self.repo.docs.system_design.workdir / i)
                    for i in list(self.repo.docs.system_design.changed_files.keys())
                ]
            for i in changed_system_design_filenames:
                if task_filename.name == Path(i).name:
                    design_filename = Path(i)
                    break
            # 获取更改的代码计划和变更文件
            code_plan_and_change_filename = None
            if hasattr(self.input_args, "changed_code_plan_and_change_filenames"):
                changed_code_plan_and_change_filenames = self.input_args.changed_code_plan_and_change_filenames
            else:
                changed_code_plan_and_change_filenames = [
                    str(self.repo.docs.code_plan_and_change.workdir / i)
                    for i in list(self.repo.docs.code_plan_and_change.changed_files.keys())
                ]
            for i in changed_code_plan_and_change_filenames:
                if task_filename.name == Path(i).name:
                    code_plan_and_change_filename = Path(i)
                    break
            # 加载任务文档、设计文档和代码计划文档
            design_doc = await Document.load(filename=design_filename, project_path=self.repo.workdir)
            task_doc = await Document.load(filename=task_filename, project_path=self.repo.workdir)
            code_plan_and_change_doc = await Document.load(
                filename=code_plan_and_change_filename, project_path=self.repo.workdir
            )
            # 解析任务并初始化Python文件夹
            task_list = self._parse_tasks(task_doc)
            await self._init_python_folder(task_list)
            for task_filename in task_list:
                if self.context.kwargs.src_filename and task_filename != self.context.kwargs.src_filename:
                    continue
                old_code_doc = await self.repo.srcs.get(task_filename)
                if not old_code_doc:
                    old_code_doc = Document(
                        root_path=str(self.repo.src_relative_path), filename=task_filename, content=""
                    )
                if not code_plan_and_change_doc:
                    context = CodingContext(
                        filename=task_filename, design_doc=design_doc, task_doc=task_doc, code_doc=old_code_doc
                    )
                else:
                    context = CodingContext(
                        filename=task_filename,
                        design_doc=design_doc,
                        task_doc=task_doc,
                        code_doc=old_code_doc,
                        code_plan_and_change_doc=code_plan_and_change_doc,
                    )
                coding_doc = Document(
                    root_path=str(self.repo.src_relative_path),
                    filename=task_filename,
                    content=context.model_dump_json(),
                )
                if task_filename in changed_files.docs:
                    logger.warning(
                        f"Log to expose potential conflicts: {coding_doc.model_dump_json()} & "
                        f"{changed_files.docs[task_filename].model_dump_json()}"
                    )
                changed_files.docs[task_filename] = coding_doc
        self.code_todos = [
            WriteCode(i_context=i, repo=self.repo, input_args=self.input_args, context=self.context, llm=self.llm)
            for i in changed_files.docs.values()
        ]
        # 处理用户直接修改的代码
        dependency = await self.repo.git_repo.get_dependency()
        for filename in changed_src_files:
            if filename in changed_files.docs:
                continue
            coding_doc = await self._new_coding_doc(filename=filename, dependency=dependency)
            if not coding_doc:
                continue  # `__init__.py` 文件直接跳过
            changed_files.docs[filename] = coding_doc
            self.code_todos.append(
                WriteCode(
                    i_context=coding_doc, repo=self.repo, input_args=self.input_args, context=self.context, llm=self.llm
                )
            )

        if self.code_todos:
            self.set_todo(self.code_todos[0])

    async def _new_summarize_actions(self):
        # 获取所有源文件
        src_files = self.repo.srcs.all_files
        # 为每对 (system_design_doc, task_doc) 生成一个 SummarizeCode 行动
        summarizations = defaultdict(list)
        for filename in src_files:
            # 获取文件的依赖项
            dependencies = await self.repo.srcs.get_dependency(filename=filename)
            # 加载代码总结上下文
            ctx = CodeSummarizeContext.loads(filenames=list(dependencies))
            summarizations[ctx].append(filename)
        # 对每个上下文执行 SummarizeCode 行动
        for ctx, filenames in summarizations.items():
            # 如果没有设计文件或任务文件，则跳过
            if not ctx.design_filename or not ctx.task_filename:
                continue  # 由 `init_python_folder` 创建的 `__init__.py`
            ctx.codes_filenames = filenames
            new_summarize = SummarizeCode(
                i_context=ctx, repo=self.repo, input_args=self.input_args, context=self.context, llm=self.llm
            )
            # 检查是否已存在相同任务文件的 summarize action
            for i, act in enumerate(self.summarize_todos):
                if act.i_context.task_filename == new_summarize.i_context.task_filename:
                    self.summarize_todos[i] = new_summarize
                    new_summarize = None
                    break
            # 如果没有找到相同任务文件的 summarize action，添加新 action
            if new_summarize:
                self.summarize_todos.append(new_summarize)
        # 如果有待处理的 summarize actions，设置第一个作为待办事项
        if self.summarize_todos:
            self.set_todo(self.summarize_todos[0])

    async def _new_code_plan_and_change_action(self, cause_by: str):
        """为后续的待办事项创建一个 WriteCodePlanAndChange 行动。"""
        options = {}
        # 如果不是 FixBug 操作，加载需求文档
        if cause_by != any_to_str(FixBug):
            requirement_doc = await Document.load(filename=self.input_args.requirements_filename)
            options["requirement"] = requirement_doc.content
        else:
            # 如果是 FixBug 操作，加载问题文档
            fixbug_doc = await Document.load(filename=self.input_args.issue_filename)
            options["issue"] = fixbug_doc.content
        # 如果存在多个不相关的需求文件，这段逻辑会出错
        if hasattr(self.input_args, "changed_prd_filenames"):
            code_plan_and_change_ctx = CodePlanAndChangeContext(
                requirement=options.get("requirement", ""),
                issue=options.get("issue", ""),
                prd_filename=self.input_args.changed_prd_filenames[0],
                design_filename=self.input_args.changed_system_design_filenames[0],
                task_filename=self.input_args.changed_task_filenames[0],
            )
        else:
            code_plan_and_change_ctx = CodePlanAndChangeContext(
                requirement=options.get("requirement", ""),
                issue=options.get("issue", ""),
                prd_filename=str(self.repo.docs.prd.workdir / self.repo.docs.prd.all_files[0]),
                design_filename=str(self.repo.docs.system_design.workdir / self.repo.docs.system_design.all_files[0]),
                task_filename=str(self.repo.docs.task.workdir / self.repo.docs.task.all_files[0]),
            )
        # 创建 WriteCodePlanAndChange 行动并设置为待办事项
        self.rc.todo = WriteCodePlanAndChange(
            i_context=code_plan_and_change_ctx,
            repo=self.repo,
            input_args=self.input_args,
            context=self.context,
            llm=self.llm,
        )

    @property
    def action_description(self) -> str:
        """AgentStore 使用此属性显示当前角色应该执行的操作描述。"""
        return self.next_todo_action

    async def _init_python_folder(self, task_list: List[str]):
        # 为每个任务文件初始化 Python 文件夹
        for i in task_list:
            filename = Path(i)
            if filename.suffix != ".py":
                continue
            workdir = self.repo.srcs.workdir / filename.parent
            if not workdir.exists():
                workdir = self.repo.workdir / filename.parent
            # 调用 init_python_folder 函数初始化文件夹
            await init_python_folder(workdir)

    async def _is_fixbug(self) -> bool:
        """判断是否为 FixBug 操作。"""
        return bool(self.input_args and hasattr(self.input_args, "issue_filename"))

    async def _get_any_code_plan_and_change(self) -> Optional[Document]:
        """获取任何已更改的代码计划和变更文档。"""
        changed_files = self.repo.docs.code_plan_and_change.changed_files
        for filename in changed_files.keys():
            doc = await self.repo.docs.code_plan_and_change.get(filename)
            if doc and doc.content:
                return doc
        return None