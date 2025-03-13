#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.2 of RFC 135, add incremental iteration functionality.
            3. Move the document storage operations related to WritePRD from the save operation of WriteDesign.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
@Modified By: mashenquan, 2024/5/31. Implement Chapter 3 of RFC 236.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from metagpt.actions import Action, ActionOutput
from metagpt.actions.action_node import ActionNode
from metagpt.actions.fix_bug import FixBug
from metagpt.actions.write_prd_an import (
    COMPETITIVE_QUADRANT_CHART,
    PROJECT_NAME,
    REFINED_PRD_NODE,
    WP_IS_RELATIVE_NODE,
    WP_ISSUE_TYPE_NODE,
    WRITE_PRD_NODE,
)
from metagpt.const import (
    BUGFIX_FILENAME,
    COMPETITIVE_ANALYSIS_FILE_REPO,
    REQUIREMENT_FILENAME,
)
from metagpt.logs import logger
from metagpt.schema import AIMessage, Document, Documents, Message
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import (
    CodeParser,
    aread,
    awrite,
    rectify_pathname,
    save_json_to_markdown,
    to_markdown_code_block,
)
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.mermaid import mermaid_to_file
from metagpt.utils.project_repo import ProjectRepo
from metagpt.utils.report import DocsReporter, GalleryReporter

CONTEXT_TEMPLATE = """
### Project Name
{project_name}

### Original Requirements
{requirements}

### Search Information
-
"""

NEW_REQ_TEMPLATE = """
### Legacy Content
{old_prd}

### New Requirements
{requirements}
"""


@register_tool(include_functions=["run"])
class WritePRD(Action):
    """WritePRD 处理以下几种情况：
    1. Bugfix：如果需求是一个 bug 修复，将生成 bugfix 文档。
    2. 新需求：如果需求是一个新需求，将生成 PRD 文档。
    3. 需求更新：如果需求是一个更新，将更新 PRD 文档。
    """

    repo: Optional[ProjectRepo] = Field(default=None, exclude=True)  # 项目仓库
    input_args: Optional[BaseModel] = Field(default=None, exclude=True)  # 输入参数

    async def run(
        self,
        with_messages: List[Message] = None,
        *,
        user_requirement: str = "",
        output_pathname: str = "",
        legacy_prd_filename: str = "",
        extra_info: str = "",
        **kwargs,
    ) -> Union[AIMessage, str]:
        """
        编写产品需求文档（PRD）。

        参数:
            user_requirement (str): 用户需求的描述字符串。
            output_pathname (str, optional): 文档的输出文件路径。默认为 ""。
            legacy_prd_filename (str, optional): 用作参考的旧 PRD 文件路径。默认为 ""。
            extra_info (str, optional): 附加信息，默认为 ""。
            **kwargs: 其他关键字参数。

        返回:
            str: 生成的产品需求文档的文件路径。

        示例:
            # 写一个新的 PRD（产品需求文档）
            >>> user_requirement = "写一个贪吃蛇游戏"
            >>> output_pathname = "snake_game/docs/prd.json"
            >>> extra_info = "如果有额外的信息"
            >>> write_prd = WritePRD()
            >>> result = await write_prd.run(user_requirement=user_requirement, output_pathname=output_pathname, extra_info=extra_info)
            >>> print(result)
            PRD 文件名: "/absolute/path/to/snake_game/docs/prd.json"

            # 重写现有 PRD（产品需求文档），并保存到新路径。
            >>> user_requirement = "写一个贪吃蛇游戏的 PRD，加入新的功能，如 Web UI"
            >>> legacy_prd_filename = "/absolute/path/to/snake_game/docs/prd.json"
            >>> output_pathname = "/absolute/path/to/snake_game/docs/prd_new.json"
            >>> extra_info = "如果有额外的信息"
            >>> write_prd = WritePRD()
            >>> result = await write_prd.run(user_requirement=user_requirement, legacy_prd_filename=legacy_prd_filename, extra_info=extra_info)
            >>> print(result)
            PRD 文件名: "/absolute/path/to/snake_game/docs/prd_new.json"
        """
        if not with_messages:
            return await self._execute_api(
                user_requirement=user_requirement,
                output_pathname=output_pathname,
                legacy_prd_filename=legacy_prd_filename,
                extra_info=extra_info,
            )

        self.input_args = with_messages[-1].instruct_content
        if not self.input_args:
            self.repo = ProjectRepo(self.context.kwargs.project_path)
            await self.repo.docs.save(filename=REQUIREMENT_FILENAME, content=with_messages[-1].content)
            self.input_args = AIMessage.create_instruct_value(
                kvs={
                    "project_path": self.context.kwargs.project_path,
                    "requirements_filename": str(self.repo.docs.workdir / REQUIREMENT_FILENAME),
                    "prd_filenames": [str(self.repo.docs.prd.workdir / i) for i in self.repo.docs.prd.all_files],
                },
                class_name="PrepareDocumentsOutput",
            )
        else:
            self.repo = ProjectRepo(self.input_args.project_path)
        req = await Document.load(filename=self.input_args.requirements_filename)
        docs: list[Document] = [
            await Document.load(filename=i, project_path=self.repo.workdir) for i in self.input_args.prd_filenames
        ]

        if not req:
            raise FileNotFoundError("未找到需求文档。")

        if await self._is_bugfix(req.content):
            logger.info(f"检测到 Bugfix: {req.content}")
            return await self._handle_bugfix(req)
        # 删除上轮的 bugfix 文件，以防冲突
        await self.repo.docs.delete(filename=BUGFIX_FILENAME)

        # 如果需求与其他文档相关，则更新它们，否则创建一个新文档
        if related_docs := await self.get_related_docs(req, docs):
            logger.info(f"检测到需求更新: {req.content}")
            await self._handle_requirement_update(req=req, related_docs=related_docs)
        else:
            logger.info(f"检测到新需求: {req.content}")
            await self._handle_new_requirement(req)

        kvs = self.input_args.model_dump()
        kvs["changed_prd_filenames"] = [
            str(self.repo.docs.prd.workdir / i) for i in list(self.repo.docs.prd.changed_files.keys())
        ]
        kvs["project_path"] = str(self.repo.workdir)
        kvs["requirements_filename"] = str(self.repo.docs.workdir / REQUIREMENT_FILENAME)
        self.context.kwargs.project_path = str(self.repo.workdir)
        return AIMessage(
            content="PRD 已完成。"
            + "\n".join(
                list(self.repo.docs.prd.changed_files.keys())
                + list(self.repo.resources.prd.changed_files.keys())
                + list(self.repo.resources.competitive_analysis.changed_files.keys())
            ),
            instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="WritePRDOutput"),
            cause_by=self,
        )

    async def _handle_bugfix(self, req: Document) -> AIMessage:
        # ... bugfix 逻辑 ...
        await self.repo.docs.save(filename=BUGFIX_FILENAME, content=req.content)
        await self.repo.docs.save(filename=REQUIREMENT_FILENAME, content="")
        return AIMessage(
            content=f"收到新问题: {BUGFIX_FILENAME}",
            cause_by=FixBug,
            instruct_content=AIMessage.create_instruct_value(
                {
                    "project_path": str(self.repo.workdir),
                    "issue_filename": str(self.repo.docs.workdir / BUGFIX_FILENAME),
                    "requirements_filename": str(self.repo.docs.workdir / REQUIREMENT_FILENAME),
                },
                class_name="IssueDetail",
            ),
            send_to="Alex",  # 工程师名称
        )

    async def _new_prd(self, requirement: str) -> ActionNode:
        project_name = self.project_name
        context = CONTEXT_TEMPLATE.format(requirements=requirement, project_name=project_name)
        exclude = [PROJECT_NAME.key] if project_name else []
        node = await WRITE_PRD_NODE.fill(
            req=context, llm=self.llm, exclude=exclude, schema=self.prompt_schema
        )  # schema=schema
        return node

    async def _handle_new_requirement(self, req: Document) -> ActionOutput:
        """处理新需求"""
        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "prd"}, "meta")
            node = await self._new_prd(req.content)
            await self._rename_workspace(node)
            new_prd_doc = await self.repo.docs.prd.save(
                filename=FileRepository.new_filename() + ".json", content=node.instruct_content.model_dump_json()
            )
            await self._save_competitive_analysis(new_prd_doc)
            md = await self.repo.resources.prd.save_pdf(doc=new_prd_doc)
            await reporter.async_report(self.repo.workdir / md.root_relative_path, "path")
            return Documents.from_iterable(documents=[new_prd_doc]).to_action_output()

    async def _handle_requirement_update(self, req: Document, related_docs: list[Document]) -> ActionOutput:
        # ... 需求更新逻辑 ...
        for doc in related_docs:
            await self._update_prd(req=req, prd_doc=doc)
        return Documents.from_iterable(documents=related_docs).to_action_output()

    async def _is_bugfix(self, context: str) -> bool:
        if not self.repo.code_files_exists():
            return False
        node = await WP_ISSUE_TYPE_NODE.fill(req=context, llm=self.llm)
        return node.get("issue_type") == "BUG"

    async def get_related_docs(self, req: Document, docs: list[Document]) -> list[Document]:
        """获取相关文档"""
        # 优化: 使用 gather 提高速度
        return [i for i in docs if await self._is_related(req, i)]

    async def _is_related(self, req: Document, old_prd: Document) -> bool:
        context = NEW_REQ_TEMPLATE.format(old_prd=old_prd.content, requirements=req.content)
        node = await WP_IS_RELATIVE_NODE.fill(req=context, llm=self.llm)
        return node.get("is_relative") == "YES"

    async def _merge(self, req: Document, related_doc: Document) -> Document:
        if not self.project_name:
            self.project_name = Path(self.project_path).name
        prompt = NEW_REQ_TEMPLATE.format(requirements=req.content, old_prd=related_doc.content)
        node = await REFINED_PRD_NODE.fill(req=prompt, llm=self.llm, schema=self.prompt_schema)
        related_doc.content = node.instruct_content.model_dump_json()
        await self._rename_workspace(node)
        return related_doc

    async def _update_prd(self, req: Document, prd_doc: Document) -> Document:
        # 使用 DocsReporter 进行异步报告
        async with DocsReporter(enable_llm_stream=True) as reporter:
            # 先报告 PRD 元数据
            await reporter.async_report({"type": "prd"}, "meta")
            # 合并请求文档和现有 PRD 文档
            new_prd_doc: Document = await self._merge(req=req, related_doc=prd_doc)
            # 保存新的 PRD 文档
            await self.repo.docs.prd.save_doc(doc=new_prd_doc)
            # 保存竞争分析图
            await self._save_competitive_analysis(new_prd_doc)
            # 保存 PDF 格式的 PRD
            md = await self.repo.resources.prd.save_pdf(doc=new_prd_doc)
            # 记录 PDF 的路径
            await reporter.async_report(self.repo.workdir / md.root_relative_path, "path")
        # 返回新的 PRD 文档
        return new_prd_doc

    async def _save_competitive_analysis(self, prd_doc: Document, output_filename: Path = None):
        # 解析 PRD 文档内容
        m = json.loads(prd_doc.content)
        # 获取竞争象限图表数据
        quadrant_chart = m.get(COMPETITIVE_QUADRANT_CHART.key)
        if not quadrant_chart:
            return
        # 设置输出文件路径
        pathname = output_filename or self.repo.workdir / COMPETITIVE_ANALYSIS_FILE_REPO / Path(prd_doc.filename).stem
        pathname.parent.mkdir(parents=True, exist_ok=True)
        # 将 mermaid 图表保存为文件
        await mermaid_to_file(self.config.mermaid.engine, quadrant_chart, pathname)
        # 获取 SVG 图像路径
        image_path = pathname.parent / f"{pathname.name}.svg"
        # 如果图像文件存在，报告该路径
        if image_path.exists():
            await GalleryReporter().async_report(image_path, "path")

    async def _rename_workspace(self, prd):
        # 如果项目名称为空，尝试从 PRD 或 Action 中提取项目名称
        if not self.project_name:
            if isinstance(prd, (ActionOutput, ActionNode)):
                ws_name = prd.instruct_content.model_dump()["Project Name"]
            else:
                ws_name = CodeParser.parse_str(block="Project Name", text=prd)
            if ws_name:
                self.project_name = ws_name
        # 如果仓库存在，重命名 Git 仓库的根目录
        if self.repo:
            self.repo.git_repo.rename_root(self.project_name)

    async def _execute_api(
        self, user_requirement: str, output_pathname: str, legacy_prd_filename: str, extra_info: str
    ) -> str:
        # 生成请求内容
        content = "#### User Requirements\n{user_requirement}\n#### Extra Info\n{extra_info}\n".format(
            user_requirement=to_markdown_code_block(val=user_requirement),
            extra_info=to_markdown_code_block(val=extra_info),
        )
        # 使用 DocsReporter 进行异步报告
        async with DocsReporter(enable_llm_stream=True) as reporter:
            # 先报告 PRD 元数据
            await reporter.async_report({"type": "prd"}, "meta")
            # 创建新的 PRD 文档
            req = Document(content=content)
            if not legacy_prd_filename:
                node = await self._new_prd(requirement=req.content)
                new_prd = Document(content=node.instruct_content.model_dump_json())
            else:
                # 读取现有 PRD 文档，并与请求文档合并
                content = await aread(filename=legacy_prd_filename)
                old_prd = Document(content=content)
                new_prd = await self._merge(req=req, related_doc=old_prd)

            # 设置输出文件路径
            if not output_pathname:
                output_pathname = self.config.workspace.path / "docs" / "prd.json"
            elif not Path(output_pathname).is_absolute():
                output_pathname = self.config.workspace.path / output_pathname
            output_pathname = rectify_pathname(path=output_pathname, default_filename="prd.json")
            # 保存 PRD 内容到文件
            await awrite(filename=output_pathname, data=new_prd.content)
            # 保存竞争分析结果
            competitive_analysis_filename = output_pathname.parent / f"{output_pathname.stem}-competitive-analysis"
            await self._save_competitive_analysis(prd_doc=new_prd, output_filename=Path(competitive_analysis_filename))
            # 将 PRD 内容保存为 markdown 文件
            md_output_filename = output_pathname.with_suffix(".md")
            await save_json_to_markdown(content=new_prd.content, output_filename=md_output_filename)
            # 报告 markdown 文件的路径
            await reporter.async_report(md_output_filename, "path")
        # 返回生成的 PRD 文件路径信息
        return f'PRD filename: "{str(output_pathname)}". The  product requirement document (PRD) has been completed.'
