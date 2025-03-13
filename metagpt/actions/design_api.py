#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : design_api.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.3 of RFC 135, add incremental iteration functionality.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
@Modified By: mashenquan, 2024/5/31. Implement Chapter 3 of RFC 236.
"""
import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from metagpt.actions import Action
from metagpt.actions.design_api_an import (
    DATA_STRUCTURES_AND_INTERFACES,
    DESIGN_API_NODE,
    PROGRAM_CALL_FLOW,
    REFINED_DATA_STRUCTURES_AND_INTERFACES,
    REFINED_DESIGN_NODE,
    REFINED_PROGRAM_CALL_FLOW,
)
from metagpt.const import DATA_API_DESIGN_FILE_REPO, SEQ_FLOW_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import AIMessage, Document, Documents, Message
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import (
    aread,
    awrite,
    rectify_pathname,
    save_json_to_markdown,
    to_markdown_code_block,
)
from metagpt.utils.mermaid import mermaid_to_file
from metagpt.utils.project_repo import ProjectRepo
from metagpt.utils.report import DocsReporter, GalleryReporter

NEW_REQ_TEMPLATE = """
### Legacy Content
{old_design}

### New Requirements
{context}
"""


# 定义一个新的设计编写工具，继承自Action类
@register_tool(include_functions=["run"])
class WriteDesign(Action):
    # 定义类的属性
    name: str = ""  # 设计名称
    i_context: Optional[str] = None  # 可选的上下文信息
    desc: str = (
        "基于PRD文档，思考系统设计，设计对应的API、数据结构、库表、流程和路径。"
        "请提供清晰、详细的设计和反馈。"
    )  # 设计描述
    repo: Optional[ProjectRepo] = Field(default=None, exclude=True)  # 可选的项目仓库
    input_args: Optional[BaseModel] = Field(default=None, exclude=True)  # 可选的输入参数

    # 运行该设计生成的逻辑
    async def run(
        self,
        with_messages: List[Message] = None,  # 可选的消息列表
        *,
        user_requirement: str = "",  # 用户需求
        prd_filename: str = "",  # PRD文档文件名
        legacy_design_filename: str = "",  # 旧设计文件名
        extra_info: str = "",  # 额外信息
        output_pathname: str = "",  # 输出文件路径
        **kwargs,
    ) -> Union[AIMessage, str]:
        """
        写系统设计。

        参数:
            user_requirement (str): 用户对系统设计的要求。
            prd_filename (str, optional): 产品需求文档（PRD）的文件名。
            legacy_design_filename (str, optional): 旧版设计文档的文件名。
            extra_info (str, optional): 系统设计中包含的附加信息。
            output_pathname (str, optional): 生成文档的输出文件路径。

        返回:
            str: 生成的系统设计文件路径。
        """

        # 如果没有提供消息，则执行API请求，生成新的设计
        if not with_messages:
            return await self._execute_api(
                user_requirement=user_requirement,
                prd_filename=prd_filename,
                legacy_design_filename=legacy_design_filename,
                extra_info=extra_info,
                output_pathname=output_pathname,
            )

        # 获取消息中的最新指令内容
        self.input_args = with_messages[-1].instruct_content
        self.repo = ProjectRepo(self.input_args.project_path)  # 获取项目仓库
        changed_prds = self.input_args.changed_prd_filenames  # 获取修改过的PRD文件
        changed_system_designs = [
            str(self.repo.docs.system_design.workdir / i)
            for i in list(self.repo.docs.system_design.changed_files.keys())  # 获取修改过的系统设计文件
        ]

        # 对于那些发生变化的PRD和设计文件，重新生成设计内容
        changed_files = Documents()
        for filename in changed_prds:
            doc = await self._update_system_design(filename=filename)  # 更新系统设计
            changed_files.docs[filename] = doc

        # 更新系统设计
        for filename in changed_system_designs:
            if filename in changed_files.docs:
                continue
            doc = await self._update_system_design(filename=filename)
            changed_files.docs[filename] = doc

        # 如果没有修改文件，打印日志
        if not changed_files.docs:
            logger.info("Nothing has changed.")

        # 等待所有的系统设计文件处理完成后再发送发布消息，为后续步骤提供优化空间
        kvs = self.input_args.model_dump()
        kvs["changed_system_design_filenames"] = [
            str(self.repo.docs.system_design.workdir / i)
            for i in list(self.repo.docs.system_design.changed_files.keys())  # 获取修改过的文件路径
        ]
        return AIMessage(
            content="Designing is complete. "
            + "\n".join(
                list(self.repo.docs.system_design.changed_files.keys())
                + list(self.repo.resources.data_api_design.changed_files.keys())
                + list(self.repo.resources.seq_flow.changed_files.keys())
            ),
            instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="WriteDesignOutput"),  # 返回设计输出的内容
            cause_by=self,
        )

    # 创建新的系统设计
    async def _new_system_design(self, context):
        node = await DESIGN_API_NODE.fill(req=context, llm=self.llm, schema=self.prompt_schema)
        return node

    # 合并PRD文档与现有的系统设计文档
    async def _merge(self, prd_doc, system_design_doc):
        context = NEW_REQ_TEMPLATE.format(old_design=system_design_doc.content, context=prd_doc.content)
        node = await REFINED_DESIGN_NODE.fill(req=context, llm=self.llm, schema=self.prompt_schema)
        system_design_doc.content = node.instruct_content.model_dump_json()
        return system_design_doc

    # 更新系统设计文件
    async def _update_system_design(self, filename) -> Document:
        root_relative_path = Path(filename).relative_to(self.repo.workdir)  # 获取相对路径
        prd = await Document.load(filename=filename, project_path=self.repo.workdir)  # 加载PRD文件
        old_system_design_doc = await self.repo.docs.system_design.get(root_relative_path.name)  # 获取旧的系统设计文档

        async with DocsReporter(enable_llm_stream=True) as reporter:  # 使用报告器生成新的设计
            await reporter.async_report({"type": "design"}, "meta")
            if not old_system_design_doc:  # 如果没有找到旧的设计，创建一个新的
                system_design = await self._new_system_design(context=prd.content)
                doc = await self.repo.docs.system_design.save(
                    filename=prd.filename,
                    content=system_design.instruct_content.model_dump_json(),
                    dependencies={prd.root_relative_path},
                )
            else:  # 如果有旧设计，合并新的PRD和旧设计
                doc = await self._merge(prd_doc=prd, system_design_doc=old_system_design_doc)
                await self.repo.docs.system_design.save_doc(doc=doc, dependencies={prd.root_relative_path})
            await self._save_data_api_design(doc)  # 保存数据API设计
            await self._save_seq_flow(doc)  # 保存流程图
            md = await self.repo.resources.system_design.save_pdf(doc=doc)  # 保存PDF格式的设计文档
            await reporter.async_report(self.repo.workdir / md.root_relative_path, "path")
        return doc

    # 保存数据API设计到文件
    async def _save_data_api_design(self, design_doc, output_filename: Path = None):
        m = json.loads(design_doc.content)
        data_api_design = m.get(DATA_STRUCTURES_AND_INTERFACES.key) or m.get(REFINED_DATA_STRUCTURES_AND_INTERFACES.key)
        if not data_api_design:
            return
        pathname = output_filename or self.repo.workdir / DATA_API_DESIGN_FILE_REPO / Path(
            design_doc.filename
        ).with_suffix("")
        await self._save_mermaid_file(data_api_design, pathname)  # 保存mermaid图表
        logger.info(f"Save class view to {str(pathname)}")

    # 保存流程图到文件
    async def _save_seq_flow(self, design_doc, output_filename: Path = None):
        m = json.loads(design_doc.content)
        seq_flow = m.get(PROGRAM_CALL_FLOW.key) or m.get(REFINED_PROGRAM_CALL_FLOW.key)
        if not seq_flow:
            return
        pathname = output_filename or self.repo.workdir / Path(SEQ_FLOW_FILE_REPO) / Path(
            design_doc.filename
        ).with_suffix("")
        await self._save_mermaid_file(seq_flow, pathname)  # 保存mermaid图表
        logger.info(f"Saving sequence flow to {str(pathname)}")

    # 保存mermaid文件到指定路径
    async def _save_mermaid_file(self, data: str, pathname: Path):
        pathname.parent.mkdir(parents=True, exist_ok=True)  # 创建父级目录
        await mermaid_to_file(self.config.mermaid.engine, data, pathname)  # 将mermaid图表保存到文件
        image_path = pathname.parent / f"{pathname.name}.svg"
        if image_path.exists():
            await GalleryReporter().async_report(image_path, "path")  # 将图像报告上传

    # 执行API请求生成系统设计
    async def _execute_api(
        self,
        user_requirement: str = "",
        prd_filename: str = "",
        legacy_design_filename: str = "",
        extra_info: str = "",
        output_pathname: str = "",
    ) -> str:
        prd_content = ""
        if prd_filename:
            prd_filename = rectify_pathname(path=prd_filename, default_filename="prd.json")  # 修正路径名
            prd_content = await aread(filename=prd_filename)  # 读取PRD内容
        context = "### User Requirements\n{user_requirement}\n### Extra_info\n{extra_info}\n### PRD\n{prd}\n".format(
            user_requirement=to_markdown_code_block(user_requirement),
            extra_info=to_markdown_code_block(extra_info),
            prd=to_markdown_code_block(prd_content),
        )
        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "design"}, "meta")
            if not legacy_design_filename:  # 如果没有旧设计文件，创建新的
                node = await self._new_system_design(context=context)
                design = Document(content=node.instruct_content.model_dump_json())
            else:
                design = await self._update_system_design(filename=legacy_design_filename)  # 更新设计
            design = await self.repo.docs.system_design.save(
                filename=prd_filename,
                content=design.content,
                dependencies={prd_filename},
            )  # 保存新的设计文档
            await self._save_data_api_design(design)  # 保存数据API设计
            await self._save_seq_flow(design)  # 保存流程设计
            md = await self.repo.resources.system_design.save_pdf(design)  # 保存PDF
            await reporter.async_report(self.repo.workdir / md.root_relative_path, "path")
        return f"系统设计文件保存在{output_pathname}。"