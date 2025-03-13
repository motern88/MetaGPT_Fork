#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : prepare_documents.py
@Desc: PrepareDocuments Action: initialize project folder and add new requirements to docs/requirements.txt.
        RFC 135 2.2.3.5.1.
"""
import shutil
from pathlib import Path
from typing import Dict, Optional

from metagpt.actions import Action, UserRequirement
from metagpt.const import REQUIREMENT_FILENAME
from metagpt.logs import logger
from metagpt.schema import AIMessage
from metagpt.utils.common import any_to_str
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.project_repo import ProjectRepo


class PrepareDocuments(Action):
    """PrepareDocuments Action: 初始化项目文件夹，并将新的需求添加到 docs/requirements.txt 文件中。"""

    name: str = "PrepareDocuments"  # 动作的名称
    i_context: Optional[str] = None  # 可选的上下文
    key_descriptions: Optional[Dict[str, str]] = None  # 可选的描述键值对
    send_to: str  # 发送消息的目标

    def __init__(self, **kwargs):
        """初始化方法，若 key_descriptions 未提供则设置默认值"""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        if not self.key_descriptions:
            # 若未提供 key_descriptions，则设置默认值
            self.key_descriptions = {"project_path": '如果“Original Requirement”中有项目路径，则为该路径'}

    @property
    def config(self):
        """返回上下文中的配置"""
        return self.context.config

    def _init_repo(self) -> ProjectRepo:
        """初始化 Git 环境"""
        if not self.config.project_path:
            # 若项目路径未指定，则创建新的路径
            name = self.config.project_name or FileRepository.new_filename()
            path = Path(self.config.workspace.path) / name
        else:
            path = Path(self.config.project_path)  # 使用指定的项目路径

        # 如果路径存在且配置中未指定 `inc`，则删除该路径
        if path.exists() and not self.config.inc:
            shutil.rmtree(path)

        # 设置项目路径和 `inc` 配置
        self.context.kwargs.project_path = path
        self.context.kwargs.inc = self.config.inc
        return ProjectRepo(path)  # 返回初始化后的项目仓库

    async def run(self, with_messages, **kwargs):
        """创建并初始化工作区文件夹，初始化 Git 环境"""
        # 从与消息关联的 UserRequirement 中提取用户需求
        user_requirements = [i for i in with_messages if i.cause_by == any_to_str(UserRequirement)]
        if not self.config.project_path and user_requirements and self.key_descriptions:
            # 如果没有指定项目路径且存在用户需求，则解析并设置相关资源
            args = await user_requirements[0].parse_resources(llm=self.llm, key_descriptions=self.key_descriptions)
            for k, v in args.items():
                if not v or k in ["resources", "reason"]:  # 排除不需要的字段
                    continue
                self.context.kwargs.set(k, v)  # 将资源设置到上下文中
                logger.info(f"{k}={v}")  # 日志记录
            if self.context.kwargs.project_path:
                # 如果项目路径已经设置，更新项目配置
                self.config.update_via_cli(
                    project_path=self.context.kwargs.project_path,
                    project_name="",
                    inc=False,
                    reqa_file=self.context.kwargs.reqa_file or "",
                    max_auto_summarize_code=0,
                )

        repo = self._init_repo()  # 初始化 Git 仓库

        # 将从用户需求中提取到的新需求写入 `docs/requirements.txt` 文件
        await repo.docs.save(filename=REQUIREMENT_FILENAME, content=with_messages[0].content)

        # 发送消息通知 WritePRD 动作，指示其使用 `docs/requirements.txt` 和 `docs/prd/` 目录中的内容处理需求
        return AIMessage(
            content="",  # 空内容
            instruct_content=AIMessage.create_instruct_value(
                kvs={  # 提供给 WritePRD 的参数
                    "project_path": str(repo.workdir),
                    "requirements_filename": str(repo.docs.workdir / REQUIREMENT_FILENAME),
                    "prd_filenames": [str(repo.docs.prd.workdir / i) for i in repo.docs.prd.all_files],
                },
                class_name="PrepareDocumentsOutput",  # 输出的类名
            ),
            cause_by=self,  # 当前动作是消息的来源
            send_to=self.send_to,  # 发送目标
        )
