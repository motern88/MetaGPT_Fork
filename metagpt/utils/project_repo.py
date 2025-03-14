#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/8
@Author  : mashenquan
@File    : project_repo.py
@Desc    : Wrapper for GitRepository and FileRepository of project.
    Implementation of Chapter 4.6 of https://deepwisdom.feishu.cn/wiki/CUK4wImd7id9WlkQBNscIe9cnqh
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from metagpt.const import (
    CLASS_VIEW_FILE_REPO,
    CODE_PLAN_AND_CHANGE_FILE_REPO,
    CODE_PLAN_AND_CHANGE_PDF_FILE_REPO,
    CODE_SUMMARIES_FILE_REPO,
    CODE_SUMMARIES_PDF_FILE_REPO,
    COMPETITIVE_ANALYSIS_FILE_REPO,
    DATA_API_DESIGN_FILE_REPO,
    DOCS_FILE_REPO,
    GRAPH_REPO_FILE_REPO,
    PRD_PDF_FILE_REPO,
    PRDS_FILE_REPO,
    REQUIREMENT_FILENAME,
    RESOURCES_FILE_REPO,
    SD_OUTPUT_FILE_REPO,
    SEQ_FLOW_FILE_REPO,
    SYSTEM_DESIGN_FILE_REPO,
    SYSTEM_DESIGN_PDF_FILE_REPO,
    TASK_FILE_REPO,
    TASK_PDF_FILE_REPO,
    TEST_CODES_FILE_REPO,
    TEST_OUTPUTS_FILE_REPO,
    VISUAL_GRAPH_REPO_FILE_REPO,
)
from metagpt.utils.common import get_project_srcs_path
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.git_repository import GitRepository


class DocFileRepositories(FileRepository):
    """
    该类包含与文档相关的文件仓库，包括不同的文档类别。
    """
    prd: FileRepository  # 产品需求文件仓库
    system_design: FileRepository  # 系统设计文件仓库
    task: FileRepository  # 任务文件仓库
    code_summary: FileRepository  # 代码总结文件仓库
    graph_repo: FileRepository  # 图形库文件仓库
    class_view: FileRepository  # 类视图文件仓库
    code_plan_and_change: FileRepository  # 代码计划和更改文件仓库

    def __init__(self, git_repo):
        """
        初始化方法，创建与文档相关的各个文件仓库。
        """
        super().__init__(git_repo=git_repo, relative_path=DOCS_FILE_REPO)

        self.prd = git_repo.new_file_repository(relative_path=PRDS_FILE_REPO)
        self.system_design = git_repo.new_file_repository(relative_path=SYSTEM_DESIGN_FILE_REPO)
        self.task = git_repo.new_file_repository(relative_path=TASK_FILE_REPO)
        self.code_summary = git_repo.new_file_repository(relative_path=CODE_SUMMARIES_FILE_REPO)
        self.graph_repo = git_repo.new_file_repository(relative_path=GRAPH_REPO_FILE_REPO)
        self.class_view = git_repo.new_file_repository(relative_path=CLASS_VIEW_FILE_REPO)
        self.code_plan_and_change = git_repo.new_file_repository(relative_path=CODE_PLAN_AND_CHANGE_FILE_REPO)


class ResourceFileRepositories(FileRepository):
    """
    该类包含与资源相关的文件仓库，包括不同的资源文件类别。
    """
    competitive_analysis: FileRepository  # 竞争分析文件仓库
    data_api_design: FileRepository  # 数据API设计文件仓库
    seq_flow: FileRepository  # 序列流文件仓库
    system_design: FileRepository  # 系统设计PDF文件仓库
    prd: FileRepository  # 产品需求PDF文件仓库
    api_spec_and_task: FileRepository  # API规范与任务PDF文件仓库
    code_summary: FileRepository  # 代码总结PDF文件仓库
    sd_output: FileRepository  # 系统设计输出文件仓库
    code_plan_and_change: FileRepository  # 代码计划和更改PDF文件仓库
    graph_repo: FileRepository  # 可视化图形文件仓库

    def __init__(self, git_repo):
        """
        初始化方法，创建与资源相关的各个文件仓库。
        """
        super().__init__(git_repo=git_repo, relative_path=RESOURCES_FILE_REPO)

        self.competitive_analysis = git_repo.new_file_repository(relative_path=COMPETITIVE_ANALYSIS_FILE_REPO)
        self.data_api_design = git_repo.new_file_repository(relative_path=DATA_API_DESIGN_FILE_REPO)
        self.seq_flow = git_repo.new_file_repository(relative_path=SEQ_FLOW_FILE_REPO)
        self.system_design = git_repo.new_file_repository(relative_path=SYSTEM_DESIGN_PDF_FILE_REPO)
        self.prd = git_repo.new_file_repository(relative_path=PRD_PDF_FILE_REPO)
        self.api_spec_and_task = git_repo.new_file_repository(relative_path=TASK_PDF_FILE_REPO)
        self.code_summary = git_repo.new_file_repository(relative_path=CODE_SUMMARIES_PDF_FILE_REPO)
        self.sd_output = git_repo.new_file_repository(relative_path=SD_OUTPUT_FILE_REPO)
        self.code_plan_and_change = git_repo.new_file_repository(relative_path=CODE_PLAN_AND_CHANGE_PDF_FILE_REPO)
        self.graph_repo = git_repo.new_file_repository(relative_path=VISUAL_GRAPH_REPO_FILE_REPO)


class ProjectRepo(FileRepository):
    """
    该类表示一个项目仓库，包括文档、资源、测试和代码文件等。
    """
    def __init__(self, root: str | Path | GitRepository):
        """
        初始化方法，接受一个根目录或Git仓库，创建项目仓库对象。
        """
        if isinstance(root, str) or isinstance(root, Path):
            git_repo_ = GitRepository(local_path=Path(root))
        elif isinstance(root, GitRepository):
            git_repo_ = root
        else:
            raise ValueError("Invalid root")
        super().__init__(git_repo=git_repo_, relative_path=Path("."))
        self._git_repo = git_repo_
        self.docs = DocFileRepositories(self._git_repo)  # 文档文件仓库
        self.resources = ResourceFileRepositories(self._git_repo)  # 资源文件仓库
        self.tests = self._git_repo.new_file_repository(relative_path=TEST_CODES_FILE_REPO)  # 测试代码仓库
        self.test_outputs = self._git_repo.new_file_repository(relative_path=TEST_OUTPUTS_FILE_REPO)  # 测试输出仓库
        self._srcs_path = None
        self.code_files_exists()

    def __str__(self):
        """
        返回项目仓库的字符串表示，包括文档、源代码等信息。
        """
        repo_str = f"ProjectRepo({self._git_repo.workdir})"
        docs_str = f"Docs({self.docs.all_files})"
        srcs_str = f"Srcs({self.srcs.all_files})"
        return f"{repo_str}\n{docs_str}\n{srcs_str}"

    @property
    async def requirement(self):
        """
        异步获取项目的需求文件。
        """
        return await self.docs.get(filename=REQUIREMENT_FILENAME)

    @property
    def git_repo(self) -> GitRepository:
        """
        返回Git仓库对象。
        """
        return self._git_repo

    @property
    def workdir(self) -> Path:
        """
        返回项目的工作目录路径。
        """
        return Path(self.git_repo.workdir)

    @property
    def srcs(self) -> FileRepository:
        """
        获取源代码文件仓库，必须先调用with_srcs方法设置源代码路径。
        """
        if not self._srcs_path:
            raise ValueError("Call with_srcs first.")
        return self._git_repo.new_file_repository(self._srcs_path)

    def code_files_exists(self) -> bool:
        """
        检查项目是否包含代码文件。
        """
        src_workdir = get_project_srcs_path(self.git_repo.workdir)
        if not src_workdir.exists():
            return False
        code_files = self.with_src_path(path=src_workdir).srcs.all_files
        if not code_files:
            return False
        return bool(code_files)

    def with_src_path(self, path: str | Path) -> ProjectRepo:
        """
        设置源代码路径并返回当前项目仓库对象。
        """
        path = Path(path)
        if path.is_relative_to(self.workdir):
            self._srcs_path = path.relative_to(self.workdir)
        else:
            self._srcs_path = path
        return self

    @property
    def src_relative_path(self) -> Path | None:
        """
        返回源代码相对路径。
        """
        return self._srcs_path

    @staticmethod
    def search_project_path(filename: str | Path) -> Optional[Path]:
        """
        搜索给定文件或路径所在的项目根目录。
        """
        root = Path(filename).parent if Path(filename).is_file() else Path(filename)
        root = root.resolve()
        while str(root) != "/":
            git_repo = root / ".git"
            if git_repo.exists():
                return root
            root = root.parent
        return None
