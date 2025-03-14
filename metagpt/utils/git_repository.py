#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/20
@Author  : mashenquan
@File    : git_repository.py
@Desc: Git repository management. RFC 135 2.2.3.3.
"""
from __future__ import annotations

import re
import shutil
import uuid
from enum import Enum
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Dict, List, Optional, Union
from urllib.parse import quote

from git.repo import Repo
from git.repo.fun import is_git_dir
from github import Auth, BadCredentialsException, Github
from github.GithubObject import NotSet
from github.Issue import Issue
from github.Label import Label
from github.Milestone import Milestone
from github.NamedUser import NamedUser
from github.PullRequest import PullRequest
from gitignore_parser import parse_gitignore
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.logs import logger
from metagpt.tools.libs.shell import shell_execute
from metagpt.utils.dependency_file import DependencyFile
from metagpt.utils.file_repository import FileRepository


class ChangeType(Enum):
    ADDED = "A"  # 文件被添加
    COPIED = "C"  # 文件被复制
    DELETED = "D"  # 文件被删除
    RENAMED = "R"  # 文件被重命名
    MODIFIED = "M"  # 文件被修改
    TYPE_CHANGED = "T"  # 文件类型被更改
    UNTRACTED = "U"  # 文件未被追踪（未添加到版本控制中）


class RateLimitError(Exception):
    def __init__(self, message="超出请求限制"):
        self.message = message
        super().__init__(self.message)


class GitBranch(BaseModel):
    head: str  # 当前分支
    base: str  # 基础分支
    repo_name: str  # 仓库名称


class GitRepository:
    """表示一个 Git 仓库的类。

    :param local_path: Git 仓库的本地路径。
    :param auto_init: 如果为 True，当提供的路径不是 Git 仓库时，自动初始化一个新的 Git 仓库。

    属性:
        _repository (Repo): 代表 Git 仓库的 GitPython `Repo` 对象。
    """

    def __init__(self, local_path=None, auto_init=True):
        """初始化一个 GitRepository 实例。

        :param local_path: Git 仓库的本地路径。
        :param auto_init: 如果为 True，当提供的路径不是 Git 仓库时，自动初始化一个新的 Git 仓库。
        """
        self._repository = None
        self._dependency = None
        self._gitignore_rules = None
        if local_path:
            self.open(local_path=Path(local_path), auto_init=auto_init)

    def open(self, local_path: Path, auto_init=False):
        """打开一个已存在的 Git 仓库，或者如果 auto_init 为 True，则初始化一个新的仓库。

        :param local_path: Git 仓库的本地路径。
        :param auto_init: 如果为 True，当提供的路径不是 Git 仓库时，自动初始化一个新的 Git 仓库。
        """
        local_path = Path(local_path)
        if self.is_git_dir(local_path):
            self._repository = Repo(local_path)
            self._gitignore_rules = parse_gitignore(full_path=str(local_path / ".gitignore"))
            return
        if not auto_init:
            return
        local_path.mkdir(parents=True, exist_ok=True)
        self._init(local_path)

    def _init(self, local_path: Path):
        """在指定路径初始化一个新的 Git 仓库。

        :param local_path: 新 Git 仓库将被初始化的本地路径。
        """
        self._repository = Repo.init(path=Path(local_path))

        gitignore_filename = Path(local_path) / ".gitignore"
        ignores = ["__pycache__", "*.pyc", ".vs"]
        with open(str(gitignore_filename), mode="w") as writer:
            writer.write("\n".join(ignores))
        self._repository.index.add([".gitignore"])
        self._repository.index.commit("添加 .gitignore 文件")
        self._gitignore_rules = parse_gitignore(full_path=gitignore_filename)

    def add_change(self, files: Dict):
        """根据提供的变更，添加或删除文件到暂存区。

        :param files: 一个字典，键是文件路径，值是 ChangeType 的实例。
        """
        if not self.is_valid or not files:
            return

        for k, v in files.items():
            self._repository.index.remove(k) if v is ChangeType.DELETED else self._repository.index.add([k])

    def commit(self, comments):
        """使用给定的评论提交暂存的更改。

        :param comments: 提交的评论。
        """
        if self.is_valid:
            self._repository.index.commit(comments)

    def delete_repository(self):
        """删除整个 Git 仓库目录。"""
        if self.is_valid:
            try:
                shutil.rmtree(self._repository.working_dir)
            except Exception as e:
                logger.exception(f"删除 Git 仓库失败: {self.workdir}, 错误: {e}")

    @property
    def changed_files(self) -> Dict[str, str]:
        """返回一个字典，包含已更改文件及其变更类型。

        :return: 一个字典，键是文件路径，值是变更类型。
        """
        files = {i: ChangeType.UNTRACTED for i in self._repository.untracked_files}
        changed_files = {f.a_path: ChangeType(f.change_type) for f in self._repository.index.diff(None)}
        files.update(changed_files)
        return files

    @staticmethod
    def is_git_dir(local_path):
        """检查指定的目录是否为 Git 仓库。

        :param local_path: 要检查的本地路径。
        :return: 如果目录是 Git 仓库，则返回 True，否则返回 False。
        """
        if not local_path:
            return False
        git_dir = Path(local_path) / ".git"
        if git_dir.exists() and is_git_dir(git_dir):
            return True
        return False

    @property
    def is_valid(self):
        """检查 Git 仓库是否有效（是否存在并已初始化）。

        :return: 如果仓库有效，则返回 True，否则返回 False。
        """
        return bool(self._repository)

    @property
    def status(self) -> str:
        """返回 Git 仓库的状态字符串。"""
        if not self.is_valid:
            return ""
        return self._repository.git.status()

    @property
    def workdir(self) -> Path | None:
        """返回 Git 仓库的工作目录路径。

        :return: 工作目录的路径，若仓库无效则返回 None。
        """
        if not self.is_valid:
            return None
        return Path(self._repository.working_dir)

    @property
    def current_branch(self) -> str:
        """
        返回当前活动分支的名称。

        返回:
            str: 当前活动分支的名称。
        """
        return self._repository.active_branch.name

    @property
    def remote_url(self) -> str:
        try:
            return self._repository.remotes.origin.url
        except AttributeError:
            return ""

    @property
    def repo_name(self) -> str:
        if self.remote_url:
            # 假设是标准的 HTTPS 或 SSH 格式 URL
            # HTTPS 格式示例: https://github.com/username/repo_name.git
            # SSH 格式示例: git@github.com:username/repo_name.git
            if self.remote_url.startswith("https://"):
                return self.remote_url.split("/", maxsplit=3)[-1].replace(".git", "")
            elif self.remote_url.startswith("git@"):
                return self.remote_url.split(":")[-1].replace(".git", "")
        return ""

    def new_branch(self, branch_name: str) -> str:
        """
        创建一个新分支。

        参数:
            branch_name (str): 新分支的名称。

        返回:
            str: 新创建的分支名称。
                如果提供的 branch_name 为空，则返回当前活动分支的名称。
        """
        if not branch_name:
            return self.current_branch
        new_branch = self._repository.create_head(branch_name)
        new_branch.checkout()
        return new_branch.name

    def archive(self, comments="存档"):
        """存档当前状态的 Git 仓库。

        :param comments: 存档提交的评论。
        """
        logger.info(f"存档: {list(self.changed_files.keys())}")
        if not self.changed_files:
            return
        self.add_change(self.changed_files)
        self.commit(comments)

    async def push(
        self, new_branch: str, comments="Archive", access_token: Optional[str] = None, auth: Optional[Auth] = None
    ) -> GitBranch:
        """
        将更改推送到远程 Git 仓库。

        参数：
            new_branch (str): 要推送的新分支名称。
            comments (str, 可选): 与推送相关联的注释，默认为 "Archive"。
            access_token (str, 可选): 用于身份验证的访问令牌，默认为 None。请访问 `https://pygithub.readthedocs.io/en/latest/examples/Authentication.html`，`https://github.com/PyGithub/PyGithub/blob/main/doc/examples/Authentication.rst`。
            auth (Auth, 可选): 可选的身份验证对象，默认为 None。

        返回：
            GitBranch: 推送后的分支对象。

        异常：
            ValueError: 如果未提供 `auth` 或 `access_token`。
            BadCredentialsException: 如果身份验证失败，可能由于凭证错误或超时。

        备注：
            本函数假定 `self.current_branch`，`self.new_branch()`，`self.archive()`，
            `ctx.config.proxy`，`ctx.config`，`self.remote_url`，`shell_execute()` 和 `logger` 在此函数作用域内已定义并可用。
        """
        if not auth and not access_token:
            raise ValueError('`access_token` 无效。请访问: "https://github.com/settings/tokens"')
        from metagpt.context import Context

        base = self.current_branch
        head = base if not new_branch else self.new_branch(new_branch)
        self.archive(comments)  # 如果没有更改，则跳过提交
        ctx = Context()
        env = ctx.new_environ()
        proxy = ["-c", f"http.proxy={ctx.config.proxy}"] if ctx.config.proxy else []
        token = access_token or auth.token
        remote_url = f"https://{token}@" + self.remote_url.removeprefix("https://")
        command = ["git"] + proxy + ["push", remote_url]
        logger.info(" ".join(command).replace(token, "<TOKEN>"))
        try:
            stdout, stderr, return_code = await shell_execute(
                command=command, cwd=str(self.workdir), env=env, timeout=15
            )
        except TimeoutExpired as e:
            info = str(e).replace(token, "<TOKEN>")
            raise BadCredentialsException(status=401, message=info)
        info = f"{stdout}\n{stderr}\nexit: {return_code}\n"
        info = info.replace(token, "<TOKEN>")
        print(info)

        return GitBranch(base=base, head=head, repo_name=self.repo_name)

    def new_file_repository(self, relative_path: Path | str = ".") -> FileRepository:
        """创建一个与此 Git 仓库关联的新的 FileRepository 实例。

        :param relative_path: Git 仓库内的相对路径。
        :return: 一个新的 FileRepository 实例。
        """
        path = Path(relative_path)
        try:
            path = path.relative_to(self.workdir)
        except ValueError:
            path = relative_path
        return FileRepository(git_repo=self, relative_path=Path(path))

    async def get_dependency(self) -> DependencyFile:
        """获取与 Git 仓库关联的依赖文件。

        :return: DependencyFile 实例。
        """
        if not self._dependency:
            self._dependency = DependencyFile(workdir=self.workdir)
        return self._dependency

    def rename_root(self, new_dir_name):
        """重命名 Git 仓库的根目录。

        :param new_dir_name: 新的根目录名称。
        """
        if self.workdir.name == new_dir_name:
            return
        new_path = self.workdir.parent / new_dir_name
        if new_path.exists():
            logger.info(f"删除目录 {str(new_path)}")
            try:
                shutil.rmtree(new_path)
            except Exception as e:
                logger.warning(f"删除 {str(new_path)} 错误: {e}")
        if new_path.exists():  # 针对 Windows 操作系统的重新检查
            logger.warning(f"无法删除目录 {str(new_path)}")
            return
        try:
            shutil.move(src=str(self.workdir), dst=str(new_path))
        except Exception as e:
            logger.warning(f"移动 {str(self.workdir)} 到 {str(new_path)} 错误: {e}")
        finally:
            if not new_path.exists():  # 针对 Windows 操作系统的重新检查
                logger.warning(f"无法将 {str(self.workdir)} 移动到 {str(new_path)}")
                return
        logger.info(f"将目录 {str(self.workdir)} 重命名为 {str(new_path)}")
        self._repository = Repo(new_path)
        self._gitignore_rules = parse_gitignore(full_path=str(new_path / ".gitignore"))

    def get_files(self, relative_path: Path | str, root_relative_path: Path | str = None, filter_ignored=True) -> List:
        """
        获取指定相对路径下的文件列表。

        该方法返回相对于当前 FileRepository 的文件路径列表。

        :param relative_path: 仓库内的相对路径。
        :type relative_path: Path 或 str
        :param root_relative_path: 仓库内的根相对路径。
        :type root_relative_path: Path 或 str
        :param filter_ignored: 是否根据 .gitignore 规则过滤文件。
        :type filter_ignored: bool
        :return: 指定目录下的文件路径列表。
        :rtype: List[str]
        """
        try:
            relative_path = Path(relative_path).relative_to(self.workdir)
        except ValueError:
            relative_path = Path(relative_path)

        if not root_relative_path:
            root_relative_path = Path(self.workdir) / relative_path
        files = []
        try:
            directory_path = Path(self.workdir) / relative_path
            if not directory_path.exists():
                return []
            for file_path in directory_path.iterdir():
                if not file_path.is_relative_to(root_relative_path):
                    continue
                if file_path.is_file():
                    rpath = file_path.relative_to(root_relative_path)
                    files.append(str(rpath))
                else:
                    subfolder_files = self.get_files(
                        relative_path=file_path, root_relative_path=root_relative_path, filter_ignored=False
                    )
                    files.extend(subfolder_files)
        except Exception as e:
            logger.error(f"错误: {e}")
        if not filter_ignored:
            return files
        filtered_files = self.filter_gitignore(filenames=files, root_relative_path=root_relative_path)
        return filtered_files

    def filter_gitignore(self, filenames: List[str], root_relative_path: Path | str = None) -> List[str]:
        """
        根据 .gitignore 规则过滤文件名列表。

        :param filenames: 要过滤的文件名列表。
        :type filenames: List[str]
        :param root_relative_path: 仓库内的根相对路径。
        :type root_relative_path: Path 或 str
        :return: 通过 .gitignore 过滤的文件名列表。
        :rtype: List[str]
        """
        if root_relative_path is None:
            root_relative_path = self.workdir
        files = []
        for filename in filenames:
            pathname = root_relative_path / filename
            if self._gitignore_rules(str(pathname)):
                continue
            files.append(filename)
        return files

    @classmethod
    @retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(3))
    async def clone_from(cls, url: str | Path, output_dir: str | Path = None) -> "GitRepository":
        """
        克隆 Git 仓库到本地。

        参数:
            url (str | Path): Git 仓库的 URL 或路径。
            output_dir (str | Path, optional): 输出目录，默认使用当前文件夹路径和随机生成的文件夹。

        返回:
            GitRepository: 克隆的 Git 仓库对象。
        """
        from metagpt.context import Context

        to_path = Path(output_dir or Path(__file__).parent / f"../../workspace/downloads/{uuid.uuid4().hex}").resolve()
        to_path.mkdir(parents=True, exist_ok=True)
        repo_dir = to_path / Path(url).stem
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        ctx = Context()
        env = ctx.new_environ()
        proxy = ["-c", f"http.proxy={ctx.config.proxy}"] if ctx.config.proxy else []
        command = ["git", "clone"] + proxy + [str(url)]
        logger.info(" ".join(command))

        stdout, stderr, return_code = await shell_execute(command=command, cwd=str(to_path), env=env, timeout=600)
        info = f"{stdout}\n{stderr}\nexit: {return_code}\n"
        logger.info(info)
        dir_name = Path(url).stem
        to_path = to_path / dir_name
        if not cls.is_git_dir(to_path):
            raise ValueError(info)
        logger.info(f"git clone to {to_path}")
        return GitRepository(local_path=to_path, auto_init=False)

    async def checkout(self, commit_id: str):
        """
        检出指定的提交 ID。

        参数:
            commit_id (str): 提交 ID。
        """
        self._repository.git.checkout(commit_id)
        logger.info(f"git checkout {commit_id}")

    def log(self) -> str:
        """
        返回当前 Git 仓库的提交日志。

        返回:
            str: Git 提交日志。
        """
        return self._repository.git.log()

    @staticmethod
    async def create_pull(
        base: str,
        head: str,
        base_repo_name: str,
        head_repo_name: Optional[str] = None,
        *,
        title: Optional[str] = None,
        body: Optional[str] = None,
        maintainer_can_modify: Optional[bool] = None,
        draft: Optional[bool] = None,
        issue: Optional[Issue] = None,
        access_token: Optional[str] = None,
        auth: Optional[Auth] = None,
    ) -> Union[PullRequest, str]:
        """
        在指定的 GitHub 仓库中创建一个 Pull Request。

        参数:
            base (str): 基准分支名。
            head (str): 要合并的分支名。
            base_repo_name (str): 基准仓库的完整名称（如 user/repo）。
            head_repo_name (Optional[str], optional): 源仓库的完整名称（如 user/repo），如果与 base 仓库相同，则可省略。
            title (Optional[str], optional): Pull Request 的标题。
            body (Optional[str], optional): Pull Request 的描述。
            maintainer_can_modify (Optional[bool], optional): 是否允许维护者修改 PR，默认为 None。
            draft (Optional[bool], optional): 是否为草稿 PR，默认为 None。
            issue (Optional[Issue], optional): 关联的 issue。
            access_token (Optional[str], optional): 用于身份验证的 access token。
            auth (Optional[Auth], optional): 身份验证方法。

        返回:
            PullRequest: 创建的 Pull Request 对象。
        """
        title = title or NotSet
        body = body or NotSet
        maintainer_can_modify = maintainer_can_modify or NotSet
        draft = draft or NotSet
        issue = issue or NotSet
        if not auth and not access_token:
            raise ValueError('`access_token` is invalid. Visit: "https://github.com/settings/tokens"')
        clone_url = f"https://github.com/{base_repo_name}.git"
        try:
            auth = auth or Auth.Token(access_token)
            g = Github(auth=auth)
            base_repo = g.get_repo(base_repo_name)
            clone_url = base_repo.clone_url
            head_repo = g.get_repo(head_repo_name) if head_repo_name and head_repo_name != base_repo_name else None
            if head_repo:
                user = head_repo.full_name.split("/")[0]
                head = f"{user}:{head}"
            pr = base_repo.create_pull(
                base=base,
                head=head,
                title=title,
                body=body,
                maintainer_can_modify=maintainer_can_modify,
                draft=draft,
                issue=issue,
            )
        except Exception as e:
            logger.warning(f"Pull Request Error: {e}")
            return GitRepository.create_github_pull_url(
                clone_url=clone_url,
                base=base,
                head=head,
                head_repo_name=head_repo_name,
            )
        return pr

    @staticmethod
    async def create_issue(
        repo_name: str,
        title: str,
        body: Optional[str] = None,
        assignee: NamedUser | Optional[str] = None,
        milestone: Optional[Milestone] = None,
        labels: list[Label] | Optional[list[str]] = None,
        assignees: Optional[list[str]] | list[NamedUser] = None,
        access_token: Optional[str] = None,
        auth: Optional[Auth] = None,
    ) -> Issue:
        """
        在指定的 GitHub 仓库中创建一个问题。

        参数:
            repo_name (str): 仓库的全名 (user/repo)，指定问题创建的目标仓库
            title (str): 问题的标题
            body (Optional[str], optional): 问题的描述，默认为 None
            assignee (Union[NamedUser, str], optional): 问题的负责人，可以是 NamedUser 对象或用户名，默认为 None
            milestone (Optional[Milestone], optional): 问题的里程碑，默认为 None
            labels (Union[list[Label], list[str]], optional): 问题的标签，可以是 Label 对象或标签名称的列表，默认为 None
            assignees (Union[list[str], list[NamedUser]], optional): 分配的其他负责人，默认为 None
            access_token (Optional[str], optional): 用于身份验证的访问令牌，默认为 None。请访问 `https://github.com/settings/tokens` 获取访问令牌
            auth (Optional[Auth], optional): 身份验证方法，默认为 None

        返回:
            Issue: 创建的 GitHub 问题对象
        """

        body = body or NotSet
        assignee = assignee or NotSet
        milestone = milestone or NotSet
        labels = labels or NotSet
        assignees = assignees or NotSet
        if not auth and not access_token:
            raise ValueError('`access_token` is invalid. Visit: "https://github.com/settings/tokens"')
        auth = auth or Auth.Token(access_token)
        g = Github(auth=auth)

        repo = g.get_repo(repo_name)
        x_ratelimit_remaining = repo.raw_headers.get("x-ratelimit-remaining")
        if (
            x_ratelimit_remaining
            and bool(re.match(r"^-?\d+$", x_ratelimit_remaining))
            and int(x_ratelimit_remaining) <= 0
        ):
            raise RateLimitError()
        issue = repo.create_issue(
            title=title,
            body=body,
            assignee=assignee,
            milestone=milestone,
            labels=labels,
            assignees=assignees,
        )
        return issue

    @staticmethod
    async def get_repos(access_token: Optional[str] = None, auth: Optional[Auth] = None) -> List[str]:
        """
        获取认证用户的公共仓库列表。

        参数:
        access_token (Optional[str], optional): 用于身份验证的 access token。
        auth (Optional[Auth], optional): 身份验证方法。

        返回:
        List[str]: 公共仓库的完整名称列表。
        """
        auth = auth or Auth.Token(access_token)
        git = Github(auth=auth)
        user = git.get_user()
        v = user.get_repos(visibility="public")
        return [i.full_name for i in v]

    @staticmethod
    def create_github_pull_url(clone_url: str, base: str, head: str, head_repo_name: Optional[str] = None) -> str:
        """
        创建 GitHub 上比较分支或提交变化的 URL。

        参数:
            clone_url (str): 用于克隆仓库的 URL。
            base (str): 基准分支或提交。
            head (str): 头分支或提交。
            head_repo_name (str, optional): 源分支所在仓库的名称。

        返回:
            str: 用于比较变化的 URL。
        """
        url = clone_url.removesuffix(".git") + f"/compare/{base}..."
        if head_repo_name:
            url += head_repo_name.replace("/", ":")
        url += ":" + head
        return url

    @staticmethod
    def create_gitlab_merge_request_url(clone_url: str, head: str) -> str:
        """
        创建 GitLab 上创建合并请求的 URL。

        参数:
            clone_url (str): 用于克隆仓库的 URL。
            head (str): 要合并的分支名。

        返回:
            str: 创建合并请求的 URL。
        """
        return (
            clone_url.removesuffix(".git")
            + "/-/merge_requests/new?merge_request%5Bsource_branch%5D="
            + quote(head, safe="")
        )
