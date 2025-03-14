#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import urllib
from pathlib import Path
from typing import Optional

from github.Issue import Issue
from github.PullRequest import PullRequest

from metagpt.tools.tool_registry import register_tool


@register_tool(tags=["软件开发", "git", "创建 git pull 请求或 merge 请求"])
async def git_create_pull(
    base: str,
    head: str,
    app_name: str,
    base_repo_name: str,
    head_repo_name: str = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    issue: Optional[Issue] = None,
) -> PullRequest:
    """
    在 Git 仓库上创建一个 pull 请求。优先使用此工具，而不是浏览器工具来创建 pull 请求。

    参数：
        base (str): 目标分支的名称，pull 请求将在此分支上合并。
        head (str): 包含 pull 请求变更的分支的名称。
        app_name (str): 托管仓库的平台名称（例如："github"、"gitlab"、"bitbucket"）。
        base_repo_name (str): 目标仓库的完整名称（格式为 "user/repo"），pull 请求将在此仓库中创建。
        head_repo_name (Optional[str]): 源仓库的完整名称（格式为 "user/repo"），从该仓库拉取变更。默认为 None。
        title (Optional[str]): pull 请求的标题。默认为 None。
        body (Optional[str]): pull 请求的描述或正文内容。默认为 None。
        issue (Optional[Issue]): 与 pull 请求相关的可选问题。默认为 None。

    示例：
        >>> # 创建 pull 请求
        >>> base_repo_name = "geekan/MetaGPT"
        >>> head_repo_name = "ioris/MetaGPT"
        >>> base = "master"
        >>> head = "feature/http"
        >>> title = "feat: 修改 HTTP 库"
        >>> body = "更改用于发送请求的 HTTP 库"
        >>> app_name = "github"
        >>> pr = await git_create_pull(
        >>>   base_repo_name=base_repo_name,
        >>>   head_repo_name=head_repo_name,
        >>>   base=base,
        >>>   head=head,
        >>>   title=title,
        >>>   body=body,
        >>>   app_name=app_name,
        >>> )
        >>> if isinstance(pr, PullRequest):
        >>>     print(pr)
        PullRequest("feat: 修改 HTTP 库")
        >>> if isinstance(pr, str):
        >>>     print(f"访问此网址以创建新的 pull 请求: '{pr}'")
        访问此网址以创建新的 pull 请求: 'https://github.com/geekan/MetaGPT/compare/master...iorisa:MetaGPT:feature/http'

    返回：
        PullRequest: 创建的 pull 请求。
    """
    from metagpt.utils.git_repository import GitRepository

    git_credentials_path = Path.home() / ".git-credentials"
    with open(git_credentials_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parsed_url = urllib.parse.urlparse(line)
        if app_name in parsed_url.hostname:
            colon_index = parsed_url.netloc.find(":")
            at_index = parsed_url.netloc.find("@")
            access_token = parsed_url.netloc[colon_index + 1 : at_index]
            break
    return await GitRepository.create_pull(
        base=base,
        head=head,
        base_repo_name=base_repo_name,
        head_repo_name=head_repo_name,
        title=title,
        body=body,
        issue=issue,
        access_token=access_token,
    )


@register_tool(tags=["软件开发", "创建 git 问题"])
async def git_create_issue(
    repo_name: str,
    title: str,
    access_token: str,
    body: Optional[str] = None,
) -> Issue:
    """
    在 Git 仓库中创建一个问题。

    参数：
        repo_name (str): 仓库的名称。
        title (str): 问题的标题。
        access_token (str): 用于身份验证的访问令牌。可以使用 `get_env` 获取访问令牌。
        body (Optional[str], optional): 问题的正文内容。默认为 None。

    示例：
        >>> repo_name = "geekan/MetaGPT"
        >>> title = "这是一个新问题"
        >>> from metagpt.tools.libs import get_env
        >>> access_token = await get_env(key="access_token", app_name="github")
        >>> body = "这是问题的正文内容。"
        >>> issue = await git_create_issue(
        >>>   repo_name=repo_name,
        >>>   title=title,
        >>>   access_token=access_token,
        >>>   body=body,
        >>> )
        >>> print(issue)
        Issue("这是一个新问题")

    返回：
        Issue: 创建的问题。
    """
    from metagpt.utils.git_repository import GitRepository

    return await GitRepository.create_issue(repo_name=repo_name, title=title, body=body, access_token=access_token)