import difflib
import json
from pathlib import Path
from typing import Optional

import aiofiles
from bs4 import BeautifulSoup
from unidiff import PatchSet

import metagpt.ext.cr
from metagpt.ext.cr.actions.code_review import CodeReview as CodeReview_
from metagpt.ext.cr.actions.modify_code import ModifyCode
from metagpt.ext.cr.utils.schema import Point
from metagpt.tools.libs.browser import Browser
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.report import EditorReporter


@register_tool(tags=["codereview"], include_functions=["review", "fix"])
class CodeReview:
    """审查和修复拉取请求（PR）中的补丁内容。"""

    async def review(
            self,
            patch_path: str,
            output_file: str,
            point_file: Optional[str] = None,
    ) -> str:
        """审查 PR 并保存代码审查评论。

        注意：
            如果用户未指定输出路径，将使用当前工作目录中的相对路径保存。

        参数：
            patch_path: 补丁文件的本地路径或拉取请求的 URL。
            output_file: 输出文件路径，用于保存代码审查评论。
            point_file: 指定代码审查点的文件路径。如果未指定，不需要传递该参数。

        示例：

            >>> cr = CodeReview()
            >>> await cr.review(patch_path="https://github.com/geekan/MetaGPT/pull/136", output_file="cr/MetaGPT_136.json")
            >>> await cr.review(patch_path="/data/uploads/dev-master.diff", output_file="cr/dev-master.json")
            >>> await cr.review(patch_path="/data/uploads/main.py", output_file="cr/main.json")
        """
        # 获取补丁内容
        patch = await self._get_patch_content(patch_path)
        # 获取代码审查点文件，如果没有指定，则使用默认的 "points.json" 文件
        point_file = point_file if point_file else Path(metagpt.ext.cr.__file__).parent / "points.json"
        await EditorReporter().async_report(str(point_file), "path")

        # 读取审查点内容
        async with aiofiles.open(point_file, "rb") as f:
            cr_point_content = await f.read()
            cr_points = [Point(**i) for i in json.loads(cr_point_content)]

        try:
            # 运行代码审查
            comments = await CodeReview_().run(patch, cr_points, output_file)
        except ValueError as e:
            return str(e)

        # 返回审查结果
        return f"缺陷数量：{len(comments)}，评论已保存至 {output_file}，审查点已保存至 {str(point_file)}"

    async def fix(
            self,
            patch_path: str,
            cr_file: str,
            output_dir: str,
    ) -> str:
        """根据代码审查评论修复补丁内容。

        参数：
            patch_path: 补丁文件的本地路径或拉取请求的 URL。
            cr_file: 保存代码审查评论的文件路径。
            output_dir: 保存修复后的补丁文件的目录路径。
        """
        # 获取补丁内容
        patch = await self._get_patch_content(patch_path)

        # 读取评论文件
        async with aiofiles.open(cr_file, "r", encoding="utf-8") as f:
            comments = json.loads(await f.read())

        # 运行代码修复
        await ModifyCode(pr="").run(patch, comments, output_dir)

        # 返回修复结果
        return f"修复后的补丁文件已保存至 {output_dir}"

    async def _get_patch_content(self, patch_path):
        """获取补丁内容，可以是本地文件或拉取请求的 URL。"""
        if patch_path.startswith(("https://", "http://")):
            # 如果是 URL，则使用浏览器获取补丁内容
            async with Browser() as browser:
                await browser.goto(f"{patch_path}.diff")
                patch_file_content = await browser.page.content()
                if patch_file_content.startswith("<html>"):
                    soup = BeautifulSoup(patch_file_content, "html.parser")
                    pre = soup.find("pre")
                    if pre:
                        patch_file_content = pre.text
        else:
            # 如果是本地文件，则直接读取文件内容
            async with aiofiles.open(patch_path, encoding="utf-8") as f:
                patch_file_content = await f.read()
                await EditorReporter().async_report(patch_path)

            # 如果文件不是 diff 或 patch 格式，则生成一个 diff 文件
            if not patch_path.endswith((".diff", ".patch")):
                name = Path(patch_path).name
                patch_file_content = "".join(
                    difflib.unified_diff([], patch_file_content.splitlines(keepends=True), "/dev/null", f"b/{name}"),
                )
                patch_file_content = f"diff --git a/{name} b/{name}\n{patch_file_content}"

        # 返回补丁内容
        patch: PatchSet = PatchSet(patch_file_content)
        return patch
