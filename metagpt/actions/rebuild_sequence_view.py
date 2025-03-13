#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4
@Author  : mashenquan
@File    : rebuild_sequence_view.py
@Desc    : Reconstruct sequence view information through reverse engineering.
    Implement RFC197, https://deepwisdom.feishu.cn/wiki/VyK0wfq56ivuvjklMKJcmHQknGt
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions import Action
from metagpt.const import GRAPH_REPO_FILE_REPO
from metagpt.logs import logger
from metagpt.repo_parser import CodeBlockInfo, DotClassInfo
from metagpt.schema import UMLClassView
from metagpt.utils.common import (
    add_affix,
    aread,
    auto_namespace,
    concat_namespace,
    general_after_log,
    list_files,
    parse_json_code_block,
    read_file_block,
    split_namespace,
)
from metagpt.utils.di_graph_repository import DiGraphRepository
from metagpt.utils.graph_repository import SPO, GraphKeyword, GraphRepository


class ReverseUseCase(BaseModel):
    """
    Represents a reverse engineered use case.

    Attributes:
        description (str): A description of the reverse use case.
        inputs (List[str]): List of inputs for the reverse use case.
        outputs (List[str]): List of outputs for the reverse use case.
        actors (List[str]): List of actors involved in the reverse use case.
        steps (List[str]): List of steps for the reverse use case.
        reason (str): The reason behind the reverse use case.
    """

    description: str
    inputs: List[str]
    outputs: List[str]
    actors: List[str]
    steps: List[str]
    reason: str


class ReverseUseCase(BaseModel):
    """
    代表逆向工程的用例。

    属性:
        description (str): 逆向用例的描述。
        inputs (List[str]): 逆向用例的输入列表。
        outputs (List[str]): 逆向用例的输出列表。
        actors (List[str]): 参与该逆向用例的角色列表。
        steps (List[str]): 逆向用例的步骤列表。
        reason (str): 逆向用例的原因说明。
    """

    description: str
    inputs: List[str]
    outputs: List[str]
    actors: List[str]
    steps: List[str]
    reason: str


class ReverseUseCaseDetails(BaseModel):
    """
    代表逆向工程用例的详细信息。

    属性:
        description (str): 逆向用例详细信息的描述。
        use_cases (List[ReverseUseCase]): 逆向用例的列表。
        relationship (List[str]): 与逆向用例相关的关系列表。
    """

    description: str
    use_cases: List[ReverseUseCase]
    relationship: List[str]


class RebuildSequenceView(Action):
    """
    代表通过逆向工程重建序列视图的操作。

    属性:
        graph_db (Optional[GraphRepository]): 可选的图数据库操作实例。
    """

    graph_db: Optional[GraphRepository] = None

    async def run(self, with_messages=None, format=None):
        """
        实现 `Action` 的 `run` 方法。

        参数:
            with_messages (Optional[Type]): 可选参数，指定要响应的消息。
            format (str): 提供的提示格式。
        """
        format = format if format else self.config.prompt_schema
        graph_repo_pathname = self.context.git_repo.workdir / GRAPH_REPO_FILE_REPO / self.context.git_repo.workdir.name
        self.graph_db = await DiGraphRepository.load_from(str(graph_repo_pathname.with_suffix(".json")))

        # 如果没有上下文，则搜索主入口
        if not self.i_context:
            entries = await self._search_main_entry()
        else:
            entries = [SPO(subject=self.i_context, predicate="", object_="")]

        for entry in entries:
            # 重建主序列视图
            await self._rebuild_main_sequence_view(entry)
            while await self._merge_sequence_view(entry):
                pass
        # 保存图数据库
        await self.graph_db.save()

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _rebuild_main_sequence_view(self, entry: SPO):
        """
        通过逆向工程重建源代码的序列图。

        参数:
            entry (SPO): 图数据库中与 `__name__:__main__` 相关的 SPO 对象。
        """
        filename = entry.subject.split(":", 1)[0]
        rows = await self.graph_db.select(predicate=GraphKeyword.IS, object_=GraphKeyword.CLASS)
        classes = []
        prefix = filename + ":"
        for r in rows:
            if prefix in r.subject:
                classes.append(r)
                await self._rebuild_use_case(r.subject)
        participants = await self._search_participants(split_namespace(entry.subject)[0])
        class_details = []
        class_views = []
        for c in classes:
            detail = await self._get_class_detail(c.subject)
            if not detail:
                continue
            class_details.append(detail)
            view = await self._get_uml_class_view(c.subject)
            if view:
                class_views.append(view)

            # 收集参与者信息
            actors = await self._get_participants(c.subject)
            participants.update(set(actors))

        use_case_blocks = []
        for c in classes:
            use_cases = await self._get_class_use_cases(c.subject)
            use_case_blocks.append(use_cases)

        prompt_blocks = ["## Use Cases\n" + "\n".join(use_case_blocks)]
        block = "## Participants\n"
        for p in participants:
            block += f"- {p}\n"
        prompt_blocks.append(block)

        block = "## Mermaid Class Views\n```mermaid\n"
        block += "\n\n".join([c.get_mermaid() for c in class_views])
        block += "\n```\n"
        prompt_blocks.append(block)

        block = "## Source Code\n```python\n"
        block += await self._get_source_code(filename)
        block += "\n```\n"
        prompt_blocks.append(block)

        prompt = "\n---\n".join(prompt_blocks)

        # 向 LLM 提交请求，生成 Mermaid 序列图
        rsp = await self.llm.aask(
            msg=prompt,
            system_msgs=[
                "你是一个 Python 代码到 Mermaid 序列图的转换器。",
                "将给定的 Markdown 文本转换为 Mermaid 序列图。",
                "返回合并后的 Mermaid 序列图，并以 Markdown 代码块格式返回。",
            ],
            stream=False,
        )
        # 处理 Mermaid 序列图
        sequence_view = rsp.removeprefix("```mermaid").removesuffix("```")
        rows = await self.graph_db.select(subject=entry.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW)
        for r in rows:
            if r.predicate == GraphKeyword.HAS_SEQUENCE_VIEW:
                await self.graph_db.delete(subject=r.subject, predicate=r.predicate, object_=r.object_)
        await self.graph_db.insert(
            subject=entry.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW, object_=sequence_view
        )
        await self.graph_db.insert(
            subject=entry.subject,
            predicate=GraphKeyword.HAS_SEQUENCE_VIEW_VER,
            object_=concat_namespace(datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3], add_affix(sequence_view)),
        )
        for c in classes:
            await self.graph_db.insert(
                subject=entry.subject, predicate=GraphKeyword.HAS_PARTICIPANT, object_=auto_namespace(c.subject)
            )
        await self._save_sequence_view(subject=entry.subject, content=sequence_view)

    async def _merge_sequence_view(self, entry: SPO) -> bool:
        """
        为提供的 SPO 入口增强额外信息。

        参数:
            entry (SPO): 图数据库中的 SPO 对象。

        返回:
            bool: 如果已增强信息则返回 True，否则返回 False。
        """
        new_participant = await self._search_new_participant(entry)
        if not new_participant:
            return False

        # 合并参与者信息
        await self._merge_participant(entry, new_participant)
        return True

    async def _search_main_entry(self) -> List:
        """
        异步搜索与 `__name__:__main__` 相关的 SPO 对象。

        返回:
            List: 包含主入口序列图相关信息的列表。
        """
        rows = await self.graph_db.select(predicate=GraphKeyword.HAS_PAGE_INFO)
        tag = "__name__:__main__"
        entries = []
        for r in rows:
            if tag in r.subject or tag in r.object_:
                entries.append(r)
        return entries

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _rebuild_use_case(self, ns_class_name: str):
        """
        异步重建与指定命名空间类名相关的用例。

        参数:
            ns_class_name (str): 命名空间类名，用于重建用例。
        """
        rows = await self.graph_db.select(subject=ns_class_name, predicate=GraphKeyword.HAS_CLASS_USE_CASE)
        if rows:
            return

        detail = await self._get_class_detail(ns_class_name)
        if not detail:
            return
        participants = set()
        participants.update(set(detail.compositions))
        participants.update(set(detail.aggregations))
        class_view = await self._get_uml_class_view(ns_class_name)
        source_code = await self._get_source_code(ns_class_name)

        prompt_blocks = []
        block = "## Participants\n"
        for p in participants:
            block += f"- {p}\n"
        prompt_blocks.append(block)
        block = "## Mermaid Class Views\n```mermaid\n"
        block += class_view.get_mermaid()
        block += "\n```\n"
        prompt_blocks.append(block)
        block = "## Source Code\n```python\n"
        block += source_code
        block += "\n```\n"
        prompt_blocks.append(block)
        prompt = "\n---\n".join(prompt_blocks)

        rsp = await self.llm.aask(
            msg=prompt,
            system_msgs=[
                "你是一个 Python 代码到 UML 2.0 用例图的转换器。",
                '生成的 UML 2.0 用例图必须包含在 "Participants" 中列出的角色或实体。',
                "生成的 UML 2.0 用例图中的演员和用例的功能描述，不能与 'Mermaid 类视图' 中的信息冲突。",
                '在 `if __name__ == "__main__":` 下的 "源代码" 部分包含有关外部系统与内部系统交互的信息。',
                "返回一个 markdown 格式的 JSON 对象，包含：\n"
                '- "description" 键：解释源代码的整体目的；\n'
                '- "use_cases" 键：列出所有用例，每个用例包含 `description`、`inputs`、`outputs`、`actors`、`steps` 和 `reason`。\n'
                '- "relationship" 键：列出用例之间的关系描述。\n',
            ],
            stream=False,
        )

        code_blocks = parse_json_code_block(rsp)
        for block in code_blocks:
            detail = ReverseUseCaseDetails.model_validate_json(block)
            await self.graph_db.insert(
                subject=ns_class_name, predicate=GraphKeyword.HAS_CLASS_USE_CASE, object_=detail.model_dump_json()
            )

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        after=general_after_log(logger),
    )
    async def _rebuild_sequence_view(self, ns_class_name: str):
        """
        异步重建指定命名空间前缀类名的序列图。

        参数:
            ns_class_name (str): 需要重建序列图的命名空间前缀类名。
        """
        await self._rebuild_use_case(ns_class_name)

        prompts_blocks = []
        use_case_markdown = await self._get_class_use_cases(ns_class_name)
        if not use_case_markdown:  # 外部类
            await self.graph_db.insert(subject=ns_class_name, predicate=GraphKeyword.HAS_SEQUENCE_VIEW, object_="")
            return
        block = f"## 用例\n{use_case_markdown}"
        prompts_blocks.append(block)

        participants = await self._get_participants(ns_class_name)
        block = "## 参与者\n" + "\n".join([f"- {s}" for s in participants])
        prompts_blocks.append(block)

        view = await self._get_uml_class_view(ns_class_name)
        block = "## Mermaid 类视图\n```mermaid\n"
        block += view.get_mermaid()
        block += "\n```\n"
        prompts_blocks.append(block)

        block = "## 源代码\n```python\n"
        block += await self._get_source_code(ns_class_name)
        block += "\n```\n"
        prompts_blocks.append(block)
        prompt = "\n---\n".join(prompts_blocks)

        rsp = await self.llm.aask(
            prompt,
            system_msgs=[
                "你是一个 Mermaid 序列图翻译器，专注于功能细节。",
                "将 markdown 文本翻译成 Mermaid 序列图。",
                "回答必须简洁。",
                "返回一个 markdown mermaid 代码块。",
            ],
            stream=False,
        )

        sequence_view = rsp.removeprefix("```mermaid").removesuffix("```")
        await self.graph_db.insert(
            subject=ns_class_name, predicate=GraphKeyword.HAS_SEQUENCE_VIEW, object_=sequence_view
        )

    async def _get_participants(self, ns_class_name: str) -> List[str]:
        """
        异步获取指定命名空间前缀的序列图参与者列表。

        参数:
            ns_class_name (str): 需要获取参与者的命名空间前缀类名。

        返回:
            List[str]: 包含序列图参与者的字符串列表。
        """
        participants = set()
        detail = await self._get_class_detail(ns_class_name)
        if not detail:
            return []
        participants.update(set(detail.compositions))
        participants.update(set(detail.aggregations))
        return list(participants)

    async def _get_class_use_cases(self, ns_class_name: str) -> str:
        """
        异步获取命名空间前缀类的用例信息。

        参数:
            ns_class_name (str): 需要获取用例信息的命名空间前缀类名。

        返回:
            str: 包含该类的用例信息的字符串。
        """
        block = ""
        rows = await self.graph_db.select(subject=ns_class_name, predicate=GraphKeyword.HAS_CLASS_USE_CASE)
        for i, r in enumerate(rows):
            detail = ReverseUseCaseDetails.model_validate_json(r.object_)
            block += f"\n### {i + 1}. {detail.description}"
            for j, use_case in enumerate(detail.use_cases):
                block += f"\n#### {i + 1}.{j + 1}. {use_case.description}\n"
                block += "\n##### 输入\n" + "\n".join([f"- {s}" for s in use_case.inputs])
                block += "\n##### 输出\n" + "\n".join([f"- {s}" for s in use_case.outputs])
                block += "\n##### 参与者\n" + "\n".join([f"- {s}" for s in use_case.actors])
                block += "\n##### 步骤\n" + "\n".join([f"- {s}" for s in use_case.steps])
            block += "\n#### 用例关系\n" + "\n".join([f"- {s}" for s in detail.relationship])
        return block + "\n"

    async def _get_class_detail(self, ns_class_name: str) -> DotClassInfo | None:
        """
        异步获取命名空间前缀类的 dot 格式详细信息。

        参数:
            ns_class_name (str): 需要获取详细信息的命名空间前缀类名。

        返回:
            Union[DotClassInfo, None]: 返回 DotClassInfo 对象表示 dot 格式的类详细信息，若无详细信息则返回 None。
        """
        rows = await self.graph_db.select(subject=ns_class_name, predicate=GraphKeyword.HAS_DETAIL)
        if not rows:
            return None
        dot_class_info = DotClassInfo.model_validate_json(rows[0].object_)
        return dot_class_info

    async def _get_uml_class_view(self, ns_class_name: str) -> UMLClassView | None:
        """
        异步获取命名空间前缀类的 UML 2.0 格式详细信息。

        参数:
            ns_class_name (str): 需要获取 UML 2.0 类视图信息的命名空间前缀类名。

        返回:
            Union[UMLClassView, None]: 返回 UMLClassView 对象表示 UML 2.0 格式的类视图，若无视图则返回 None。
        """
        rows = await self.graph_db.select(subject=ns_class_name, predicate=GraphKeyword.HAS_CLASS_VIEW)
        if not rows:
            return None
        class_view = UMLClassView.model_validate_json(rows[0].object_)
        return class_view

    async def _get_source_code(self, ns_class_name: str) -> str:
        """
        异步获取指定命名空间前缀类的源代码。

        参数:
            ns_class_name (str): 需要获取源代码的命名空间前缀类名。

        返回:
            str: 包含该类源代码的字符串。
        """
        rows = await self.graph_db.select(subject=ns_class_name, predicate=GraphKeyword.HAS_PAGE_INFO)
        filename = split_namespace(ns_class_name=ns_class_name)[0]
        if not rows:
            src_filename = RebuildSequenceView.get_full_filename(root=self.i_context, pathname=filename)
            if not src_filename:
                return ""
            return await aread(filename=src_filename, encoding="utf-8")
        code_block_info = CodeBlockInfo.model_validate_json(rows[0].object_)
        return await read_file_block(
            filename=filename, lineno=code_block_info.lineno, end_lineno=code_block_info.end_lineno
        )

    @staticmethod
    def get_full_filename(root: str | Path, pathname: str | Path) -> Path | None:
        """
        将包名称转换为模块的完整路径。

        参数:
            root (Union[str, Path]): 根路径或字符串表示的包路径。
            pathname (Union[str, Path]): 模块的路径或字符串表示。

        返回:
            Union[Path, None]: 返回模块的完整路径，若无法确定路径则返回 None。

        示例:
            如果 `root`(工作目录) 为 "/User/xxx/github/MetaGPT/metagpt"，而 `pathname` 为
            "metagpt/management/skill_manager.py"，则返回值为
            "/User/xxx/github/MetaGPT/metagpt/management/skill_manager.py"
        """
        if re.match(r"^/.+", str(pathname)):
            return pathname
        files = list_files(root=root)
        postfix = "/" + str(pathname)
        for i in files:
            if str(i).endswith(postfix):
                return i
        return None

    @staticmethod
    def parse_participant(mermaid_sequence_diagram: str) -> List[str]:
        """
        解析提供的 Mermaid 序列图，返回参与者列表。

        参数:
            mermaid_sequence_diagram (str): 要解析的 Mermaid 序列图字符串。

        返回:
            List[str]: 从序列图中提取的参与者列表。
        """
        pattern = r"participant ([\w\.]+)"
        matches = re.findall(pattern, mermaid_sequence_diagram)
        matches = [re.sub(r"[\\/'\"]+", "", i) for i in matches]
        return matches

    async def _search_new_participant(self, entry: SPO) -> str | None:
        """
        异步查找一个尚未增强序列图的参与者。

        参数:
            entry (SPO): 表示图数据库中关系的 SPO 对象。

        返回:
            Union[str, None]: 找到的未增强序列图的参与者，若未找到则返回 None。
        """
        rows = await self.graph_db.select(subject=entry.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW)
        if not rows:
            return None
        sequence_view = rows[0].object_
        rows = await self.graph_db.select(subject=entry.subject, predicate=GraphKeyword.HAS_PARTICIPANT)
        merged_participants = []
        for r in rows:
            name = split_namespace(r.object_)[-1]
            merged_participants.append(name)
        participants = self.parse_participant(sequence_view)
        for p in participants:
            if p in merged_participants:
                continue
            return p
        return None


    @retry(
        wait=wait_random_exponential(min=1, max=20),  # 设置重试时的等待时间为随机指数分布，最小1秒，最大20秒
        stop=stop_after_attempt(6),  # 设置最多重试6次
        after=general_after_log(logger),  # 在每次重试后记录日志
    )
    async def _merge_participant(self, entry: SPO, class_name: str):
        """
        将`class_name`的序列图增强到`entry`的序列图中。

        参数:
            entry (SPO): 代表基础序列图的SPO对象。
            class_name (str): 要增强的类名对应的序列图。
        """
        rows = await self.graph_db.select(predicate=GraphKeyword.IS, object_=GraphKeyword.CLASS)
        participants = []
        for r in rows:
            name = split_namespace(r.subject)[-1]
            if name == class_name:  # 如果类名匹配
                participants.append(r)
        if len(participants) == 0:  # 如果没有找到匹配的参与者，表示是外部参与者
            await self.graph_db.insert(
                subject=entry.subject, predicate=GraphKeyword.HAS_PARTICIPANT, object_=concat_namespace("?", class_name)
            )
            return
        if len(participants) > 1:  # 如果有多个匹配的参与者
            for r in participants:
                await self.graph_db.insert(
                    subject=entry.subject, predicate=GraphKeyword.HAS_PARTICIPANT, object_=auto_namespace(r.subject)
                )
            return

        participant = participants[0]  # 获取第一个参与者
        await self._rebuild_sequence_view(participant.subject)
        sequence_views = await self.graph_db.select(
            subject=participant.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW
        )
        if not sequence_views:  # 如果没有找到序列视图，表示该类是外部类
            return
        rows = await self.graph_db.select(subject=entry.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW)
        prompt = f"```mermaid\n{sequence_views[0].object_}\n```\n---\n```mermaid\n{rows[0].object_}\n```"

        rsp = await self.llm.aask(
            prompt,
            system_msgs=[  # 提供给LLM的系统消息，指导其处理
                "你是一个将序列图合并为一个的工具。",
                "同名的参与者被认为是相同的。",
                "返回合并后的Mermaid序列图，格式为Markdown代码块。",
            ],
            stream=False,
        )

        sequence_view = rsp.removeprefix("```mermaid").removesuffix("```")  # 去除Mermaid代码块标记
        rows = await self.graph_db.select(subject=entry.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW)
        for r in rows:
            await self.graph_db.delete(subject=r.subject, predicate=r.predicate, object_=r.object_)
        await self.graph_db.insert(
            subject=entry.subject, predicate=GraphKeyword.HAS_SEQUENCE_VIEW, object_=sequence_view
        )
        await self.graph_db.insert(
            subject=entry.subject,
            predicate=GraphKeyword.HAS_SEQUENCE_VIEW_VER,
            object_=concat_namespace(datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3], add_affix(sequence_view)),
        )
        await self.graph_db.insert(
            subject=entry.subject, predicate=GraphKeyword.HAS_PARTICIPANT, object_=auto_namespace(participant.subject)
        )
        await self._save_sequence_view(subject=entry.subject, content=sequence_view)  # 保存合并后的序列图

    async def _save_sequence_view(self, subject: str, content: str):
        pattern = re.compile(r"[^a-zA-Z0-9]")  # 匹配非字母数字字符
        name = re.sub(pattern, "_", subject)  # 替换为下划线
        filename = Path(name).with_suffix(".sequence_diagram.mmd")  # 生成文件名，后缀为.mmd
        await self.context.repo.resources.data_api_design.save(filename=str(filename), content=content)  # 保存文件

    async def _search_participants(self, filename: str) -> Set:
        content = await self._get_source_code(filename)  # 获取源代码

        rsp = await self.llm.aask(
            msg=content,
            system_msgs=[  # 提供给LLM的系统消息，指导其处理
                "你是一个列出源文件中所有类名的工具。",
                "返回一个Markdown格式的JSON对象，包含："
                '- 一个"class_names"键，表示文件中使用的所有类名列表；'
                '- 一个"reasons"键，列出所有原因对象，每个对象包含"class_name"键表示类名，"reference"键解释类在文件中的引用位置。',
            ],
        )

        class _Data(BaseModel):
            class_names: List[str]
            reasons: List

        json_blocks = parse_json_code_block(rsp)  # 解析JSON代码块
        data = _Data.model_validate_json(json_blocks[0])  # 验证数据并转为模型实例
        return set(data.class_names)  # 返回类名集合
