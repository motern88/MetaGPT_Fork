#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : tutorial_assistant.py
"""

from datetime import datetime
from typing import Dict

from metagpt.actions.write_tutorial import WriteContent, WriteDirectory
from metagpt.const import TUTORIAL_PATH
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.utils.file import File


class TutorialAssistant(Role):
    """
    教程助手，输入一句话生成教程文档，格式为 Markdown。

    参数:
        name: 角色的名字。
        profile: 角色的描述。
        goal: 角色的目标。
        constraints: 角色的约束或要求。
        language: 教程文档生成的语言。
    """

    name: str = "Stitch"  # 角色名字，默认为 Stitch
    profile: str = "Tutorial Assistant"  # 角色描述，默认为 "Tutorial Assistant"
    goal: str = "Generate tutorial documents"  # 角色的目标，生成教程文档
    constraints: str = "Strictly follow Markdown's syntax, with neat and standardized layout"  # 角色的约束，严格按照 Markdown 语法，布局整洁标准化
    language: str = "Chinese"  # 生成教程文档的语言，默认为中文

    topic: str = ""  # 教程的主题
    main_title: str = ""  # 教程的主标题
    total_content: str = ""  # 完整的教程内容

    def __init__(self, **kwargs):
        """初始化教程助手角色"""
        super().__init__(**kwargs)
        self.set_actions([WriteDirectory(language=self.language)])  # 设置初始动作为写目录
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)  # 设置反应模式为顺序反应

    async def _handle_directory(self, titles: Dict) -> Message:
        """
        处理教程文档的目录部分。

        参数:
            titles: 包含标题和目录结构的字典，例如：
                    {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}

        返回:
            包含目录信息的消息。
        """
        self.main_title = titles.get("title")  # 获取主标题
        directory = f"{self.main_title}\n"  # 初始化目录字符串
        self.total_content += f"# {self.main_title}"  # 将主标题添加到教程内容中
        actions = list(self.actions)  # 获取当前的动作列表
        for first_dir in titles.get("directory"):  # 遍历目录
            actions.append(WriteContent(language=self.language, directory=first_dir))  # 添加写内容的动作
            key = list(first_dir.keys())[0]  # 获取目录的主项
            directory += f"- {key}\n"  # 添加主项到目录中
            for second_dir in first_dir[key]:  # 遍历子目录
                directory += f"  - {second_dir}\n"  # 添加子目录到目录中
        self.set_actions(actions)  # 更新角色的动作
        self.rc.max_react_loop = len(self.actions)  # 设置反应的最大循环次数为动作数
        return Message()  # 返回一个空消息

    async def _act(self) -> Message:
        """根据角色的决策执行一个动作，并返回结果消息"""
        todo = self.rc.todo  # 获取待办任务
        if type(todo) is WriteDirectory:  # 如果任务是写目录
            msg = self.rc.memory.get(k=1)[0]  # 获取记忆中的消息
            self.topic = msg.content  # 将消息内容设为当前主题
            resp = await todo.run(topic=self.topic)  # 执行写目录任务
            logger.info(resp)  # 打印响应
            return await self._handle_directory(resp)  # 处理目录并返回结果
        resp = await todo.run(topic=self.topic)  # 执行任务
        logger.info(resp)  # 打印响应
        if self.total_content != "":  # 如果已有内容
            self.total_content += "\n\n\n"  # 添加分隔符
        self.total_content += resp  # 将响应内容添加到完整教程中
        return Message(content=resp, role=self.profile)  # 返回包含响应的消息

    async def react(self) -> Message:
        """生成教程并保存文件"""
        msg = await super().react()  # 调用父类的 react 方法
        root_path = TUTORIAL_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取保存路径，按日期和时间命名
        await File.write(root_path, f"{self.main_title}.md", self.total_content.encode("utf-8"))  # 保存文件
        msg.content = str(root_path / f"{self.main_title}.md")  # 设置消息内容为文件路径
        return msg  # 返回保存路径的消息
