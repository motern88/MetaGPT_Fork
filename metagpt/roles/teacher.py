#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/27
@Author  : mashenquan
@File    : teacher.py
@Desc    : Used by Agent Store
@Modified By: mashenquan, 2023/8/22. A definition has been provided for the return value of _think: returning false indicates that further reasoning cannot continue.

"""

import re

from metagpt.actions import UserRequirement
from metagpt.actions.write_teaching_plan import TeachingPlanBlock, WriteTeachingPlanPart
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.common import any_to_str, awrite


class Teacher(Role):
    """
    支持可配置的教师角色，教学语言和本地语言可以通过配置进行替换。
    """

    name: str = "Lily"  # 教师的名字，默认为 Lily
    profile: str = "{teaching_language} Teacher"  # 教师的角色描述，使用教学语言填充
    goal: str = "writing a {language} teaching plan part by part"  # 教师的目标，分部分编写教学计划
    constraints: str = "writing in {language}"  # 教师的约束条件，编写内容必须使用指定语言
    desc: str = ""  # 教师描述，默认为空

    def __init__(self, **kwargs):
        """初始化教师角色并根据上下文替换相关字段的值"""
        super().__init__(**kwargs)
        # 使用 WriteTeachingPlanPart 格式化方法替换各个字段中的内容
        self.name = WriteTeachingPlanPart.format_value(self.name, self.context)
        self.profile = WriteTeachingPlanPart.format_value(self.profile, self.context)
        self.goal = WriteTeachingPlanPart.format_value(self.goal, self.context)
        self.constraints = WriteTeachingPlanPart.format_value(self.constraints, self.context)
        self.desc = WriteTeachingPlanPart.format_value(self.desc, self.context)

    async def _think(self) -> bool:
        """分部分完成教学计划的编写"""
        if not self.actions:  # 如果没有设定任何动作
            # 如果没有最新的新闻或者新闻的来源不符合要求，则抛出异常
            if not self.rc.news or self.rc.news[0].cause_by != any_to_str(UserRequirement):
                raise ValueError("Lesson content invalid.")
            actions = []
            print(TeachingPlanBlock.TOPICS)  # 打印教学计划主题
            for topic in TeachingPlanBlock.TOPICS:  # 对每个主题创建一个教学计划部分动作
                act = WriteTeachingPlanPart(i_context=self.rc.news[0].content, topic=topic, llm=self.llm)
                actions.append(act)
            self.set_actions(actions)  # 设置教师角色的动作

        # 如果没有待做的任务，设置状态为 0 并返回
        if self.rc.todo is None:
            self._set_state(0)
            return True

        # 如果状态小于总状态数，更新状态并返回
        if self.rc.state + 1 < len(self.states):
            self._set_state(self.rc.state + 1)
            return True

        # 如果任务完成，设置待做任务为 None
        self.set_todo(None)
        return False

    async def _react(self) -> Message:
        """根据思考的结果生成反应"""
        ret = Message(content="")  # 初始化一个空的消息
        while True:
            await self._think()  # 执行思考过程
            if self.rc.todo is None:
                break  # 如果没有待做的任务，则跳出循环
            logger.debug(f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")  # 打印当前状态和任务
            msg = await self._act()  # 执行动作
            if ret.content != "":
                ret.content += "\n\n\n"  # 如果消息内容不为空，添加分隔符
            ret.content += msg.content  # 将消息内容追加到返回消息中
        logger.info(ret.content)  # 打印生成的教学计划内容
        await self.save(ret.content)  # 保存教学计划内容
        return ret  # 返回生成的消息

    async def save(self, content):
        """保存教学计划到文件"""
        filename = Teacher.new_file_name(self.course_title)  # 根据课程标题生成文件名
        pathname = self.config.workspace.path / "teaching_plan"  # 定义文件保存路径
        pathname.mkdir(exist_ok=True)  # 如果目录不存在，则创建目录
        pathname = pathname / filename  # 拼接完整路径
        await awrite(pathname, content)  # 异步写入内容到文件
        logger.info(f"Save to:{pathname}")  # 打印保存的路径

    @staticmethod
    def new_file_name(lesson_title, ext=".md"):
        """根据 `lesson_title` 和扩展名 `ext` 创建相关的文件名"""
        # 定义需要替换的特殊字符
        illegal_chars = r'[#@$%!*&\\/:*?"<>|\n\t \']'
        # 用下划线替换特殊字符
        filename = re.sub(illegal_chars, "_", lesson_title) + ext
        return re.sub(r"_+", "_", filename)  # 替换多个下划线为一个下划线

    @property
    def course_title(self):
        """返回教学计划的课程标题"""
        default_title = "teaching_plan"  # 默认标题为 teaching_plan
        for act in self.actions:
            if act.topic != TeachingPlanBlock.COURSE_TITLE:  # 如果主题不是课程标题，跳过
                continue
            if act.rsp is None:
                return default_title  # 如果响应为空，返回默认标题
            title = act.rsp.lstrip("# \n")  # 去除响应前缀的特殊字符
            if "\n" in title:
                ix = title.index("\n")  # 获取换行符的位置
                title = title[0:ix]  # 截取课程标题的有效部分
            return title  # 返回标题

        return default_title  # 如果没有找到有效的课程标题，返回默认标题