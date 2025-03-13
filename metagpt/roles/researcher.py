#!/usr/bin/env python
"""
@Modified By: mashenquan, 2023/8/22. A definition has been provided for the return value of _think: returning false indicates that further reasoning cannot continue.
@Modified By: mashenquan, 2023-11-1. According to Chapter 2.2.1 and 2.2.2 of RFC 116, change the data type of
        the `cause_by` value in the `Message` to a string to support the new message distribution feature.
"""

import asyncio
import re

from pydantic import BaseModel

from metagpt.actions import Action, CollectLinks, ConductResearch, WebBrowseAndSummarize
from metagpt.actions.research import get_research_system_text
from metagpt.const import RESEARCH_PATH
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message


class Report(BaseModel):
    topic: str  # 研究报告的主题
    links: dict[str, list[str]] = None  # 存储链接及其对应的描述
    summaries: list[tuple[str, str]] = None  # 存储链接及其总结的元组列表
    content: str = ""  # 研究报告的内容


class Researcher(Role):
    name: str = "David"  # 角色名称
    profile: str = "Researcher"  # 角色描述
    goal: str = "Gather information and conduct research"  # 目标：收集信息并进行研究
    constraints: str = "Ensure accuracy and relevance of information"  # 限制条件：确保信息的准确性和相关性
    language: str = "en-us"  # 默认语言为英语（美国）
    enable_concurrency: bool = True  # 是否启用并发处理

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 初始化父类
        self.set_actions([CollectLinks, WebBrowseAndSummarize, ConductResearch])  # 设置当前角色的任务动作
        self._set_react_mode(RoleReactMode.BY_ORDER.value, len(self.actions))  # 设置响应模式
        if self.language not in ("en-us", "zh-cn"):
            logger.warning(f"The language `{self.language}` has not been tested, it may not work.")  # 警告：当前语言未测试，可能无法正常工作

    async def _act(self) -> Message:
        """执行任务并返回消息"""
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")  # 输出任务日志
        todo = self.rc.todo  # 获取当前任务
        msg = self.rc.memory.get(k=1)[0]  # 获取记忆中的最新消息
        if isinstance(msg.instruct_content, Report):
            instruct_content = msg.instruct_content
            topic = instruct_content.topic  # 从消息中获取报告主题
        else:
            topic = msg.content  # 否则使用消息内容作为主题

        research_system_text = self.research_system_text(topic, todo)  # 获取研究系统文本
        if isinstance(todo, CollectLinks):
            links = await todo.run(topic, 4, 4)  # 执行链接收集任务
            ret = Message(
                content="", instruct_content=Report(topic=topic, links=links), role=self.profile, cause_by=todo
            )  # 返回收集到的链接
        elif isinstance(todo, WebBrowseAndSummarize):
            links = instruct_content.links  # 获取链接
            todos = (
                todo.run(*url, query=query, system_text=research_system_text) for (query, url) in links.items() if url
            )
            if self.enable_concurrency:
                summaries = await asyncio.gather(*todos)  # 启用并发执行所有任务
            else:
                summaries = [await i for i in todos]  # 顺序执行任务
            summaries = list((url, summary) for i in summaries for (url, summary) in i.items() if summary)  # 过滤无总结的链接
            ret = Message(
                content="", instruct_content=Report(topic=topic, summaries=summaries), role=self.profile, cause_by=todo
            )  # 返回总结后的报告
        else:
            summaries = instruct_content.summaries  # 获取已有总结
            summary_text = "\n---\n".join(f"url: {url}\nsummary: {summary}" for (url, summary) in summaries)  # 拼接总结文本
            content = await self.rc.todo.run(topic, summary_text, system_text=research_system_text)  # 执行任务并生成报告内容
            ret = Message(
                content="",
                instruct_content=Report(topic=topic, content=content),
                role=self.profile,
                cause_by=self.rc.todo,
            )  # 返回完整报告
        self.rc.memory.add(ret)  # 将生成的报告添加到记忆中
        return ret

    def research_system_text(self, topic, current_task: Action) -> str:
        """返回用于研究的系统文本，支持子类自定义

        允许子类根据主题定义其自己的系统提示。
        返回以前的实现，以确保向后兼容
        Args:
            topic:
            language:

        Returns: str
        """
        return get_research_system_text(topic, self.language)

    async def react(self) -> Message:
        """响应操作并生成报告"""
        msg = await super().react()  # 调用父类的反应方法
        report = msg.instruct_content  # 获取报告内容
        self.write_report(report.topic, report.content)  # 写入报告
        return msg

    def write_report(self, topic: str, content: str):
        """将研究报告保存为 Markdown 文件"""
        filename = re.sub(r'[\\/:"*?<>|]+', " ", topic)  # 清理非法文件名字符
        filename = filename.replace("\n", "")  # 去除换行符
        if not RESEARCH_PATH.exists():
            RESEARCH_PATH.mkdir(parents=True)  # 如果研究路径不存在，创建它
        filepath = RESEARCH_PATH / f"{filename}.md"  # 生成文件路径
        filepath.write_text(content)  # 将内容写入文件


if __name__ == "__main__":
    import fire

    async def main(topic: str, language: str = "en-us", enable_concurrency: bool = True):
        """启动 Researcher 角色并运行任务"""
        role = Researcher(language=language, enable_concurrency=enable_concurrency)  # 初始化角色
        await role.run(topic)  # 执行任务

    fire.Fire(main)  # 通过 fire 库运行 main 函数
