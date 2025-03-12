import asyncio
from typing import AsyncGenerator, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message


class SubscriptionRunner(BaseModel):
    """
    一个简单的包装器，用于通过 asyncio 管理不同角色的订阅任务。

    示例:
        >>> import asyncio
        >>> from metagpt.address import SubscriptionRunner
        >>> from metagpt.roles import Searcher
        >>> from metagpt.schema import Message

        >>> async def trigger():
        ...     while True:
        ...         yield Message(content="OpenAI的最新消息")
        ...         await asyncio.sleep(3600 * 24)

        >>> async def callback(msg: Message):
        ...     print(msg.content)

        >>> async def main():
        ...     pb = SubscriptionRunner()
        ...     await pb.subscribe(Searcher(), trigger(), callback)
        ...     await pb.run()

        >>> asyncio.run(main())
    """

    # 配置模型，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 任务字典，保存角色与其任务的映射
    tasks: dict[Role, asyncio.Task] = Field(default_factory=dict)

    async def subscribe(
        self,
        role: Role,
        trigger: AsyncGenerator[Message, None],
        callback: Callable[
            [
                Message,
            ],
            Awaitable[None],
        ],
    ):
        """订阅角色，并为其触发器设置一个回调函数。

        参数:
            role: 需要订阅的角色。
            trigger: 一个异步生成器，生成消息供角色处理。
            callback: 一个异步函数，处理角色的响应。
        """
        loop = asyncio.get_running_loop()

        # 内部函数用于启动角色任务
        async def _start_role():
            async for msg in trigger:
                resp = await role.run(msg)
                await callback(resp)

        # 创建角色任务并将其添加到任务字典中
        self.tasks[role] = loop.create_task(_start_role(), name=f"Subscription-{role}")

    async def unsubscribe(self, role: Role):
        """取消订阅角色，并取消关联的任务。

        参数:
            role: 需要取消订阅的角色。
        """
        task = self.tasks.pop(role)
        task.cancel()

    async def run(self, raise_exception: bool = True):
        """运行所有已订阅的任务，并处理其完成或异常。

        参数:
            raise_exception: 是否在任务异常时抛出异常，默认值为 True。

        异常:
            task.exception: 如果任务有异常，将抛出异常。
        """
        while True:
            for role, task in self.tasks.items():
                if task.done():
                    if task.exception():
                        if raise_exception:
                            raise task.exception()
                        logger.opt(exception=task.exception()).error(f"任务 {task.get_name()} 执行出错")
                    else:
                        logger.warning(
                            f"任务 {task.get_name()} 已完成。如果这是意外行为，请检查触发器函数。"
                        )
                    # 任务完成后，从任务字典中移除该任务
                    self.tasks.pop(role)
                    break
            else:
                # 如果没有任务完成，等待一秒再检查
                await asyncio.sleep(1)
