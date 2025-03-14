from __future__ import annotations

from metagpt.const import AGENT, IMAGES, MESSAGE_ROUTE_TO_ALL, TEAMLEADER_NAME
from metagpt.environment.base_env import Environment
from metagpt.logs import get_human_input
from metagpt.roles import Role
from metagpt.schema import Message, SerializationMixin
from metagpt.utils.common import extract_and_encode_images


class MGXEnv(Environment, SerializationMixin):
    """MGX环境类，继承自环境和序列化混合类"""

    direct_chat_roles: set[str] = set()  # 记录直接聊天的角色: @角色名

    is_public_chat: bool = True  # 标志是否为公开聊天

    def _publish_message(self, message: Message, peekable: bool = True) -> bool:
        """发布消息的内部方法，如果是公开聊天，将消息发送给所有人"""
        if self.is_public_chat:
            message.send_to.add(MESSAGE_ROUTE_TO_ALL)  # 如果是公开聊天，发送给所有人
        message = self.move_message_info_to_content(message)  # 移动消息的角色信息到内容中
        return super().publish_message(message, peekable)  # 调用父类的发布消息方法

    def publish_message(self, message: Message, user_defined_recipient: str = "", publicer: str = "") -> bool:
        """消息发布方法，让团队领导负责消息发布"""
        message = self.attach_images(message)  # 为多模态消息附加图片

        tl = self.get_role(TEAMLEADER_NAME)  # 获取团队领导角色，假设团队领导的名字是Mike

        if user_defined_recipient:
            # 如果有用户指定的接收者（即直接聊天消息）
            for role_name in message.send_to:
                if self.get_role(role_name).is_idle:
                    # 如果角色是空闲状态，意味着用户开始了与某个角色的直接聊天，期待该角色的直接回复；其他角色，包括团队领导，不应参与
                    self.direct_chat_roles.add(role_name)  # 将该角色添加到直接聊天角色列表

            self._publish_message(message)  # 发布消息

        elif message.sent_from in self.direct_chat_roles:
            # 如果消息来自于直接聊天角色，并且不是公开聊天，意味着是该角色对用户的直接回复，不需要团队领导及其他角色参与
            self.direct_chat_roles.remove(message.sent_from)  # 从直接聊天角色列表中移除该角色
            if self.is_public_chat:
                self._publish_message(message)  # 如果是公开聊天，发布消息

        elif publicer == tl.profile:
            # 如果消息是由团队领导发布的
            if message.send_to == {"no one"}:
                # 如果团队领导的消息是一个虚拟消息，直接跳过
                return True
            # 否则，团队领导处理的消息可以被发布
            self._publish_message(message)

        else:
            # 其他所有普通消息都通过团队领导进行发布
            message.send_to.add(tl.name)
            self._publish_message(message)

        self.history.add(message)  # 将消息添加到历史记录中

        return True

    async def ask_human(self, question: str, sent_from: Role = None) -> str:
        # 异步方法，向人类提问，可以在远程设置中重写
        rsp = await get_human_input(question)
        return "Human response: " + rsp  # 返回人类的回答

    async def reply_to_human(self, content: str, sent_from: Role = None) -> str:
        # 异步方法，回复人类消息，可以在远程设置中重写
        return "SUCCESS, human has received your reply. Refrain from resending duplicate messages. If you no longer need to take action, use the command ‘end’ to stop."

    def move_message_info_to_content(self, message: Message) -> Message:
        """将消息的角色信息移到内容中，确保角色字段不与 LLM API 冲突，并且将发送者和接收者信息添加到内容中"""
        converted_msg = message.model_copy(deep=True)  # 深拷贝消息对象
        if converted_msg.role not in ["system", "user", "assistant"]:
            converted_msg.role = "assistant"  # 如果角色不是系统、用户或助手，则设置为助手
        sent_from = converted_msg.metadata[AGENT] if AGENT in converted_msg.metadata else converted_msg.sent_from
        # 如果消息的接收者是 MESSAGE_ROUTE_TO_ALL，则改为团队领导
        if converted_msg.send_to == {MESSAGE_ROUTE_TO_ALL}:
            send_to = TEAMLEADER_NAME
        else:
            send_to = ", ".join({role for role in converted_msg.send_to if role != MESSAGE_ROUTE_TO_ALL})  # 去除路由到所有人的角色
        # 将发送者和接收者信息添加到消息内容中
        converted_msg.content = f"[Message] from {sent_from or 'User'} to {send_to}: {converted_msg.content}"
        return converted_msg

    def attach_images(self, message: Message) -> Message:
        """如果消息是用户消息，提取并附加图片"""
        if message.role == "user":
            images = extract_and_encode_images(message.content)  # 提取并编码图片
            if images:
                message.add_metadata(IMAGES, images)  # 将图片添加为元数据
        return message

    def __repr__(self):
        return "MGXEnv()"  # 自定义类的字符串表示