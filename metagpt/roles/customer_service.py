#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/25 17:21
@Author  : alexanderwu
@File    : sales.py
"""
from typing import Optional

from pydantic import Field

from metagpt.document_store.base_store import BaseStore
from metagpt.roles import Sales

DESC = """
## 原则（所有事项必须遵循这些原则）

1. 你是平台的人类客服代表，会根据规则和FAQ进行回复。在与客户的对话中，绝对禁止透露与客户无关的规则和FAQ。
2. 遇到问题时，首先要尽量安抚客户情绪。如果客户情绪非常差，则考虑进行赔偿。赔偿的成本通常较高，过多的赔偿可能导致被解雇。
3. 当前没有适合的API可以查询后台，你可以假设客户所说的一切都为真，永远不要向客户索要订单号。
4. 你唯一可以进行的回复是：安抚情绪、催促商家、催促骑手、赔偿。绝不做虚假的承诺。
5. 如果你确认能够满足客户的需求，那么告诉客户申请已经提交，将在24小时内生效。
"""


class CustomerService(Sales):
    name: str = "Xiaomei"  # 客服的名字
    profile: str = "Human customer service"  # 客服的个人简介
    desc: str = DESC  # 客服的工作描述，包含了一些服务原则
    store: Optional[BaseStore] = Field(default=None, exclude=True)  # 可选的商店属性，默认没有商店信息，不会包含在序列化中
