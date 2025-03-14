#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 16:03
@Author  : mashenquan
@File    : openapi_v3_hello.py
@Desc    : Implement the OpenAPI Specification 3.0 demo and use the following command to test the HTTP service:

        curl -X 'POST' \
        'http://localhost:8082/openapi/greeting/dave' \
        -H 'accept: text/plain' \
        -H 'Content-Type: application/json' \
        -d '{}'
"""
from pathlib import Path

import connexion


# openapi 实现
async def post_greeting(name: str) -> str:
    """根据输入的名字返回问候语

    :param name: 用户的名字
    :return: 返回一个包含用户名字的问候语
    """
    return f"Hello {name}\n"


if __name__ == "__main__":
    # 获取 OpenAPI 规范文件所在的目录
    specification_dir = Path(__file__).parent.parent.parent / "docs/.well-known"

    # 创建一个异步应用实例
    app = connexion.AsyncApp(__name__, specification_dir=str(specification_dir))

    # 加载 OpenAPI 规范文件，并将其与应用绑定
    app.add_api("openapi.yaml", arguments={"title": "Hello World Example"})

    # 启动应用，监听 8082 端口
    app.run(port=8082)
