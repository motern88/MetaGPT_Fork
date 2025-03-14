#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/17
@Author  : mashenquan
@File    : metagpt_oas3_api_svc.py
@Desc    : MetaGPT OpenAPI Specification 3.0 REST API service

        curl -X 'POST' \
        'http://localhost:8080/openapi/greeting/dave' \
        -H 'accept: text/plain' \
        -H 'Content-Type: application/json' \
        -d '{}'
"""

from pathlib import Path

import connexion


def oas_http_svc():
    """启动 OAS 3.0 OpenAPI HTTP 服务"""
    print("http://localhost:8080/oas3/ui/")  # 输出 OpenAPI UI 的访问地址
    specification_dir = Path(__file__).parent.parent.parent / "docs/.well-known"  # 指定 OpenAPI 文档的目录
    app = connexion.AsyncApp(__name__, specification_dir=str(specification_dir))  # 创建一个异步的 API 应用实例
    app.add_api("metagpt_oas3_api.yaml")  # 添加第一个 API 规范
    app.add_api("openapi.yaml")  # 添加第二个 API 规范
    app.run(port=8080)  # 启动应用，并监听 8080 端口


if __name__ == "__main__":
    oas_http_svc()  # 如果是主程序，启动 HTTP 服务
