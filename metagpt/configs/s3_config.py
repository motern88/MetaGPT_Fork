#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 19:07
@Author  : alexanderwu
@File    : s3_config.py
"""
from metagpt.utils.yaml_model import YamlModelWithoutDefault


class S3Config(YamlModelWithoutDefault):
    """S3配置类
    该类用于配置连接到 S3 服务所需的认证和存储桶信息。

    属性：
        access_key (str): S3的访问密钥，用于身份验证。
        secret_key (str): S3的秘密密钥，用于身份验证。
        endpoint (str): S3服务的端点URL。
        bucket (str): 存储桶的名称。
    """

    access_key: str  # S3的访问密钥，用于身份验证。
    secret_key: str  # S3的秘密密钥，用于身份验证。
    endpoint: str  # S3服务的端点URL。
    bucket: str  # 存储桶的名称。
