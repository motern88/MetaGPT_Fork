#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/17
@Author  : mashenquan
@File    : text_to_speech.py
@Desc    : Text-to-Speech skill, which provides text-to-speech functionality
"""
from typing import Optional

from metagpt.config2 import Config
from metagpt.const import BASE64_FORMAT
from metagpt.tools.azure_tts import oas3_azsure_tts
from metagpt.tools.iflytek_tts import oas3_iflytek_tts
from metagpt.utils.s3 import S3


async def text_to_speech(
    text,
    lang="zh-CN",
    voice="zh-CN-XiaomoNeural",
    style="affectionate",
    role="Girl",
    config: Optional[Config] = None,
):
    """文本转语音（Text-to-Speech）

    详细信息请参考：`https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`

    :param lang: 语言代码（如 "en" 代表英语）或区域代码（如 "en-US" 代表美国英语）。
                 详细信息参考：`https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param voice: 选择语音的具体 ID，具体语音列表请参考：
                  `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
                  `https://speech.microsoft.com/portal/voicegallery`
    :param style: 语音表达风格，如“愉悦”、“同理心”、“冷静”等。
                  详细信息参考：`https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param role: 语音角色，使同一语音可以表现出不同的年龄和性别。
                 详细信息参考：`https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param text: 需要转换为语音的文本内容。
    :param subscription_key: Azure 语音服务 API 访问密钥，获取方式：
                             `https://portal.azure.com/` > `资源管理` > `密钥和端点`
    :param region: Azure 语音服务的资源所在的区域，在调用 API 时可能需要使用。
    :param iflytek_app_id: iFlyTek（科大讯飞）语音服务的应用 ID，获取方式：
                            `https://console.xfyun.cn/services/tts`
    :param iflytek_api_key: iFlyTek WebAPI 访问密钥，获取方式：
                            `https://console.xfyun.cn/services/tts`
    :param iflytek_api_secret: iFlyTek WebAPI 访问密钥，获取方式：
                               `https://console.xfyun.cn/services/tts`
    :return: 成功时返回 Base64 编码的 .wav/.mp3 音频数据，否则返回空字符串。
    """
    config = config if config else Config.default()
    subscription_key = config.azure_tts_subscription_key
    region = config.azure_tts_region

    # 使用 Azure 语音服务
    if subscription_key and region:
        audio_declaration = "data:audio/wav;base64,"  # 音频 Base64 编码前缀
        base64_data = await oas3_azsure_tts(text, lang, voice, style, role, subscription_key, region)

        # 上传音频数据到 S3，并获取 URL
        s3 = S3(config.s3)
        url = await s3.cache(data=base64_data, file_ext=".wav", format=BASE64_FORMAT)
        if url:
            return f"[{text}]({url})"  # 返回 Markdown 格式的音频链接
        return audio_declaration + base64_data if base64_data else base64_data  # 直接返回 Base64 编码数据

    # 使用 iFlyTek（科大讯飞）语音服务
    iflytek_app_id = config.iflytek_app_id
    iflytek_api_key = config.iflytek_api_key
    iflytek_api_secret = config.iflytek_api_secret
    if iflytek_app_id and iflytek_api_key and iflytek_api_secret:
        audio_declaration = "data:audio/mp3;base64,"  # 音频 Base64 编码前缀
        base64_data = await oas3_iflytek_tts(
            text=text, app_id=iflytek_app_id, api_key=iflytek_api_key, api_secret=iflytek_api_secret
        )

        # 上传音频数据到 S3，并获取 URL
        s3 = S3(config.s3)
        url = await s3.cache(data=base64_data, file_ext=".mp3", format=BASE64_FORMAT)
        if url:
            return f"[{text}]({url})"  # 返回 Markdown 格式的音频链接
        return audio_declaration + base64_data if base64_data else base64_data  # 直接返回 Base64 编码数据

    # 如果两种服务的必要参数都未提供，则抛出异常
    raise ValueError(
        "azure_tts_subscription_key, azure_tts_region, iflytek_app_id, iflytek_api_key, iflytek_api_secret 配置缺失"
    )
