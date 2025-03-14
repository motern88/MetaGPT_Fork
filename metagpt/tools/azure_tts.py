#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/9 22:22
@Author  : Leo Xiao
@File    : azure_tts.py
@Modified by: mashenquan, 2023/8/17. Azure TTS OAS3 api, which provides text-to-speech functionality
"""
import base64
from pathlib import Path
from uuid import uuid4

import aiofiles
from azure.cognitiveservices.speech import AudioConfig, SpeechConfig, SpeechSynthesizer

from metagpt.logs import logger


class AzureTTS:
    """Azure 语音合成 (Text-to-Speech)"""

    def __init__(self, subscription_key, region):
        """
        初始化 Azure TTS 客户端

        :param subscription_key: 用于访问 Azure AI 服务 API 的密钥，见：https://portal.azure.com/ > 资源管理 > 密钥和终结点
        :param region: 资源所在的区域，在调用 API 时需要使用该字段。
        """
        self.subscription_key = subscription_key
        self.region = region

    # 参数参考：https://learn.microsoft.com/zh-cn/azure/cognitive-services/speech-service/language-support?tabs=tts#voice-styles-and-roles
    async def synthesize_speech(self, lang, voice, text, output_file):
        """使用 Azure 语音合成 API 将文本转换为语音

        :param lang: 语言代码，例如 en（英语），或 locale，例如 en-US（英语-美国）。
        :param voice: 使用的语音名称，详情见：https://learn.microsoft.com/zh-cn/azure/cognitive-services/speech-service/language-support?tabs=tts
        :param text: 要转换为语音的文本内容
        :param output_file: 输出的语音文件路径
        :return: 返回语音合成结果
        """
        # 配置语音合成服务
        speech_config = SpeechConfig(subscription=self.subscription_key, region=self.region)
        speech_config.speech_synthesis_voice_name = voice  # 设置使用的语音
        audio_config = AudioConfig(filename=output_file)  # 设置输出文件路径
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # 使用 SSML 格式定义语音合成内容
        ssml_string = (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xml:lang='{lang}' xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{voice}'>{text}</voice></speak>"
        )

        # 异步执行语音合成
        return synthesizer.speak_ssml_async(ssml_string).get()

    @staticmethod
    def role_style_text(role, style, text):
        """根据角色和风格生成 SSML 格式文本

        :param role: 语音的角色（如：女孩，男孩等）
        :param style: 语音的风格（如：亲切、冷静等）
        :param text: 要转换的文本
        :return: 返回 SSML 格式的文本
        """
        return f'<mstts:express-as role="{role}" style="{style}">{text}</mstts:express-as>'

    @staticmethod
    def role_text(role, text):
        """根据角色生成 SSML 格式文本

        :param role: 语音的角色（如：女孩，男孩等）
        :param text: 要转换的文本
        :return: 返回 SSML 格式的文本
        """
        return f'<mstts:express-as role="{role}">{text}</mstts:express-as>'

    @staticmethod
    def style_text(style, text):
        """根据风格生成 SSML 格式文本

        :param style: 语音的风格（如：亲切、冷静等）
        :param text: 要转换的文本
        :return: 返回 SSML 格式的文本
        """
        return f'<mstts:express-as style="{style}">{text}</mstts:express-as>'


# 导出异步函数
async def oas3_azsure_tts(text, lang="", voice="", style="", role="", subscription_key="", region=""):
    """文本转语音
    详情请参阅：`https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/language-support?tabs=tts`

    :param lang: 语言代码，默认为 zh-CN（中文）。
    :param voice: 使用的语音名称，默认为 zh-CN-XiaomoNeural（中文语音）。
    :param style: 语音风格（如：亲切、冷静等）。
    :param role: 语音角色（如：女孩，男孩等）。
    :param text: 要转换为语音的文本。
    :param subscription_key: 用于访问 Azure AI 服务 API 的密钥。
    :param region: 资源所在区域，用于调用 API。
    :return: 返回 Base64 编码的 .wav 文件数据，成功则返回，失败则返回空字符串。
    """
    if not text:
        return ""  # 如果没有提供文本，则返回空字符串

    # 设置默认参数
    if not lang:
        lang = "zh-CN"
    if not voice:
        voice = "zh-CN-XiaomoNeural"
    if not role:
        role = "Girl"
    if not style:
        style = "affectionate"

    # 根据角色和风格构造 SSML 格式文本
    xml_value = AzureTTS.role_style_text(role=role, style=style, text=text)
    tts = AzureTTS(subscription_key=subscription_key, region=region)

    # 生成文件路径并执行语音合成
    filename = Path(__file__).resolve().parent / (str(uuid4()).replace("-", "") + ".wav")
    try:
        # 调用语音合成服务生成语音文件
        await tts.synthesize_speech(lang=lang, voice=voice, text=xml_value, output_file=str(filename))

        # 读取生成的语音文件并转换为 Base64 编码
        async with aiofiles.open(filename, mode="rb") as reader:
            data = await reader.read()
            base64_string = base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.error(f"text:{text}, error:{e}")
        return ""  # 如果发生错误，返回空字符串
    finally:
        filename.unlink(missing_ok=True)  # 删除临时语音文件

    return base64_string  # 返回 Base64 编码的语音数据