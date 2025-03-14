#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/17
@Author  : mashenquan
@File    : iflytek_tts.py
@Desc    : iFLYTEK TTS OAS3 api, which provides text-to-speech functionality
"""
import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import mktime
from typing import Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import aiofiles
import websockets as websockets
from pydantic import BaseModel

from metagpt.logs import logger


class IFlyTekTTSStatus(Enum):
    STATUS_FIRST_FRAME = 0  # 第一个帧
    STATUS_CONTINUE_FRAME = 1  # 中间帧
    STATUS_LAST_FRAME = 2  # 最后一个帧


class AudioData(BaseModel):
    audio: str  # 音频数据（Base64编码的音频）
    status: int  # 状态码，指示当前帧的状态
    ced: str  # CED信息


class IFlyTekTTSResponse(BaseModel):
    code: int  # 响应码，表示请求的处理结果
    message: str  # 错误信息
    data: Optional[AudioData] = None  # 音频数据，包含音频和状态
    sid: str  # 会话ID


DEFAULT_IFLYTEK_VOICE = "xiaoyan"  # 默认的语音名称


class IFlyTekTTS(object):
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        """
        :param app_id: 应用ID，用于访问您的iFlyTek服务API，详情请见：https://console.xfyun.cn/services/tts
        :param api_key: WebAPI密钥，详情请见：https://console.xfyun.cn/services/tts
        :param api_secret: WebAPI密钥，详情请见：https://console.xfyun.cn/services/tts
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret

    async def synthesize_speech(self, text, output_file: str, voice=DEFAULT_IFLYTEK_VOICE):
        """
        合成语音

        :param text: 需要合成的文本
        :param output_file: 输出的音频文件路径
        :param voice: 语音合成的声音，默认为"xiaoyan"。详情请见：https://www.xfyun.cn/doc/tts/online_tts/API.html#%E6%8E%A5%E5%8F%A3%E8%B0%83%E7%94%A8%E6%B5%81%E7%A8%8B
        """
        url = self._create_url()  # 创建WebSocket请求URL
        data = {
            "common": {"app_id": self.app_id},
            "business": {"aue": "lame", "sfl": 1, "auf": "audio/L16;rate=16000", "vcn": voice, "tte": "utf8"},
            "data": {"status": 2, "text": str(base64.b64encode(text.encode("utf-8")), "UTF8")},
        }
        req = json.dumps(data)  # 将请求数据转换为JSON格式
        async with websockets.connect(url) as websocket:
            # 发送请求
            await websocket.send(req)

            # 接收音频数据帧
            async with aiofiles.open(str(output_file), "wb") as writer:
                while True:
                    v = await websocket.recv()
                    rsp = IFlyTekTTSResponse(**json.loads(v))  # 解析响应数据
                    if rsp.data:
                        binary_data = base64.b64decode(rsp.data.audio)  # 解码音频数据
                        await writer.write(binary_data)  # 将音频数据写入文件
                        if rsp.data.status != IFlyTekTTSStatus.STATUS_LAST_FRAME.value:
                            continue  # 如果不是最后一帧，继续接收下一帧
                    break

    def _create_url(self):
        """创建WebSocket请求的URL"""
        url = "wss://tts-api.xfyun.cn/v2/tts"
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 使用HMAC-SHA256加密生成签名
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"), signature_origin.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
            self.api_key,
            "hmac-sha256",
            "host date request-line",
            signature_sha,
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(encoding="utf-8")
        # 构建WebSocket请求的认证信息
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        # 将认证信息拼接到URL中
        url = url + "?" + urlencode(v)
        return url


# 导出函数
async def oas3_iflytek_tts(text: str, voice: str = "", app_id: str = "", api_key: str = "", api_secret: str = ""):
    """将文本转语音
    详情请查看：`https://www.xfyun.cn/doc/tts/online_tts/API.html`

    :param voice: 默认是`xiaoyan`。更多详情请参考：https://www.xfyun.cn/doc/tts/online_tts/API.html#%E6%8E%A5%E5%8F%A3%E8%B0%83%E7%94%A8%E6%B5%81%E7%A8%8B
    :param text: 需要转换为语音的文本
    :param app_id: 应用ID，用于访问您的iFlyTek服务API，详情请见：https://console.xfyun.cn/services/tts
    :param api_key: WebAPI密钥，详情请见：https://console.xfyun.cn/services/tts
    :param api_secret: WebAPI密钥，详情请见：https://console.xfyun.cn/services/tts
    :return: 返回Base64编码的.mp3音频数据，如果失败则返回空字符串
    """
    filename = Path(__file__).parent / (uuid.uuid4().hex + ".mp3")  # 生成一个唯一的文件名
    try:
        tts = IFlyTekTTS(app_id=app_id, api_key=api_key, api_secret=api_secret)
        await tts.synthesize_speech(text=text, output_file=str(filename), voice=voice)  # 合成语音
        async with aiofiles.open(str(filename), mode="rb") as reader:
            data = await reader.read()  # 读取音频文件内容
            base64_string = base64.b64encode(data).decode("utf-8")  # 将音频文件转为Base64编码
    except Exception as e:
        logger.error(f"text:{text}, error:{e}")  # 捕获异常并记录错误
        base64_string = ""
    finally:
        filename.unlink(missing_ok=True)  # 删除临时文件

    return base64_string  # 返回Base64编码的音频数据