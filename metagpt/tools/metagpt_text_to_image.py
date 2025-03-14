#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : metagpt_text_to_image.py
@Desc    : MetaGPT Text-to-Image OAS3 api, which provides text-to-image functionality.
"""
import base64
from typing import Dict, List

import aiohttp
import requests
from pydantic import BaseModel

from metagpt.logs import logger


class MetaGPTText2Image:
    def __init__(self, model_url):
        """
        :param model_url: 模型重置 API 的 URL
        """
        self.model_url = model_url

    async def text_2_image(self, text, size_type="512x512"):
        """文本生成图像

        :param text: 用于生成图像的文本。
        :param size_type: 图像的尺寸类型，支持 ['512x512', '512x768']。
        :return: 返回 Base64 编码的图像数据。
        """

        headers = {"Content-Type": "application/json"}  # 设置请求头为 JSON 格式
        dims = size_type.split("x")  # 解析图像尺寸
        data = {
            "prompt": text,  # 输入的文本提示
            "negative_prompt": "(easynegative:0.8),black, dark,Low resolution",  # 负面提示，帮助模型避免生成不良图像
            "override_settings": {"sd_model_checkpoint": "galaxytimemachinesGTM_photoV20"},  # 设置模型的检查点
            "seed": -1,  # 随机种子，-1 表示随机生成
            "batch_size": 1,  # 每次生成的图像数量
            "n_iter": 1,  # 生成迭代次数
            "steps": 20,  # 生成图像的步骤数
            "cfg_scale": 11,  # 配置比例，控制生成图像的相关性
            "width": int(dims[0]),  # 图像宽度
            "height": int(dims[1]),  # 图像高度
            "restore_faces": False,  # 是否恢复面部细节
            "tiling": False,  # 是否使用平铺效果
            "do_not_save_samples": False,  # 是否不保存样本
            "do_not_save_grid": False,  # 是否不保存网格
            "enable_hr": False,  # 是否启用高分辨率模式
            "hr_scale": 2,  # 高分辨率模式的缩放比例
            "hr_upscaler": "Latent",  # 高分辨率模式的上采样器
            "hr_second_pass_steps": 0,  # 高分辨率模式的第二步
            "hr_resize_x": 0,  # 高分辨率模式的 X 轴缩放
            "hr_resize_y": 0,  # 高分辨率模式的 Y 轴缩放
            "hr_upscale_to_x": 0,  # 高分辨率模式的 X 轴目标分辨率
            "hr_upscale_to_y": 0,  # 高分辨率模式的 Y 轴目标分辨率
            "truncate_x": 0,  # X 轴截断
            "truncate_y": 0,  # Y 轴截断
            "applied_old_hires_behavior_to": None,  # 应用旧的高分辨率行为
            "eta": None,  # 生成过程中的随机噪声
            "sampler_index": "DPM++ SDE Karras",  # 选择采样器
            "alwayson_scripts": {},  # 持续运行的脚本
        }

        class ImageResult(BaseModel):
            images: List  # 图像列表
            parameters: Dict  # 参数字典

        try:
            # 使用 aiohttp 异步请求
            async with aiohttp.ClientSession() as session:
                async with session.post(self.model_url, headers=headers, json=data) as response:
                    result = ImageResult(**await response.json())  # 获取响应结果
            if len(result.images) == 0:
                return 0  # 如果没有图像数据，返回 0
            data = base64.b64decode(result.images[0])  # 解码图像的 Base64 数据
            return data  # 返回图像数据
        except requests.exceptions.RequestException as e:
            logger.error(f"发生错误: {e}")  # 捕获请求异常并记录错误
        return 0  # 出现异常时返回 0


# 导出
async def oas3_metagpt_text_to_image(text, size_type: str = "512x512", model_url=""):
    """文本转图像

    :param text: 用于生成图像的文本。
    :param model_url: 模型重置 API 的 URL
    :param size_type: 图像的尺寸类型，支持 ['512x512', '512x768']。
    :return: 返回 Base64 编码的图像数据。
    """
    if not text:
        return ""  # 如果没有文本，返回空字符串
    return await MetaGPTText2Image(model_url).text_2_image(text, size_type=size_type)  # 调用 MetaGPTText2Image 进行文本转图像