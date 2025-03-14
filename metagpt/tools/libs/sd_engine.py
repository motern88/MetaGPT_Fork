# -*- coding: utf-8 -*-
# @Date    : 2023/7/19 16:28
# @Author  : stellahong (stellahong@deepwisdom.ai)
# @Desc    :
from __future__ import annotations

import base64
import hashlib
import io
import json
from os.path import join

import requests
from aiohttp import ClientSession
from PIL import Image, PngImagePlugin

from metagpt.const import SD_OUTPUT_FILE_REPO, SD_URL, SOURCE_ROOT
from metagpt.logs import logger
from metagpt.tools.tool_registry import register_tool

payload = {
    "prompt": "",
    "negative_prompt": "(easynegative:0.8),black, dark,Low resolution",
    "override_settings": {"sd_model_checkpoint": "galaxytimemachinesGTM_photoV20"},
    "seed": -1,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "width": 512,
    "height": 768,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "enable_hr": False,
    "hr_scale": 2,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_upscale_to_x": 0,
    "hr_upscale_to_y": 0,
    "truncate_x": 0,
    "truncate_y": 0,
    "applied_old_hires_behavior_to": None,
    "eta": None,
    "sampler_index": "DPM++ SDE Karras",
    "alwayson_scripts": {},
}

default_negative_prompt = "(easynegative:0.8),black, dark,Low resolution"


@register_tool(
    tags=["text2image", "multimodal"],
    include_functions=["__init__", "simple_run_t2i", "run_t2i", "construct_payload", "save"],
)
class SDEngine:
    """基于 Stable Diffusion 生成图像的类。

    该类提供了与 Stable Diffusion 服务交互的方法，根据文本输入生成图像。
    """

    def __init__(self, sd_url=""):
        """初始化 SDEngine 实例。

        参数：
            sd_url (str, 可选): Stable Diffusion 服务的 URL，默认为 ""。
        """
        self.sd_url = SD_URL if not sd_url else sd_url
        self.sd_t2i_url = f"{self.sd_url}/sdapi/v1/txt2img"
        self.payload = payload  # 设定默认请求参数
        logger.info(self.sd_t2i_url)

    def construct_payload(
            self,
            prompt: str,
            negtive_prompt: str = default_negative_prompt,
            width: int = 512,
            height: int = 512,
            sd_model: str = "galaxytimemachinesGTM_photoV20",
    ) -> dict:
        """构造并修改用于图像生成的 API 请求参数。

        参数：
            prompt (str): 用于生成图像的文本提示。
            negtive_prompt (str, 可选): 负面提示词，默认为 default_negative_prompt。
            width (int, 可选): 生成图像的宽度（像素）。
            height (int, 可选): 生成图像的高度（像素）。
            sd_model (str, 可选): 指定使用的 Stable Diffusion 模型，默认为 "galaxytimemachinesGTM_photoV20"。

        返回：
            dict: 更新后的请求参数字典。
        """
        self.payload["prompt"] = prompt
        self.payload["negative_prompt"] = negtive_prompt
        self.payload["width"] = width
        self.payload["height"] = height
        self.payload["override_settings"]["sd_model_checkpoint"] = sd_model
        logger.info(f"调用 SD 请求参数: {self.payload}")
        return self.payload

    def save(self, imgs, save_name=""):
        """保存生成的图像到输出目录。

        参数：
            imgs (list): 生成的图像数据（base64 编码）。
            save_name (str, 可选): 保存的图像文件名，默认为空。
        """
        save_dir = SOURCE_ROOT / SD_OUTPUT_FILE_REPO
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        batch_decode_base64_to_image(imgs, str(save_dir), save_name=save_name)

    def simple_run_t2i(self, payload: dict, auto_save: bool = True):
        """调用 Stable Diffusion API 生成图像。

        参数：
            payload (dict): 请求参数字典。
            auto_save (bool, 可选): 是否自动保存生成的图像，默认为 True。

        返回：
            list: API 返回的生成图像（base64 编码）。
        """
        with requests.Session() as session:
            logger.debug(self.sd_t2i_url)
            rsp = session.post(self.sd_t2i_url, json=payload, timeout=600)

        results = rsp.json()["images"]
        if auto_save:
            save_name = hashlib.sha256(payload["prompt"][:10].encode()).hexdigest()[:6]
            self.save(results, save_name=f"output_{save_name}")
        return results

    async def run_t2i(self, payloads: list):
        """异步调用 Stable Diffusion API 批量生成图像。

        参数：
            payloads (list): 请求参数列表，每个元素是一个字典。
        """
        session = ClientSession()
        for payload_idx, payload in enumerate(payloads):
            results = await self.run(url=self.sd_t2i_url, payload=payload, session=session)
            self.save(results, save_name=f"output_{payload_idx}")
        await session.close()

    async def run(self, url, payload, session):
        """执行 HTTP POST 请求，调用 SD API。

        参数：
            url (str): API URL。
            payload (dict): 请求参数字典。
            session (ClientSession): HTTP 请求会话。

        返回：
            list: 生成的图像数据（base64 编码）。
        """
        async with session.post(url, json=payload, timeout=600) as rsp:
            data = await rsp.read()

        rsp_json = json.loads(data)
        imgs = rsp_json["images"]

        logger.info(f"API 响应 JSON 数据: {rsp_json.keys()}")
        return imgs


def decode_base64_to_image(img, save_name):
    """解码 base64 图像并保存。

    参数：
        img (str): base64 编码的图像数据。
        save_name (str): 保存的文件名。
    """
    image = Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
    pnginfo = PngImagePlugin.PngInfo()
    logger.info(save_name)
    image.save(f"{save_name}.png", pnginfo=pnginfo)
    return pnginfo, image


def batch_decode_base64_to_image(imgs, save_dir="", save_name=""):
    """批量解码 base64 图像并保存。

    参数：
        imgs (list): base64 编码的图像列表。
        save_dir (str, 可选): 保存目录。
        save_name (str, 可选): 保存文件名前缀。
    """
    for idx, _img in enumerate(imgs):
        save_name = join(save_dir, save_name)
        decode_base64_to_image(_img, save_name=save_name)
