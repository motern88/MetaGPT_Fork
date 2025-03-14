#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : The Android external environment to integrate with Android apps
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import clip
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from pydantic import Field

from metagpt.const import DEFAULT_WORKSPACE_ROOT
from metagpt.environment.android.const import ADB_EXEC_FAIL
from metagpt.environment.android.env_space import (
    EnvAction,
    EnvActionType,
    EnvObsParams,
    EnvObsType,
    EnvObsValType,
)
from metagpt.environment.android.text_icon_localization import (
    clip_for_icon,
    crop_for_clip,
    det,
    load_model,
    ocr,
)
from metagpt.environment.base_env import ExtEnv, mark_as_readable, mark_as_writeable
from metagpt.logs import logger
from metagpt.utils.common import download_model


def load_cv_model(device: str = "cpu") -> any:
    ocr_detection = pipeline(Tasks.ocr_detection, model="damo/cv_resnet18_ocr-detection-line-level_damo")
    ocr_recognition = pipeline(Tasks.ocr_recognition, model="damo/cv_convnextTiny_ocr-recognition-document_damo")
    file_url = "https://huggingface.co/ShilongLiu/GroundingDINO/blob/main/groundingdino_swint_ogc.pth"
    target_folder = Path(f"{DEFAULT_WORKSPACE_ROOT}/weights")
    file_path = download_model(file_url, target_folder)
    groundingdino_model = load_model(file_path, device=device).eval()
    return ocr_detection, ocr_recognition, groundingdino_model


class AndroidExtEnv(ExtEnv):
    device_id: Optional[str] = Field(default=None)
    screenshot_dir: Optional[Path] = Field(default=None)
    xml_dir: Optional[Path] = Field(default=None)
    width: int = Field(default=720, description="设备屏幕的宽度")
    height: int = Field(default=1080, description="设备屏幕的高度")
    ocr_detection: any = Field(default=None, description="OCR 检测模型")
    ocr_recognition: any = Field(default=None, description="OCR 识别模型")
    groundingdino_model: any = Field(default=None, description="Clip GroundingDINO 模型")

    def __init__(self, **data: Any):
        super().__init__(**data)
        device_id = data.get("device_id")
        self.ocr_detection, self.ocr_recognition, self.groundingdino_model = load_cv_model()
        if device_id:
            devices = self.list_devices()
            if device_id not in devices:
                raise RuntimeError(f"device-id: {device_id} 未找到")
            (width, height) = self.device_shape
            self.width = data.get("width", width)
            self.height = data.get("height", height)
            self.create_device_path(self.screenshot_dir)
            self.create_device_path(self.xml_dir)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        obs = self._get_obs()

        return obs, {}

    def _get_obs(self) -> dict[str, EnvObsValType]:
        pass

    def observe(self, obs_params: Optional[EnvObsParams] = None) -> Any:
        obs_type = obs_params.obs_type if obs_params else EnvObsType.NONE
        if obs_type == EnvObsType.NONE:
            pass
        elif obs_type == EnvObsType.GET_SCREENSHOT:
            obs = self.get_screenshot(ss_name=obs_params.ss_name, local_save_dir=obs_params.local_save_dir)
        elif obs_type == EnvObsType.GET_XML:
            obs = self.get_xml(xml_name=obs_params.xml_name, local_save_dir=obs_params.local_save_dir)
        return obs

    def step(self, action: EnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        res = self._execute_env_action(action)

        obs = {}

        ret = (obs, 1.0, False, False, {"res": res})
        return ret

    def _execute_env_action(self, action: EnvAction):
        action_type = action.action_type
        res = None
        if action_type == EnvActionType.NONE:
            pass
        elif action_type == EnvActionType.SYSTEM_BACK:
            res = self.system_back()
        elif action_type == EnvActionType.SYSTEM_TAP:
            res = self.system_tap(x=action.coord[0], y=action.coord[1])
        elif action_type == EnvActionType.USER_INPUT:
            res = self.user_input(input_txt=action.input_txt)
        elif action_type == EnvActionType.USER_LONGPRESS:
            res = self.user_longpress(x=action.coord[0], y=action.coord[1])
        elif action_type == EnvActionType.USER_SWIPE:
            res = self.user_swipe(x=action.coord[0], y=action.coord[1], orient=action.orient, dist=action.dist)
        elif action_type == EnvActionType.USER_SWIPE_TO:
            res = self.user_swipe_to(start=action.coord, end=action.tgt_coord)
        return res

    @property
    def adb_prefix_si(self):
        """带有 `device_id` 和 `shell input` 的 adb 命令前缀"""
        return f"adb -s {self.device_id} shell input "

    @property
    def adb_prefix_shell(self):
        """带有 `device_id` 和 `shell` 的 adb 命令前缀"""
        return f"adb -s {self.device_id} shell "

    @property
    def adb_prefix(self):
        """带有 `device_id` 的 adb 命令前缀"""
        return f"adb -s {self.device_id} "

    def execute_adb_with_cmd(self, adb_cmd: str) -> str:
        adb_cmd = adb_cmd.replace("\\", "/")
        res = subprocess.run(adb_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        exec_res = ADB_EXEC_FAIL
        if not res.returncode:
            exec_res = res.stdout.strip()
        return exec_res

    def create_device_path(self, folder_path: Path):
        adb_cmd = f"{self.adb_prefix_shell} mkdir {folder_path} -p"
        res = self.execute_adb_with_cmd(adb_cmd)
        if res == ADB_EXEC_FAIL:
            raise RuntimeError(f"创建设备路径失败: {folder_path}")

    @property
    def device_shape(self) -> tuple[int, int]:
        adb_cmd = f"{self.adb_prefix_shell} wm size"
        shape = (0, 0)
        shape_res = self.execute_adb_with_cmd(adb_cmd)
        if shape_res != ADB_EXEC_FAIL:
            shape = tuple(map(int, shape_res.split(": ")[1].split("x")))
        return shape

    def list_devices(self):
        adb_cmd = "adb devices"
        res = self.execute_adb_with_cmd(adb_cmd)
        devices = []
        if res != ADB_EXEC_FAIL:
            devices = res.split("\n")[1:]
            devices = [device.split()[0] for device in devices]
        return devices

    @mark_as_readable
    def get_screenshot(self, ss_name: str, local_save_dir: Path) -> Path:
        """
        ss_name: 截图文件名称
        local_save_dir: 本地目录用于存储虚拟机中的图片
        """
        assert self.screenshot_dir
        ss_remote_path = Path(self.screenshot_dir).joinpath(f"{ss_name}.png")
        ss_cmd = f"{self.adb_prefix_shell} screencap -p {ss_remote_path}"
        ss_res = self.execute_adb_with_cmd(ss_cmd)
        time.sleep(0.1)
        res = ADB_EXEC_FAIL
        if ss_res != ADB_EXEC_FAIL:
            ss_local_path = Path(local_save_dir).joinpath(f"{ss_name}.png")
            pull_cmd = f"{self.adb_prefix} pull {ss_remote_path} {ss_local_path}"
            pull_res = self.execute_adb_with_cmd(pull_cmd)
            time.sleep(0.1)
            if pull_res != ADB_EXEC_FAIL:
                res = ss_local_path
        else:
            ss_cmd = f"{self.adb_prefix_shell} rm /sdcard/{ss_name}.png"
            ss_res = self.execute_adb_with_cmd(ss_cmd)
            time.sleep(0.1)
            ss_cmd = f"{self.adb_prefix_shell} screencap -p /sdcard/{ss_name}.png"
            ss_res = self.execute_adb_with_cmd(ss_cmd)
            time.sleep(0.1)
            ss_cmd = f"{self.adb_prefix} pull /sdcard/{ss_name}.png {self.screenshot_dir}"
            ss_res = self.execute_adb_with_cmd(ss_cmd)
            image_path = Path(f"{self.screenshot_dir}/{ss_name}.png")
            res = image_path
        return Path(res)

    @mark_as_readable
    def get_xml(self, xml_name: str, local_save_dir: Path) -> Path:
        """
        从设备获取XML文件并保存到本地
        :param xml_name: XML文件名
        :param local_save_dir: 本地保存路径
        :return: 本地保存的XML文件路径
        """
        xml_remote_path = Path(self.xml_dir).joinpath(f"{xml_name}.xml")
        dump_cmd = f"{self.adb_prefix_shell} uiautomator dump {xml_remote_path}"
        xml_res = self.execute_adb_with_cmd(dump_cmd)

        res = ADB_EXEC_FAIL
        if xml_res != ADB_EXEC_FAIL:
            xml_local_path = Path(local_save_dir).joinpath(f"{xml_name}.xml")
            pull_cmd = f"{self.adb_prefix} pull {xml_remote_path} {xml_local_path}"
            pull_res = self.execute_adb_with_cmd(pull_cmd)
            if pull_res != ADB_EXEC_FAIL:
                res = xml_local_path
        return Path(res)

    @mark_as_writeable
    def system_back(self) -> str:
        """
        执行设备返回操作
        :return: 执行结果
        """
        adb_cmd = f"{self.adb_prefix_si} keyevent KEYCODE_BACK"
        back_res = self.execute_adb_with_cmd(adb_cmd)
        return back_res

    @mark_as_writeable
    def system_tap(self, x: int, y: int) -> str:
        """
        在指定位置模拟点击操作
        :param x: 点击的x坐标
        :param y: 点击的y坐标
        :return: 执行结果
        """
        adb_cmd = f"{self.adb_prefix_si} tap {x} {y}"
        tap_res = self.execute_adb_with_cmd(adb_cmd)
        return tap_res

    @mark_as_writeable
    def user_input(self, input_txt: str) -> str:
        """
        向设备输入文本
        :param input_txt: 输入的文本
        :return: 执行结果
        """
        input_txt = input_txt.replace(" ", "%s").replace("'", "")
        adb_cmd = f"{self.adb_prefix_si} text {input_txt}"
        input_res = self.execute_adb_with_cmd(adb_cmd)
        return input_res

    @mark_as_writeable
    def user_longpress(self, x: int, y: int, duration: int = 500) -> str:
        """
        在指定位置执行长按操作
        :param x: 长按的x坐标
        :param y: 长按的y坐标
        :param duration: 按住的时间（毫秒）
        :return: 执行结果
        """
        adb_cmd = f"{self.adb_prefix_si} swipe {x} {y} {x} {y} {duration}"
        press_res = self.execute_adb_with_cmd(adb_cmd)
        return press_res

    @mark_as_writeable
    def user_swipe(self, x: int, y: int, orient: str = "up", dist: str = "medium", if_quick: bool = False) -> str:
        """
        在设备上执行滑动操作
        :param x: 起始点的x坐标
        :param y: 起始点的y坐标
        :param orient: 滑动方向（up、down、left、right）
        :param dist: 滑动距离（long、medium、short）
        :param if_quick: 是否快速滑动
        :return: 执行结果
        """
        dist_unit = int(self.width / 10)
        if dist == "long":
            dist_unit *= 3
        elif dist == "medium":
            dist_unit *= 2

        if orient == "up":
            offset = 0, -2 * dist_unit
        elif orient == "down":
            offset = 0, 2 * dist_unit
        elif orient == "left":
            offset = -1 * dist_unit, 0
        elif orient == "right":
            offset = dist_unit, 0
        else:
            return ADB_EXEC_FAIL

        duration = 100 if if_quick else 400
        adb_cmd = f"{self.adb_prefix_si} swipe {x} {y} {x + offset[0]} {y + offset[1]} {duration}"
        swipe_res = self.execute_adb_with_cmd(adb_cmd)
        return swipe_res

    @mark_as_writeable
    def user_swipe_to(self, start: tuple[int, int], end: tuple[int, int], duration: int = 400) -> str:
        """
        执行滑动操作，从指定起始点滑动到指定结束点
        :param start: 起始点坐标
        :param end: 结束点坐标
        :param duration: 滑动时长（毫秒）
        :return: 执行结果
        """
        adb_cmd = f"{self.adb_prefix_si} swipe {start[0]} {start[1]} {end[0]} {end[1]} {duration}"
        swipe_res = self.execute_adb_with_cmd(adb_cmd)
        return swipe_res

    @mark_as_writeable
    def user_exit(self) -> str:
        """
        执行设备退出操作，返回桌面
        :return: 执行结果
        """
        adb_cmd = f"{self.adb_prefix_shell} am start -a android.intent.action.MAIN -c android.intent.category.HOME"
        exit_res = self.execute_adb_with_cmd(adb_cmd)
        return exit_res

    def _ocr_text(self, text: str) -> list:
        """
        使用OCR技术识别指定文本并返回结果
        :param text: 需要识别的文本
        :return: 识别结果，包括坐标和图像信息
        """
        image = self.get_screenshot("screenshot", self.screenshot_dir)
        iw, ih = Image.open(image).size
        x, y = self.device_shape
        if iw > ih:
            x, y = y, x
            iw, ih = ih, iw
        in_coordinate, out_coordinate = ocr(image, text, self.ocr_detection, self.ocr_recognition, iw, ih)
        output_list = [in_coordinate, out_coordinate, x, y, iw, ih, image]
        return output_list

    @mark_as_writeable
    def user_open_app(self, app_name: str) -> str:
        """
        打开指定应用
        :param app_name: 应用名称
        :return: 执行结果
        """
        ocr_result = self._ocr_text(app_name)
        in_coordinate, _, x, y, iw, ih = (
            ocr_result[0],
            ocr_result[1],
            ocr_result[2],
            ocr_result[3],
            ocr_result[4],
            ocr_result[5],
        )
        if len(in_coordinate) == 0:
            logger.info(f"No App named {app_name}.")
            return "no app here"
        else:
            tap_coordinate = [
                (in_coordinate[0][0] + in_coordinate[0][2]) / 2,
                (in_coordinate[0][1] + in_coordinate[0][3]) / 2,
            ]
            tap_coordinate = [round(tap_coordinate[0] / iw, 2), round(tap_coordinate[1] / ih, 2)]
            return self.system_tap(tap_coordinate[0] * x, (tap_coordinate[1] - round(50 / y, 2)) * y)

    @mark_as_writeable
    def user_click_text(self, text: str) -> str:
        """
        点击指定文本
        :param text: 需要点击的文本
        :return: 执行结果
        """
        ocr_result = self._ocr_text(text)
        in_coordinate, out_coordinate, x, y, iw, ih, _ = (
            ocr_result[0],
            ocr_result[1],
            ocr_result[2],
            ocr_result[3],
            ocr_result[4],
            ocr_result[5],
            ocr_result[6],
        )
        if len(out_coordinate) == 0:
            logger.info(
                f'Failed to execute action click text ({text}). The text "{text}" is not detected in the screenshot.'
            )
        elif len(out_coordinate) == 1:
            tap_coordinate = [
                (in_coordinate[0][0] + in_coordinate[0][2]) / 2,
                (in_coordinate[0][1] + in_coordinate[0][3]) / 2,
            ]
            tap_coordinate = [round(tap_coordinate[0] / iw, 2), round(tap_coordinate[1] / ih, 2)]
            return self.system_tap(tap_coordinate[0] * x, tap_coordinate[1] * y)
        else:
            logger.info(
                f'Failed to execute action click text ({text}). There are too many text "{text}" in the screenshot.'
            )

    @mark_as_writeable
    def user_stop(self):
        """
        停止执行
        """
        logger.info("Successful execution of tasks")

    @mark_as_writeable
    def user_click_icon(self, icon_shape_color: str) -> str:
        """
        点击指定的图标（根据形状和颜色）
        :param icon_shape_color: 图标的形状和颜色描述
        :return: 执行结果
        """
        screenshot_path = self.get_screenshot("screenshot", self.screenshot_dir)
        image = screenshot_path
        iw, ih = Image.open(image).size
        x, y = self.device_shape
        if iw > ih:
            x, y = y, x
            iw, ih = ih, iw
        in_coordinate, out_coordinate = det(image, "icon", self.groundingdino_model)  # 检测图标
        if len(out_coordinate) == 1:  # 只有一个图标
            tap_coordinate = [
                (in_coordinate[0][0] + in_coordinate[0][2]) / 2,
                (in_coordinate[0][1] + in_coordinate[0][3]) / 2,
            ]
            tap_coordinate = [round(tap_coordinate[0] / iw, 2), round(tap_coordinate[1] / ih, 2)]
            return self.system_tap(tap_coordinate[0] * x, tap_coordinate[1] * y)
        else:
            temp_file = Path(f"{DEFAULT_WORKSPACE_ROOT}/temp")
            temp_file.mkdir(parents=True, exist_ok=True)
            hash_table, clip_filter = [], []
            for i, (td, box) in enumerate(zip(in_coordinate, out_coordinate)):
                if crop_for_clip(image, td, i, temp_file):
                    hash_table.append(td)
                    crop_image = f"{i}.png"
                    clip_filter.append(temp_file.joinpath(crop_image))
            clip_model, clip_preprocess = clip.load("ViT-B/32")  # FIXME: device=device
            clip_filter = clip_for_icon(clip_model, clip_preprocess, clip_filter, icon_shape_color)
            final_box = hash_table[clip_filter]
            tap_coordinate = [(final_box[0] + final_box[2]) / 2, (final_box[1] + final_box[3]) / 2]
            tap_coordinate = [round(tap_coordinate[0] / iw, 2), round(tap_coordinate[1] / ih, 2)]
            print(tap_coordinate[0] * x, tap_coordinate[1] * y)
            return self.system_tap(tap_coordinate[0] * x, tap_coordinate[1] * y)
