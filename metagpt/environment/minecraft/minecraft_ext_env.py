#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : The Minecraft external environment to integrate with Minecraft game
#           refs to `voyager bridge.py`

import json
import time
from typing import Any, Optional

import requests
from pydantic import ConfigDict, Field, model_validator

from metagpt.base.base_env_space import BaseEnvAction, BaseEnvObsParams
from metagpt.environment.base_env import ExtEnv, mark_as_writeable
from metagpt.environment.minecraft.const import (
    MC_CKPT_DIR,
    MC_CORE_INVENTORY_ITEMS,
    MC_CURRICULUM_OB,
    MC_DEFAULT_WARMUP,
    METAGPT_ROOT,
)
from metagpt.environment.minecraft.process_monitor import SubprocessMonitor
from metagpt.logs import logger


class MinecraftExtEnv(ExtEnv):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mc_port: Optional[int] = Field(default=None)
    server_host: str = Field(default="http://127.0.0.1")
    server_port: str = Field(default=3000)
    request_timeout: int = Field(default=600)

    mineflayer: Optional[SubprocessMonitor] = Field(default=None, validate_default=True)

    has_reset: bool = Field(default=False)
    reset_options: Optional[dict] = Field(default=None)
    connected: bool = Field(default=False)
    server_paused: bool = Field(default=False)
    warm_up: dict = Field(default=dict())

    # 重置环境的函数
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pass

    # 观察环境状态的函数
    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        pass

    # 执行动作的函数
    def step(self, action: BaseEnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        pass

    # 获取Minecraft服务器地址
    @property
    def server(self) -> str:
        return f"{self.server_host}:{self.server_port}"

    # 模型初始化后的验证函数
    @model_validator(mode="after")
    def _post_init_ext_env(self):
        # 如果没有 mineflayer，则初始化 mineflayer 进程监控
        if not self.mineflayer:
            self.mineflayer = SubprocessMonitor(
                commands=[
                    "node",
                    METAGPT_ROOT.joinpath("metagpt", "environment", "minecraft", "mineflayer", "index.js"),
                    str(self.server_port),
                ],
                name="mineflayer",
                ready_match=r"Server started on port (\d+)",
            )
        # 如果没有 warm_up 配置，则初始化
        if not self.warm_up:
            warm_up = MC_DEFAULT_WARMUP
            if "optional_inventory_items" in warm_up:
                assert MC_CORE_INVENTORY_ITEMS is not None
                # self.core_inv_items_regex = re.compile(MC_CORE_INVENTORY_ITEMS)
                self.warm_up["optional_inventory_items"] = warm_up["optional_inventory_items"]
            else:
                self.warm_up["optional_inventory_items"] = 0
            for key in MC_CURRICULUM_OB:
                self.warm_up[key] = warm_up.get(key, MC_DEFAULT_WARMUP[key])
            self.warm_up["nearby_blocks"] = 0
            self.warm_up["inventory"] = 0
            self.warm_up["completed_tasks"] = 0
            self.warm_up["failed_tasks"] = 0

        # 初始化检查点子文件夹
        MC_CKPT_DIR.joinpath("curriculum/vectordb").mkdir(parents=True, exist_ok=True)
        MC_CKPT_DIR.joinpath("action").mkdir(exist_ok=True)
        MC_CKPT_DIR.joinpath("skill/code").mkdir(parents=True, exist_ok=True)
        MC_CKPT_DIR.joinpath("skill/description").mkdir(exist_ok=True)
        MC_CKPT_DIR.joinpath("skill/vectordb").mkdir(exist_ok=True)

    # 设置Minecraft的端口号
    def set_mc_port(self, mc_port: int):
        self.mc_port = mc_port

    # 关闭环境并停止进程
    @mark_as_writeable
    def close(self) -> bool:
        self.unpause()  # 确保服务器没有暂停
        if self.connected:
            res = requests.post(f"{self.server}/stop")  # 停止服务器
            if res.status_code == 200:
                self.connected = False  # 设置为未连接状态
        self.mineflayer.stop()  # 停止 mineflayer
        return not self.connected  # 返回是否未连接

    # 检查Minecraft进程是否正常运行
    @mark_as_writeable
    def check_process(self) -> dict:
        retry = 0
        while not self.mineflayer.is_running:
            logger.info("Mineflayer进程已退出，正在重启")
            self.mineflayer.run()  # 重启进程
            if not self.mineflayer.is_running:
                if retry > 3:
                    logger.error("Mineflayer进程启动失败")
                    raise {}
                else:
                    retry += 1
                    continue
            logger.info(self.mineflayer.ready_line)
            # 启动Minecraft服务器
            res = requests.post(
                f"{self.server}/start",
                json=self.reset_options,
                timeout=self.request_timeout,
            )
            if res.status_code != 200:
                self.mineflayer.stop()
                logger.error(f"Minecraft服务器回复错误码 {res.status_code}")
                raise {}
            return res.json()  # 返回服务器响应的数据

    # 重置环境并返回相关数据
    @mark_as_writeable
    def _reset(self, *, seed=None, options=None) -> dict:
        if options is None:
            options = {}
        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            logger.error("仅在模式为硬时才能设置inventory")
            raise {}

        self.reset_options = {
            "port": self.mc_port,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
        }

        self.unpause()  # 确保服务器没有暂停
        self.mineflayer.stop()  # 停止 mineflayer 进程
        time.sleep(1)  # 等待 mineflayer 退出

        returned_data = self.check_process()  # 检查进程并返回数据
        self.has_reset = True  # 标记环境已重置
        self.connected = True  # 标记已连接
        self.reset_options["reset"] = "soft"  # 设置重置模式为软重置
        self.pause()  # 暂停服务器
        return json.loads(returned_data)  # 返回重置后的数据

    # 执行Minecraft的步骤动作
    @mark_as_writeable
    def _step(self, code: str, programs: str = "") -> dict:
        if not self.has_reset:
            raise RuntimeError("环境尚未重置")
        self.check_process()  # 检查进程
        self.unpause()  # 确保服务器没有暂停
        data = {
            "code": code,
            "programs": programs,
        }
        # 向Minecraft服务器发送步进请求
        res = requests.post(f"{self.server}/step", json=data, timeout=self.request_timeout)
        if res.status_code != 200:
            raise RuntimeError("Minecraft服务器步进失败")
        returned_data = res.json()  # 获取返回的数据
        self.pause()  # 暂停服务器
        return json.loads(returned_data)  # 返回步进后的数据

    # 暂停Minecraft服务器
    @mark_as_writeable
    def pause(self) -> bool:
        if self.mineflayer.is_running and not self.server_paused:
            res = requests.post(f"{self.server}/pause")  # 向服务器发送暂停请求
            if res.status_code == 200:
                self.server_paused = True  # 设置服务器为暂停状态
        return self.server_paused  # 返回服务器是否已暂停

    # 取消暂停Minecraft服务器
    @mark_as_writeable
    def unpause(self) -> bool:
        if self.mineflayer.is_running and self.server_paused:
            res = requests.post(f"{self.server}/pause")  # 向服务器发送取消暂停请求
            if res.status_code == 200:
                self.server_paused = False  # 设置服务器为未暂停状态
            else:
                logger.info(f"mineflayer暂停请求结果: {res.json()}")
        return self.server_paused  # 返回服务器是否仍然处于暂停状态
