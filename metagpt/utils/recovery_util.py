# -*- coding: utf-8 -*-
# @Date    : 12/20/2023 11:07 AM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import json
from datetime import datetime
from pathlib import Path

import nbformat

from metagpt.const import DATA_PATH
from metagpt.roles.role import Role
from metagpt.utils.common import read_json_file
from metagpt.utils.save_code import save_code_file


def load_history(save_dir: str = ""):
    """
    从指定的保存目录加载计划和代码执行历史。

    参数:
        save_dir (str): 用于加载历史记录的目录路径。

    返回:
        Tuple: 包含加载的计划和 notebook 的元组。
    """

    # 构造计划文件和 notebook 文件的路径
    plan_path = Path(save_dir) / "plan.json"
    nb_path = Path(save_dir) / "history_nb" / "code.ipynb"

    # 读取计划文件和 notebook 文件
    plan = read_json_file(plan_path)
    nb = nbformat.read(open(nb_path, "r", encoding="utf-8"), as_version=nbformat.NO_CONVERT)

    return plan, nb


def save_history(role: Role, save_dir: str = ""):
    """
    将计划和代码执行历史保存到指定目录。

    参数:
        role (Role): 包含计划和执行代码属性的角色对象。
        save_dir (str): 用于保存历史记录的目录路径。

    返回:
        Path: 保存历史记录目录的路径。
    """
    # 获取当前时间并格式化为字符串
    record_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 构造保存历史记录的路径
    save_path = DATA_PATH / "output" / f"{record_time}"

    # 如果目录存在则覆盖，创建新的目录
    save_path.mkdir(parents=True, exist_ok=True)

    # 获取角色对象的计划
    plan = role.planner.plan.dict()

    # 保存计划为 JSON 格式
    with open(save_path / "plan.json", "w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, indent=4, ensure_ascii=False)

    # 保存代码文件
    save_code_file(name=Path(record_time), code_context=role.execute_code.nb, file_format="ipynb")

    return save_path
