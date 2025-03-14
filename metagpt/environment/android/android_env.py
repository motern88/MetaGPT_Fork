#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : MG Android Env

from pydantic import Field

from metagpt.environment.android.android_ext_env import AndroidExtEnv
from metagpt.environment.base_env import Environment


class AndroidEnv(AndroidExtEnv, Environment):
    """为了使用实际的 `reset` 和 `observe` 方法，继承顺序：AndroidExtEnv, Environment"""

    rows: int = Field(default=0, description="截图中网格的行数")
    cols: int = Field(default=0, description="截图中网格的列数")
