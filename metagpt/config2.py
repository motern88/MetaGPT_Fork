#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 01:25
@Author  : alexanderwu
@File    : config2.py
"""
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from metagpt.configs.browser_config import BrowserConfig
from metagpt.configs.embedding_config import EmbeddingConfig
from metagpt.configs.exp_pool_config import ExperiencePoolConfig
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.configs.mermaid_config import MermaidConfig
from metagpt.configs.omniparse_config import OmniParseConfig
from metagpt.configs.redis_config import RedisConfig
from metagpt.configs.role_custom_config import RoleCustomConfig
from metagpt.configs.role_zero_config import RoleZeroConfig
from metagpt.configs.s3_config import S3Config
from metagpt.configs.search_config import SearchConfig
from metagpt.configs.workspace_config import WorkspaceConfig
from metagpt.const import CONFIG_ROOT, METAGPT_ROOT
from metagpt.utils.yaml_model import YamlModel

# CLI 参数模型
class CLIParams(BaseModel):
    """CLI parameters"""

    project_path: str = ""  # 项目路径
    project_name: str = ""  # 项目名称
    inc: bool = False  # 是否增量更新
    reqa_file: str = ""  # 需求文件路径
    max_auto_summarize_code: int = 0  # 代码自动总结的最大限制
    git_reinit: bool = False  # 是否重新初始化 Git 仓库

    @model_validator(mode="after")
    def check_project_path(self):
        """Check project_path and project_name"""
        if self.project_path:
            self.inc = True  # 如果提供了项目路径，则启用增量更新
            # 若未提供项目名称，则使用项目路径的名称
            self.project_name = self.project_name or Path(self.project_path).name
        return self

# 配置类，继承 CLIParams 和 YamlModel
class Config(CLIParams, YamlModel):
    """Configurations for MetaGPT"""

    # Key Parameters
    llm: LLMConfig

    # RAG Embedding
    embedding: EmbeddingConfig = EmbeddingConfig()

    # omniparse 解析配置
    omniparse: OmniParseConfig = OmniParseConfig()

    # Global Proxy. Will be used if llm.proxy is not set / 全局代理（如果 llm.proxy 未设置，则使用该代理）
    proxy: str = ""

    # Tool Parameters
    search: SearchConfig = SearchConfig()  # 搜索配置
    enable_search: bool = False  # 是否启用搜索功能
    browser: BrowserConfig = BrowserConfig()  # 浏览器配置
    mermaid: MermaidConfig = MermaidConfig()  # Mermaid 代码渲染配置

    # Storage Parameters
    s3: Optional[S3Config] = None  # S3 存储配置
    redis: Optional[RedisConfig] = None  # Redis 配置

    # Misc Parameters  其他参数
    repair_llm_output: bool = False  # 是否修复 LLM 的输出
    prompt_schema: Literal["json", "markdown", "raw"] = "json"  # 提示词模式
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)  # 工作区配置
    enable_longterm_memory: bool = False  # 是否启用长期记忆
    code_validate_k_times: int = 2  # 代码验证的次数

    # Experience Pool Parameters  经验池参数
    exp_pool: ExperiencePoolConfig = Field(default_factory=ExperiencePoolConfig)  # 经验池配置

    # Will be removed in the future  未来可能会移除的参数
    metagpt_tti_url: str = ""  # MetaGPT 文字转图像（TTI）服务 URL
    language: str = "English"  # 语言
    redis_key: str = "placeholder"  # Redis 存储键
    iflytek_app_id: str = ""  # 科大讯飞 App ID
    iflytek_api_secret: str = ""  # 科大讯飞 API 密钥
    iflytek_api_key: str = ""  # 科大讯飞 API Key
    azure_tts_subscription_key: str = ""  # Azure 语音合成订阅 Key
    azure_tts_region: str = ""  # Azure 语音合成区域
    _extra: dict = dict()  # extra config dict  额外的配置信息

    # Role's custom configuration  角色的自定义配置
    roles: Optional[List[RoleCustomConfig]] = None

    # RoleZero's configuration
    role_zero: RoleZeroConfig = Field(default_factory=RoleZeroConfig)

    # 从用户主目录加载配置
    @classmethod
    def from_home(cls, path):
        """从 ~/.metagpt/config2.yaml 加载配置"""
        pathname = CONFIG_ROOT / path
        if not pathname.exists():
            return None
        return Config.from_yaml_file(pathname)

    # 加载默认配置
    @classmethod
    def default(cls, reload: bool = False, **kwargs) -> "Config":
        """加载默认配置
        - 优先级：环境变量 < 默认配置路径
        - 在默认配置路径中，后面的配置会覆盖前面的配置
        """
        default_config_paths = (
            METAGPT_ROOT / "config/config2.yaml",  # MetaGPT 根目录的默认配置
            CONFIG_ROOT / "config2.yaml",  # 用户配置目录的默认配置
        )
        if reload or default_config_paths not in _CONFIG_CACHE:
            dicts = [
                dict(os.environ),  # 读取环境变量
                *(Config.read_yaml(path) for path in default_config_paths),  # 读取默认配置文件
                kwargs  # 额外传入的参数
            ]
            final = merge_dict(dicts)  # 合并多个字典
            _CONFIG_CACHE[default_config_paths] = Config(**final)  # 存入缓存
        return _CONFIG_CACHE[default_config_paths]

    # 从 LLM 配置创建 Config 实例
    @classmethod
    def from_llm_config(cls, llm_config: dict):
        """根据用户提供的 LLM 配置创建 Config 实例
        示例：
        llm_config = {"api_type": "xxx", "api_key": "xxx", "model": "xxx"}
        gpt4 = Config.from_llm_config(llm_config)

        A = Role(name="A", profile="Democratic candidate", goal="Win the election",
                 actions=[a1], watch=[a2], config=gpt4)
        """
        llm_config = LLMConfig.model_validate(llm_config)  # 验证 LLM 配置是否合法
        dicts = [dict(os.environ)]  # 读取环境变量
        dicts += [{"llm": llm_config}]  # 添加 LLM 配置
        final = merge_dict(dicts)  # 合并多个字典
        return Config(**final)  # 返回 Config 实例

    # 通过命令行参数更新配置
    def update_via_cli(self, project_path, project_name, inc, reqa_file, max_auto_summarize_code):
        """通过命令行更新配置"""

        # 在 PrepareDocuments 操作中使用，参考 RFC 135 的 2.2.3.5.1 小节
        if project_path:
            inc = True  # 如果提供了项目路径，则启用增量更新
            project_name = project_name or Path(project_path).name  # 如果未提供项目名称，则使用路径名称
        self.project_path = project_path
        self.project_name = project_name
        self.inc = inc
        self.reqa_file = reqa_file
        self.max_auto_summarize_code = max_auto_summarize_code

    # 额外配置的 getter 方法
    @property
    def extra(self):
        return self._extra

    # 额外配置的 setter 方法
    @extra.setter
    def extra(self, value: dict):
        self._extra = value

    # 获取 OpenAI LLM 配置
    def get_openai_llm(self) -> Optional[LLMConfig]:
        """获取 OpenAI 的 LLM 配置，如果不是 OpenAI 则返回 None"""
        if self.llm.api_type == LLMType.OPENAI:
            return self.llm
        return None

    # 获取 Azure LLM 配置
    def get_azure_llm(self) -> Optional[LLMConfig]:
        """获取 Azure 的 LLM 配置，如果不是 Azure 则返回 None"""
        if self.llm.api_type == LLMType.AZURE:
            return self.llm
        return None


# 合并多个字典，以后者覆盖前者
def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """合并多个字典，后面的字典会覆盖前面的值"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

# 配置缓存
_CONFIG_CACHE = {}

# 加载默认配置
config = Config.default()
