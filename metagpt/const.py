#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from loguru import logger

import metagpt


# 获取 MetaGPT 安装包的根目录
def get_metagpt_package_root():
    """获取已安装的 MetaGPT 包的根目录"""
    package_root = Path(metagpt.__file__).parent.parent  # 获取 metagpt 模块所在的根目录
    logger.info(f"Package root set to {str(package_root)}")  # 记录日志
    return package_root


# 获取 MetaGPT 项目的根目录
def get_metagpt_root():
    """获取项目根目录"""
    # 首先检查环境变量是否指定了项目根目录
    project_root_env = os.getenv("METAGPT_PROJECT_ROOT")
    if project_root_env:
        project_root = Path(project_root_env)  # 从环境变量获取项目根目录
        logger.info(f"PROJECT_ROOT set from environment variable to {str(project_root)}")  # 记录日志
    else:
        # 如果环境变量未指定项目根目录，则回退到安装包根目录
        project_root = get_metagpt_package_root()

        # 通过检查 .git、.project_root 或 .gitignore 文件来判断是否为项目根目录
        for i in (".git", ".project_root", ".gitignore"):
            if (project_root / i).exists():
                break  # 如果找到这些文件之一，就认定该目录为项目根目录
        else:
            # 如果未找到任何标志文件，则使用当前工作目录作为项目根目录
            project_root = Path.cwd()

    return project_root


# METAGPT 项目根目录及相关变量
CONFIG_ROOT = Path.home() / ".metagpt"  # 配置文件存储在用户的 home 目录下的 .metagpt 目录
METAGPT_ROOT = get_metagpt_root()  # 依赖于环境变量 METAGPT_PROJECT_ROOT 或自动检测的项目根目录
DEFAULT_WORKSPACE_ROOT = METAGPT_ROOT / "workspace"  # 默认的工作区目录

# 示例路径
EXAMPLE_PATH = METAGPT_ROOT / "examples"  # 示例代码目录
EXAMPLE_DATA_PATH = EXAMPLE_PATH / "data"  # 示例数据目录
DATA_PATH = METAGPT_ROOT / "data"  # 数据目录
DABENCH_PATH = EXAMPLE_PATH / "di/InfiAgent-DABench/data"  # DABench 示例数据目录
EXAMPLE_BENCHMARK_PATH = EXAMPLE_PATH / "data/rag_bm"  # RAG 基准测试数据目录
TEST_DATA_PATH = METAGPT_ROOT / "tests/data"  # 测试数据目录
RESEARCH_PATH = DATA_PATH / "research"  # 研究数据目录
TUTORIAL_PATH = DATA_PATH / "tutorial_docx"  # 教程文档目录
INVOICE_OCR_TABLE_PATH = DATA_PATH / "invoice_table"  # OCR 发票表格数据目录

# 单元测试相关路径
UT_PATH = DATA_PATH / "ut"  # 单元测试数据目录
SWAGGER_PATH = UT_PATH / "files/api/"  # Swagger API 文件路径
UT_PY_PATH = UT_PATH / "files/ut/"  # 单元测试 Python 文件路径
API_QUESTIONS_PATH = UT_PATH / "files/question/"  # API 相关问题文件路径

# 序列化/反序列化存储路径
SERDESER_PATH = DEFAULT_WORKSPACE_ROOT / "storage"  # TODO: 将 `storage` 存储在各个生成的项目目录下

# 临时文件目录
TMP = METAGPT_ROOT / "tmp"

# 源代码路径
SOURCE_ROOT = METAGPT_ROOT / "metagpt"  # 源代码根目录
PROMPT_PATH = SOURCE_ROOT / "prompts"  # 提示词文件目录
SKILL_DIRECTORY = SOURCE_ROOT / "skills"  # 技能模块目录
TOOL_SCHEMA_PATH = METAGPT_ROOT / "metagpt/tools/schemas"  # 工具模式定义目录
TOOL_LIBS_PATH = METAGPT_ROOT / "metagpt/tools/libs"  # 工具库目录

# 模板路径
TEMPLATE_FOLDER_PATH = METAGPT_ROOT / "template"  # 模板目录
VUE_TEMPLATE_PATH = TEMPLATE_FOLDER_PATH / "vue_template"  # Vue 模板路径
REACT_TEMPLATE_PATH = TEMPLATE_FOLDER_PATH / "react_template"  # React 模板路径

# 真实常量定义

MEM_TTL = 24 * 30 * 3600  # 内存缓存的存活时间 (秒)，约为 30 天

# 消息路由相关常量
MESSAGE_ROUTE_FROM = "sent_from"  # 消息发送方
MESSAGE_ROUTE_TO = "send_to"  # 消息接收方
MESSAGE_ROUTE_CAUSE_BY = "cause_by"  # 消息由谁引起
MESSAGE_META_ROLE = "role"  # 消息中的角色元信息
MESSAGE_ROUTE_TO_ALL = "<all>"  # 发送到所有人
MESSAGE_ROUTE_TO_NONE = "<none>"  # 不发送给任何人
MESSAGE_ROUTE_TO_SELF = "<self>"  # 发送给自己（用于替换 `ActionOutput`）

# 文件名定义
REQUIREMENT_FILENAME = "requirement.txt"  # 需求文档
BUGFIX_FILENAME = "bugfix.txt"  # Bug 修复记录
PACKAGE_REQUIREMENTS_FILENAME = "requirements.txt"  # 依赖包需求文件

# 文档存储路径
DOCS_FILE_REPO = "docs"  # 文档根目录
PRDS_FILE_REPO = "docs/prd"  # 产品需求文档目录
SYSTEM_DESIGN_FILE_REPO = "docs/system_design"  # 系统设计文档目录
TASK_FILE_REPO = "docs/task"  # 任务文档目录
CODE_PLAN_AND_CHANGE_FILE_REPO = "docs/code_plan_and_change"  # 代码规划和变更文档目录
COMPETITIVE_ANALYSIS_FILE_REPO = "resources/competitive_analysis"  # 竞争分析文档目录
DATA_API_DESIGN_FILE_REPO = "resources/data_api_design"  # 数据 API 设计文档目录
SEQ_FLOW_FILE_REPO = "resources/seq_flow"  # 时序流程文档目录
SYSTEM_DESIGN_PDF_FILE_REPO = "resources/system_design"  # 系统设计 PDF 目录
PRD_PDF_FILE_REPO = "resources/prd"  # 产品需求 PDF 目录
TASK_PDF_FILE_REPO = "resources/api_spec_and_task"  # API 规格与任务 PDF 目录
CODE_PLAN_AND_CHANGE_PDF_FILE_REPO = "resources/code_plan_and_change"  # 代码规划和变更 PDF 目录
TEST_CODES_FILE_REPO = "tests"  # 测试代码目录
TEST_OUTPUTS_FILE_REPO = "test_outputs"  # 测试输出目录
CODE_SUMMARIES_FILE_REPO = "docs/code_summary"  # 代码总结文档目录
CODE_SUMMARIES_PDF_FILE_REPO = "resources/code_summary"  # 代码总结 PDF 目录
RESOURCES_FILE_REPO = "resources"  # 资源文件目录
SD_OUTPUT_FILE_REPO = DEFAULT_WORKSPACE_ROOT  # 系统设计输出文件目录
GRAPH_REPO_FILE_REPO = "docs/graph_repo"  # 图数据库文档目录
VISUAL_GRAPH_REPO_FILE_REPO = "resources/graph_db"  # 可视化图数据库目录
CLASS_VIEW_FILE_REPO = "docs/class_view"  # 类视图文档目录

# API 相关地址
YAPI_URL = "http://yapi.deepwisdomai.com/"  # YAPI 接口管理地址
SD_URL = "http://172.31.0.51:49094"  # SD 相关服务地址

# 默认配置
DEFAULT_LANGUAGE = "English"  # 默认语言
DEFAULT_MAX_TOKENS = 1500  # 生成文本的最大 token 数
COMMAND_TOKENS = 500  # 命令相关的 token 预算
BRAIN_MEMORY = "BRAIN_MEMORY"  # 认知系统中的记忆存储关键字
SKILL_PATH = "SKILL_PATH"  # 技能存储路径
SERPER_API_KEY = "SERPER_API_KEY"  # SERPER API 密钥
DEFAULT_TOKEN_SIZE = 500  # 默认的 token 长度

# 数据格式
BASE64_FORMAT = "base64"  # Base64 编码格式

# REDIS 相关
REDIS_KEY = "REDIS_KEY"  # Redis 关键字

# 消息 ID
IGNORED_MESSAGE_ID = "0"  # 被忽略的消息 ID

# 类关系定义
GENERALIZATION = "Generalize"  # 泛化关系
COMPOSITION = "Composite"  # 组合关系
AGGREGATION = "Aggregate"  # 聚合关系

# 超时时间
USE_CONFIG_TIMEOUT = 0  # 使用 llm.timeout 配置
LLM_API_TIMEOUT = 300  # LLM API 请求超时时间（秒）

# 助理别名
ASSISTANT_ALIAS = "response"  # 助理的默认别名

# Markdown 相关
MARKDOWN_TITLE_PREFIX = "## "  # Markdown 标题前缀

# 报告系统
METAGPT_REPORTER_DEFAULT_URL = os.environ.get("METAGPT_REPORTER_URL", "")  # MetaGPT 报告系统 URL

# 元数据定义
AGENT = "agent"  # 代理对象
IMAGES = "images"  # 图像数据

# SWE 代理（软件工程代理）
SWE_SETUP_PATH = get_metagpt_package_root() / "metagpt/tools/swe_agent_commands/setup_default.sh"  # SWE 代理的默认安装脚本路径

# 经验池
EXPERIENCE_MASK = "<experience>"  # 经验屏蔽标记

# 团队领导名称
TEAMLEADER_NAME = "Mike"  # 团队领导的默认名称

# Token 数量限制
DEFAULT_MIN_TOKEN_COUNT = 10000  # 最小 token 数
DEFAULT_MAX_TOKEN_COUNT = 100000000  # 最大 token 数
