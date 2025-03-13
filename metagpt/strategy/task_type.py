from enum import Enum

from pydantic import BaseModel

from metagpt.prompts.task_type import (
    DATA_PREPROCESS_PROMPT,
    EDA_PROMPT,
    FEATURE_ENGINEERING_PROMPT,
    IMAGE2WEBPAGE_PROMPT,
    MODEL_EVALUATE_PROMPT,
    MODEL_TRAIN_PROMPT,
    WEB_SCRAPING_PROMPT,
)


class TaskTypeDef(BaseModel):
    # 任务类型的定义类，包含任务名称、描述和指导信息
    name: str  # 任务名称
    desc: str = ""  # 任务描述，默认值为空字符串
    guidance: str = ""  # 任务指导信息，默认值为空字符串


class TaskType(Enum):
    """通过识别特定类型的任务，我们可以注入人为先验知识（指导信息）来帮助任务解决"""

    # 任务类型：EDA（探索性数据分析）
    EDA = TaskTypeDef(
        name="eda",
        desc="用于执行探索性数据分析",
        guidance=EDA_PROMPT,
    )

    # 任务类型：数据预处理
    DATA_PREPROCESS = TaskTypeDef(
        name="data preprocessing",
        desc="仅用于在数据分析或机器学习任务中进行数据预处理，"
             "一般的数据操作不属于此类型",
        guidance=DATA_PREPROCESS_PROMPT,
    )

    # 任务类型：特征工程
    FEATURE_ENGINEERING = TaskTypeDef(
        name="feature engineering",
        desc="仅用于为输入数据创建新的列",
        guidance=FEATURE_ENGINEERING_PROMPT,
    )

    # 任务类型：模型训练
    MODEL_TRAIN = TaskTypeDef(
        name="model train",
        desc="仅用于训练模型",
        guidance=MODEL_TRAIN_PROMPT,
    )

    # 任务类型：模型评估
    MODEL_EVALUATE = TaskTypeDef(
        name="model evaluate",
        desc="仅用于评估模型",
        guidance=MODEL_EVALUATE_PROMPT,
    )

    # 任务类型：图像转网页代码
    IMAGE2WEBPAGE = TaskTypeDef(
        name="image2webpage",
        desc="用于将图像转换为网页代码",
        guidance=IMAGE2WEBPAGE_PROMPT,
    )

    # 其他类型任务
    OTHER = TaskTypeDef(name="other", desc="任何不在定义类别中的任务")

    # 兼容旧版 TaskType 以支持工具推荐（基于类型匹配）
    # 如果没有人为先验知识需要注入，可以不定义任务类型

    # 任务类型：文本转图像
    TEXT2IMAGE = TaskTypeDef(
        name="text2image",
        desc="与 Stable Diffusion 模型相关的文本转图像、图像转图像任务",
    )

    # 任务类型：网页数据爬取
    WEBSCRAPING = TaskTypeDef(
        name="web scraping",
        desc="用于从网页爬取数据",
        guidance=WEB_SCRAPING_PROMPT,
    )

    # 任务类型：邮箱登录
    EMAIL_LOGIN = TaskTypeDef(
        name="email login",
        desc="用于登录电子邮件",
    )

    # 任务类型：软件开发
    DEVELOP_SOFTWARE = TaskTypeDef(
        name="develop software",
        desc="与软件开发相关的标准操作流程（SOP），如撰写 PRD、设计文档、项目计划、编写代码等",
    )

    @property
    def type_name(self):
        """获取任务类型的名称"""
        return self.value.name

    @classmethod
    def get_type(cls, type_name):
        """根据类型名称获取对应的任务类型定义

        参数：
            type_name (str): 任务类型名称

        返回：
            TaskTypeDef: 对应的任务类型定义，如果未找到则返回 None
        """
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None
