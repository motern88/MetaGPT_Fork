from pydantic import BaseModel


class ToolSchema(BaseModel):
    description: str  # 工具模式的描述，描述工具的功能或用途


class Tool(BaseModel):
    name: str  # 工具的名称
    path: str  # 工具的路径
    schemas: dict = {}  # 工具的模式，默认是空字典
    code: str = ""  # 工具的源代码，默认为空字符串
    tags: list[str] = []  # 工具的标签，默认为空列表，用于标记工具的特性或分类
