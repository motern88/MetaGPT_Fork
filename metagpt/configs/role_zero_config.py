from pydantic import Field

from metagpt.utils.yaml_model import YamlModel


class RoleZeroConfig(YamlModel):
    """角色零配置类
    该类用于配置角色的记忆相关设置，包括长期记忆和短期记忆的相关参数。

    属性：
        enable_longterm_memory (bool): 是否启用长期记忆功能。
        longterm_memory_persist_path (str): 保存长期记忆数据的目录路径。
        memory_k (int): 短期记忆的容量。
        similarity_top_k (int): 获取与当前输入最相似的长期记忆数量。
        use_llm_ranker (bool): 是否使用 LLM 重新排序器来优化结果。
    """

    enable_longterm_memory: bool = Field(default=False, description="是否启用长期记忆。")
    longterm_memory_persist_path: str = Field(default=".role_memory_data", description="保存长期记忆数据的目录路径。")
    memory_k: int = Field(default=200, description="短期记忆的容量。")
    similarity_top_k: int = Field(default=5, description="获取与当前输入最相似的长期记忆数量。")
    use_llm_ranker: bool = Field(default=False, description="是否使用 LLM 重新排序器来优化结果。")