"""FLARE 引擎。

使用 llamaindex 的 FLAREInstructQueryEngine 作为 FLAREEngine，它接受其他引擎作为参数。
例如，创建一个简单的引擎，然后将其传递给 FLAREEngine。
"""

from llama_index.core.query_engine import (  # noqa: F401
    FLAREInstructQueryEngine as FLAREEngine,  # 导入 FLAREInstructQueryEngine 并重命名为 FLAREEngine
)
