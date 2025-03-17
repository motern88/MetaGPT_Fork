# function in tools, https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
# Reference: https://github.com/KillianLucas/open-interpreter/blob/v0.1.14/interpreter/llm/setup_openai_coding_llm.py

# GENERAL_FUNCTION_SCHEMA 定义了一个通用的函数架构，用于执行代码并返回结果
GENERAL_FUNCTION_SCHEMA = {
    "name": "execute",  # 函数的名称是 "execute"
    "description": "Executes code on the user's machine, **in the users local environment**, and returns the output",  # 描述执行代码并返回结果
    "parameters": {
        "type": "object",  # 参数类型是对象
        "properties": {
            "language": {
                "type": "string",  # 编程语言，类型为字符串
                "description": "The programming language (required parameter to the `execute` function)",  # 描述：指定编程语言（execute 函数的必需参数）
                "enum": [
                    "python",  # 支持的编程语言包括 Python
                    "R",  # 支持的编程语言包括 R
                    "shell",  # 支持的编程语言包括 shell
                    "applescript",  # 支持的编程语言包括 AppleScript
                    "javascript",  # 支持的编程语言包括 JavaScript
                    "html",  # 支持的编程语言包括 HTML
                    "powershell",  # 支持的编程语言包括 PowerShell
                ],
            },
            "code": {
                "type": "string",  # 代码参数，类型为字符串
                "description": "The code to execute (required)",  # 描述：要执行的代码（必需）
            },
        },
        "required": ["language", "code"],  # 必须提供 language 和 code 两个参数
    },
}


# tool_choice 是用于 general_function_schema 的工具选择值，指明使用的函数名称
# https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
GENERAL_TOOL_CHOICE = {"type": "function", "function": {"name": "execute"}}  # 选择执行 "execute" 函数作为工具

# MULTI_MODAL_MODELS 列出支持多模态的模型名称
MULTI_MODAL_MODELS = [
    "gpt-4o",  # 支持多模态的 GPT-4o 模型
    "gpt-4o-mini",  # 支持多模态的 GPT-4o-mini 模型
    "openai/gpt-4o",  # 支持多模态的 openai/gpt-4o 模型
    "gemini-2.0-flash-exp",  # 支持多模态的 Gemini 2.0-flash-exp 模型
    "gemini-2.0-pro-exp-02-05",  # 支持多模态的 Gemini 2.0-pro-exp-02-05 模型
    "claude-3-5-sonnet-v2",  # 支持多模态的 Claude 3.5-sonnet-v2 模型
    "google/gemini-2.0-flash-exp:free",  # 支持多模态的 google/gemini-2.0-flash-exp:free 模型
    "google/gemini-2.0-pro-exp-02-05:free",  # 支持多模态的 google/gemini-2.0-pro-exp-02-05:free 模型
    "anthropic/claude-3.5-sonnet",  # 支持多模态的 Anthropic Claude 3.5-sonnet 模型
    "anthropic/claude-3.7-sonnet",  # 支持多模态的 Anthropic Claude 3.7-sonnet 模型
]
