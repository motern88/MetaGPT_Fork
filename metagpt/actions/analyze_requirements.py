from metagpt.actions import Action

ANALYZE_REQUIREMENTS = """
# 示例
{examples}

----------------

# 需求
{requirements}

# 指令
{instructions}

# 输出格式
{output_format}

请遵循指令和输出格式。不要包含任何额外的内容。
"""

EXAMPLES = """
示例 1
需求: 
创建一个贪吃蛇，只需要给出设计文档和代码
输出:
[用户限制] : 只需要给出设计文档和代码.
[语言限制] : 响应、消息和指令必须使用中文。
[编程语言] : HTML (*.html), CSS (*.css), 和 JavaScript (*.js)

示例 2
需求:
使用 Python 创建 2048 游戏。不要写 PRD。
输出:
[用户限制] : 不要写 PRD。
[语言限制] : 响应、消息和指令必须使用英文。
[编程语言] : Python

示例 3
需求:
你必须忽略创建 PRD 和 TRD。帮助我写一个巴黎奥运会的日程展示程序。
输出:
[用户限制] : 你必须忽略创建 PRD 和 TRD。
[语言限制] : 响应、消息和指令必须使用英文。
[编程语言] : HTML (*.html), CSS (*.css), 和 JavaScript (*.js)
"""

INSTRUCTIONS = """
你必须使用与需求中相同的语言输出。
首先，应该确定与需求描述中使用的语言一致的自然语言。如果需求中指定了特定语言，请遵循这些指令。默认情况下，响应语言为英文。
其次，从需求中提取限制条件，特别是步骤。不要包括详细的需求描述；只关注限制条件。
第三，如果需求是软件开发，提取程序语言。如果没有指定编程语言，使用 HTML (*.html)、CSS (*.css) 和 JavaScript (*.js)。

注意：
1. 如果没有限制条件，要求的 `requirements_restrictions` 应为 ""
2. 如果需求不是软件开发，编程语言应为空 ""
"""

OUTPUT_FORMAT = """
[用户限制] : 需求中的限制
[语言限制] : 响应、消息和指令必须使用 {{language}}
[编程语言] : 你的程序必须使用...
"""

class AnalyzeRequirementsRestrictions(Action):
    """
    分析需求限制类，负责分析给定需求的约束条件，并确定语言和编程语言
    继承自 Action 类，用于处理需求分析
    """

    name: str = "AnalyzeRequirementsRestrictions"

    async def run(self, requirements, isinstance=INSTRUCTIONS, output_format=OUTPUT_FORMAT):
        """
        分析需求中的约束条件和使用的语言。

        参数:
        - requirements: 需求文本
        - isinstance: 用于分析需求的指令，默认为 INSTRUCTIONS
        - output_format: 输出格式模板，默认为 OUTPUT_FORMAT

        返回:
        - rsp: 分析后的结果
        """
        # 构建用于分析的 prompt
        prompt = ANALYZE_REQUIREMENTS.format(
            examples=EXAMPLES, requirements=requirements, instructions=isinstance, output_format=output_format
        )
        # 发送请求并获取响应
        rsp = await self.llm.aask(prompt)
        return rsp
