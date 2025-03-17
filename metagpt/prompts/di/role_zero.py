from metagpt.const import EXPERIENCE_MASK

ROLE_INSTRUCTION = """
根据上下文，编写一个计划或修改现有计划以实现目标。一个计划由一到三个任务组成。  
如果创建计划，您应当跟踪进度并根据需要更新计划，如：`Plan.finish_current_task`, `Plan.append_task`, `Plan.reset_task`, `Plan.replace_task` 等。  
当当前任务呈现时，使用可用的命令解决该任务。  
特别注意新用户的消息，回顾对话历史，使用 `RoleZero.reply_to_human` 来回应新的用户需求。  

注意事项：  
1. 如果不断遇到错误、意外情况，或者不确定如何继续，使用 `RoleZero.ask_human` 寻求帮助。  
2. 仔细审视当前任务的进展，如果你之前的操作尚未完成任务指令，你应当继续当前任务。否则，明确地使用 `Plan.finish_current_task` 完成当前任务。  
3. 每次完成任务时，使用 `RoleZero.reply_to_human` 报告你的进展。  
4. 当所有现有任务完成且需要新任务时，首先使用 `Plan.append_task` 添加新任务。  
5. 避免重复已经完成的任务，当所有需求完成时，结束循环。  
"""

########################## ignore guidance

# Latest Observation
# {latest_observation}

# {thought_guidance}
# Finally, combine your thoughts, describe what you want to do conscisely in 20 words, including which process you will taked and whether you will end, then follow your thoughts to list the commands, adhering closely to the instructions provided.

###########################
SYSTEM_PROMPT = """
# 基本信息
{role_info}

# 数据结构
class Task(BaseModel):
    task_id: str = ""
    dependent_task_ids: list[str] = []
    instruction: str = ""
    task_type: str = ""
    assignee: str = ""

# 可用的任务类型
{task_type_desc}

# 可用的命令
{available_commands}
特殊命令：使用 {{"command_name": "end"}} 来表示不执行任何操作或表示完成所有需求并结束操作。

# 示例
{example}

# 任务说明
{instruction}

"""

CMD_EXPERIENCE_MASK = f"""
# 过去经验
{EXPERIENCE_MASK}
"""

CMD_PROMPT = (
    CMD_EXPERIENCE_MASK
    + """
# 工具状态
{current_state}

# 当前计划
{plan_status}

# 当前任务
{current_task}

# 回复语言
你必须使用 {respond_language} 回复。

请密切关注提供的示例，如果适合当前情况，可以复用该示例。
如果你打开了一个文件，行号会显示在每行的前面。
你可以使用任何可用命令来创建计划或更新计划。你可以输出多个命令，它们将按顺序执行。
如果你完成了当前任务，你将自动进入现有计划中的下一个任务，使用 Plan.finish_current_task，不要添加新任务。
回顾最新计划的结果，专注于已完成的任务。如果已完成的任务符合当前任务，可以认为它已完成。
在当前命令列表中，不允许使用 Editor.insert_content_at_line 和 Editor.edit_file_by_replace 超过一次。因为这两个命令是互斥的，执行后会改变行号。
在你的回复中，至少包含一个命令。如果你想停止，使用 {{"command_name":"end"}} 命令。

# 你的命令应以正确的命令名和参数输出，以下格式必须遵循，包含在命令前的文本说明。
例如，已完成的任务，接下来的任务，如何更新计划状态，如何回复查询，或寻求帮助。然后是命令的 JSON 数组。你必须输出**一个且只有一个**JSON数组，不能在两者之间输出多个JSON数组。

```json
[
    {
        "command_name": "ClassName.method_name" 或 "function_name",
        "args": {"arg_name": arg_value, ...}
    },
    ...
]
```
注意：你的输出JSON数据部分必须以**```json [** 开始
"""
)
THOUGHT_GUIDANCE = """
首先，描述你最近采取的行动。  
其次，描述你最近收到的消息，特别是来自用户的消息。如果有必要，制定一个计划来满足新的用户需求。  
第三，描述计划状态和当前任务。回顾历史，如果`当前任务`已被你或其他人完成，你必须使用**Plan.finish_current_task**命令来结束该任务，然后再采取任何行动，该命令会自动将你移动到下一个任务。  
第四，描述任何必要的人工交互。如果完成任务或整体需求，使用**RoleZero.reply_to_human**来报告你的进展，注意历史记录，不要重复报告。如果你未能完成当前任务、对遇到的情况不确定、需要人类的帮助，或者执行重复命令但收到重复反馈而没有进展，请使用**RoleZero.ask_human**。  
第五，描述你是否应该终止。如果满足以下任何条件，你应该使用**end**命令终止：  
 - 你已经完成了用户的整体需求  
 - 所有任务已完成且当前任务为空  
 - 你正在重复回复人类
""".strip()

REGENERATE_PROMPT = """
仔细回顾并反思历史记录，提供一个不同的响应。  
描述你是否应该使用**end**命令终止，或者使用**RoleZero.ask_human**向人类寻求帮助，或者尝试不同的方法并输出不同的命令。你不允许再次提供相同的命令。  
当所有任务已完成且需求得到满足时，你应该使用“end”来停止。  
你的反思，然后是命令的JSON数组：
"""
END_COMMAND = """
```json
[
    {
        "command_name": "end",
        "args": {}
    }
]
```
"""

SUMMARY_PROBLEM_WHEN_DUPLICATE = """你遇到了一个问题，导致了重复的命令。请直接告诉我是什么让你困惑或烦恼。请不要输出任何命令。用{language}表达你的问题，且不超过30个字。"""

ASK_HUMAN_GUIDANCE_FORMAT = """
我遇到了以下问题：
{problem}
你能给我一些指导吗？如果你想停止，请在指导中加入"<STOP>"。
"""
ASK_HUMAN_COMMAND = [{"command_name": "RoleZero.ask_human", "args": {"question": ""}}]

JSON_REPAIR_PROMPT = """
## JSON 数据
{json_data}

## JSON 解码错误
{json_decode_error}

## 输出格式
```json

```
不要在 JSON 数据中使用转义字符，尤其是在文件路径中。
请帮助检查 JSON 数据是否存在格式问题？如果有，请帮助格式化。
如果未检测到问题，则应返回未更改的原始 JSON 数据。不要省略任何信息。
输出可以被 json.loads() 函数加载的 JSON 数据格式。
"""

QUICK_THINK_SYSTEM_PROMPT = """
{role_info}
你的角色是确定给定请求的适当响应类别。

# 响应类别
## QUICK（快速响应）：
适用于可以直接回答的简单问题或请求。包括常识性询问、法律或逻辑问题、基础数学、简短的编码任务、选择题、问候、闲聊、日常计划，以及关于你或你的团队的询问。

## SEARCH（搜索）：
适用于需要检索最新或详细信息的问题。包括时间敏感或地点特定的问题，例如当前事件或天气。仅当信息不易获取时才使用此类别。  
如果提供了文件或链接，则无需搜索额外信息。

## TASK（任务）：
适用于涉及工具使用、计算机操作、多个步骤或详细说明的请求。例如软件开发、项目规划或任何需要工具使用的任务。

## AMBIGUOUS（模糊）：
适用于不清楚、缺乏足够细节或超出系统能力范围的请求。模糊请求的常见特征包括：

- 信息不完整：暗示复杂任务但缺乏关键细节的请求（例如，“重新设计这个徽标”但未指定设计要求）。
- 模糊性：广泛、未指定或不清楚的请求，难以提供精确答案。
- 不切实际的范围：过于宽泛的请求，无法在单个响应中有意义地解决（例如，“告诉我关于……的一切”）。
- 缺少文件：引用特定文档、图像或数据但未提供参考内容的请求（当提供文件、网站或数据时，必须包含内容、链接或路径）。

**注意：** 在将请求归类为 TASK 之前：
1. 考虑用户是否提供了足够的信息来执行任务。如果请求复杂但缺乏关键细节或未提供文件内容或路径，则应归类为 AMBIGUOUS。
2. 如果请求是“如何做”的问题，要求提供一般计划、方法或策略，则应归类为 QUICK。

{examples}
"""

QUICK_THINK_PROMPT = """
# 指令
确定上一条消息的意图。  
用一个简洁的思考来回应，然后提供适当的响应类别：QUICK、SEARCH、TASK 或 AMBIGUOUS。

# 格式
思考：[你的思考内容]  
响应类别：[QUICK/SEARCH/TASK/AMBIGUOUS]

# 响应：
"""


QUICK_THINK_EXAMPLES = """
# 示例

1. 请求：“如何设计一个支持实时协作的在线文档编辑平台？”  
思考：这是一个关于平台设计的直接问题，无需额外资源即可回答。  
响应类别：QUICK。

2. 请求：“监督学习和无监督学习在机器学习中有什么区别？”  
思考：这是一个可以简洁回答的常识性问题。  
响应类别：QUICK。

3. 请求：“请帮我写一个关于Python网络爬虫的学习计划。”  
思考：编写学习计划是一个可以直接回答的日常规划任务。  
响应类别：QUICK。

4. 请求：“你能帮我找到关于深度学习的最新研究论文吗？”  
思考：用户需要最新的研究，需要搜索最新的资源。  
响应类别：SEARCH。

5. 请求：“构建一个运行生命游戏模拟的个人网站。”  
思考：这是一个需要多个步骤的详细软件开发任务。  
响应类别：TASK。

6. 请求：“请为我总结这份文档。”  
思考：请求提到总结文档，但未提供文档的路径或内容，无法完成。  
响应类别：AMBIGUOUS。

7. 请求：“请为我总结这份文档 '/data/path/docmument.pdf'。”  
思考：请求提到总结文档，并提供了文档路径。可以通过工具读取文档后进行总结。  
响应类别：TASK。

8. 请求：“优化这个流程。”  
思考：请求模糊且缺乏具体细节，需要明确要优化的流程。  
响应类别：AMBIGUOUS。

9. 请求：“将styles.css中的文本颜色改为蓝色，在网页中添加一个新按钮，删除旧的背景图片。”  
思考：这是一个需要修改一个或多个文件的增量开发任务。  
响应类别：TASK。
"""
QUICK_RESPONSE_SYSTEM_PROMPT = """
{role_info}
然而，你必须直接回应用户消息，**不要**询问你的团队成员。
"""
# A tag to indicate message caused by quick think
QUICK_THINK_TAG = "快速思考"

REPORT_TO_HUMAN_PROMPT = """
## 示例
示例 1:  
用户需求：创建一个2048游戏  
回复：2048游戏的开发已完成。所有文件（index.html、style.css 和 script.js）已创建并审核完毕。

示例 2:  
用户需求：从网站上爬取并提取所有草药名称，告诉我草药的数量。  
回复：草药名称已成功提取。共提取了8种草药名称。

------------

仔细回顾历史记录，并以用户期望的语言回应以满足他们的需求。  
如果你有任何有助于解释结果的交付物（例如部署URL、文件、指标、定量结果等），请简要描述它们。  
你的回复必须简洁。  
你必须以{respond_language}语言回复。  
直接输出你的回复内容。不要添加任何输出格式。
"""
SUMMARY_PROMPT = """
总结你最近完成的工作。保持简洁。  
如果你有任何交付物，请包含它们的简要描述和文件路径。如果有任何指标、URL 或定量结果，也请包含它们。  
如果交付物是代码，仅输出文件路径。
"""

DETECT_LANGUAGE_PROMPT = """
需求是：  
{requirement}  

你必须以哪种自然语言回应？  
仅输出语言类型。
"""
