from metagpt.strategy.task_type import TaskType

EXTRA_INSTRUCTION = """
6. 处理网页任务时要小心：
 - 对于一般的信息搜索（如新闻、天气、维基等查询搜索引擎），使用SearchEnhancedQA。通常不提供链接。
 - 对于在特定网站内阅读、导航或域内搜索，使用Browser工具，例如在博客中阅读、从给定的电商网站链接中搜索产品或与网页应用交互。
 - 对于网页抓取（如批量收集数据或从提供的链接获取信息），使用DataAnalyst.write_and_execute_code工具。
 - 写代码来查看HTML内容，而不是使用Browser工具。
 - 确保在使用Browser工具时，command_name一定在可用命令列表中。

7. 在制定计划时，建议在第一次响应时仔细考虑并一次性添加所有任务，除了7.1以外。
7.1. 如果需求是查询pdf、docx、md或txt文档，请首先通过Editor.read读取该文档，无需立即计划。读取文档后，如果可以直接回答，则使用RoleZero.reply_to_human；否则，在需要进一步计算时制定计划。

8. 不要多次为同一任务执行“finish_current_task”命令。

9. 在及时完成当前任务时，例如代码写入并执行成功时，确保及时执行。

10. 使用“end”命令时，在执行之前添加“finish_current_task”命令。
"""

TASK_TYPE_DESC = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])


CODE_STATUS = """
**代码编写情况**:
{code}  # 显示已经编写的代码内容。

**执行状态**: {status}  # 显示代码执行状态，例如成功或失败。

**执行结果**: {result}  # 显示执行结果，例如输出值或错误信息。
"""
