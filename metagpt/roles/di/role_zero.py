from __future__ import annotations

import inspect
import json
import re
import traceback
from datetime import datetime
from typing import Annotated, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from metagpt.actions import Action, UserRequirement
from metagpt.actions.di.run_command import RunCommand
from metagpt.actions.search_enhanced_qa import SearchEnhancedQA
from metagpt.const import IMAGES
from metagpt.exp_pool import exp_cache
from metagpt.exp_pool.context_builders import RoleZeroContextBuilder
from metagpt.exp_pool.serializers import RoleZeroSerializer
from metagpt.logs import logger
from metagpt.memory.role_zero_memory import RoleZeroLongTermMemory
from metagpt.prompts.di.role_zero import (
    ASK_HUMAN_COMMAND,
    ASK_HUMAN_GUIDANCE_FORMAT,
    CMD_PROMPT,
    DETECT_LANGUAGE_PROMPT,
    END_COMMAND,
    JSON_REPAIR_PROMPT,
    QUICK_RESPONSE_SYSTEM_PROMPT,
    QUICK_THINK_EXAMPLES,
    QUICK_THINK_PROMPT,
    QUICK_THINK_SYSTEM_PROMPT,
    QUICK_THINK_TAG,
    REGENERATE_PROMPT,
    REPORT_TO_HUMAN_PROMPT,
    ROLE_INSTRUCTION,
    SUMMARY_PROBLEM_WHEN_DUPLICATE,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from metagpt.roles import Role
from metagpt.schema import AIMessage, Message, UserMessage
from metagpt.strategy.experience_retriever import DummyExpRetriever, ExpRetriever
from metagpt.strategy.planner import Planner
from metagpt.tools.libs.browser import Browser
from metagpt.tools.libs.editor import Editor
from metagpt.tools.tool_recommend import BM25ToolRecommender, ToolRecommender
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import CodeParser, any_to_str, extract_and_encode_images
from metagpt.utils.repair_llm_raw_output import (
    RepairType,
    repair_escape_error,
    repair_llm_raw_output,
)
from metagpt.utils.report import ThoughtReporter


@register_tool(include_functions=["ask_human", "reply_to_human"])
class RoleZero(Role):
    """一个可以动态思考和行动的角色"""

    # 基本信息
    name: str = "Zero"
    profile: str = "RoleZero"
    goal: str = ""
    system_msg: Optional[list[str]] = None  # 使用 None 来符合 llm.aask 的默认值
    system_prompt: str = SYSTEM_PROMPT  # 使用 None 来符合 llm.aask 的默认值
    cmd_prompt: str = CMD_PROMPT
    cmd_prompt_current_state: str = ""
    instruction: str = ROLE_INSTRUCTION
    task_type_desc: Optional[str] = None

    # 反应模式
    react_mode: Literal["react"] = "react"
    max_react_loop: int = 50  # 用于反应模式

    # 工具
    tools: list[str] = []  # 使用特殊符号 ["<all>"] 表示使用所有已注册的工具
    tool_recommender: Optional[ToolRecommender] = None
    tool_execution_map: Annotated[dict[str, Callable], Field(exclude=True)] = {}
    special_tool_commands: list[str] = ["Plan.finish_current_task", "end", "Terminal.run_command", "RoleZero.ask_human"]
    # 独占工具命令列表。如果多个这些命令出现，只有第一个会被保留
    exclusive_tool_commands: list[str] = [
        "Editor.edit_file_by_replace",
        "Editor.insert_content_at_line",
        "Editor.append_file",
        "Editor.open_file",
    ]
    # 默认配备三个基本工具供选择使用
    editor: Editor = Editor(enable_auto_lint=True)
    browser: Browser = Browser()

    # 经验
    experience_retriever: Annotated[ExpRetriever, Field(exclude=True)] = DummyExpRetriever()

    # 其他
    observe_all_msg_from_buffer: bool = True
    command_rsp: str = ""  # 包含命令的原始字符串
    commands: list[dict] = []  # 要执行的命令
    memory_k: int = 200  # 使用的记忆（消息）的数量
    use_fixed_sop: bool = False
    respond_language: str = ""  # 回答人类和发布消息的语言
    use_summary: bool = True  # 是否在最后进行总结

    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "RoleZero":
        # 强制使用该参数用于数据分析师
        assert self.react_mode == "react"

        # 与 DataInterpreter.set_plan_and_tool 大致相同
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop)
        if self.tools and not self.tool_recommender:
            self.tool_recommender = BM25ToolRecommender(tools=self.tools, force=True)
        self.set_actions([RunCommand])

        # HACK: 初始化 Planner，通过动态思考控制；考虑将其形式化为反应模式
        self.planner = Planner(goal="", working_memory=self.rc.working_memory, auto_run=True)

        return self

    @model_validator(mode="after")
    def set_tool_execution(self) -> "RoleZero":
        # 默认映射
        self.tool_execution_map = {
            "Plan.append_task": self.planner.plan.append_task,
            "Plan.reset_task": self.planner.plan.reset_task,
            "Plan.replace_task": self.planner.plan.replace_task,
            "RoleZero.ask_human": self.ask_human,
            "RoleZero.reply_to_human": self.reply_to_human,
        }
        if self.config.enable_search:
            self.tool_execution_map["SearchEnhancedQA.run"] = SearchEnhancedQA().run
        self.tool_execution_map.update(
            {
                f"Browser.{i}": getattr(self.browser, i)
                for i in [
                    "click",
                    "close_tab",
                    "go_back",
                    "go_forward",
                    "goto",
                    "hover",
                    "press",
                    "scroll",
                    "tab_focus",
                    "type",
                ]
            }
        )
        self.tool_execution_map.update(
            {
                f"Editor.{i}": getattr(self.editor, i)
                for i in [
                    "append_file",
                    "create_file",
                    "edit_file_by_replace",
                    "find_file",
                    "goto_line",
                    "insert_content_at_line",
                    "open_file",
                    "read",
                    "scroll_down",
                    "scroll_up",
                    "search_dir",
                    "search_file",
                    "similarity_search",
                    # "set_workdir",
                    "write",
                ]
            }
        )
        # 可以通过子类进行更新
        self._update_tool_execution()
        return self

    @model_validator(mode="after")
    def set_longterm_memory(self) -> "RoleZero":
        """如果配置启用了长期记忆，则为角色设置长期记忆。

        如果 `enable_longterm_memory` 为 True，则设置长期记忆。
        角色名称将作为集合名称。
        """

        if self.config.role_zero.enable_longterm_memory:
            # 使用 config.role_zero 来初始化长期记忆
            self.rc.memory = RoleZeroLongTermMemory(
                **self.rc.memory.model_dump(),
                persist_path=self.config.role_zero.longterm_memory_persist_path,
                collection_name=self.name.replace(" ", ""),
                memory_k=self.config.role_zero.memory_k,
                similarity_top_k=self.config.role_zero.similarity_top_k,
                use_llm_ranker=self.config.role_zero.use_llm_ranker,
            )
            logger.info(f"为角色 '{self.name}' 设置了长期记忆")

        return self

    def _update_tool_execution(self):
        pass

    async def _think(self) -> bool:
        """在'react'模式下有用。使用LLM决定是否以及接下来该做什么。"""
        # 兼容性检查
        if self.use_fixed_sop:
            return await super()._think()

        ### 0. 准备阶段 ###
        if not self.rc.todo:
            return False

        if not self.planner.plan.goal:
            self.planner.plan.goal = self.get_memories()[-1].content
            detect_language_prompt = DETECT_LANGUAGE_PROMPT.format(requirement=self.planner.plan.goal)
            self.respond_language = await self.llm.aask(detect_language_prompt)

        ### 1. 获取经验 ###
        example = self._retrieve_experience()

        ### 2. 获取计划状态 ###
        plan_status, current_task = self._get_plan_status()

        ### 3. 工具/命令信息 ###
        tools = await self.tool_recommender.recommend_tools()
        tool_info = json.dumps({tool.name: tool.schemas for tool in tools})

        ### 角色指令 ###
        instruction = self.instruction.strip()
        system_prompt = self.system_prompt.format(
            role_info=self._get_prefix(),
            task_type_desc=self.task_type_desc,
            available_commands=tool_info,
            example=example,
            instruction=instruction,
        )

        ### 动态决策 ###
        prompt = self.cmd_prompt.format(
            current_state=self.cmd_prompt_current_state,
            plan_status=plan_status,
            current_task=current_task,
            respond_language=self.respond_language,
        )

        ### 最近观察 ###
        memory = self.rc.memory.get(self.memory_k)
        memory = await self.parse_browser_actions(memory)
        memory = await self.parse_editor_result(memory)
        memory = self.parse_images(memory)

        req = self.llm.format_msg(memory + [UserMessage(content=prompt)])
        state_data = dict(
            plan_status=plan_status,
            current_task=current_task,
            instruction=instruction,
        )
        async with ThoughtReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "react"})
            self.command_rsp = await self.llm_cached_aask(req=req, system_msgs=[system_prompt], state_data=state_data)
        self.command_rsp = await self._check_duplicates(req, self.command_rsp)
        return True

    @exp_cache(context_builder=RoleZeroContextBuilder(), serializer=RoleZeroSerializer())
    async def llm_cached_aask(self, *, req: list[dict], system_msgs: list[str], **kwargs) -> str:
        """使用`exp_cache`自动管理经验。

        `RoleZeroContextBuilder` 尝试将经验添加到 `req` 中。
        `RoleZeroSerializer` 提取 `req` 的核心部分，去除冗长内容，仅保留必要的部分。
        """
        return await self.llm.aask(req, system_msgs=system_msgs)

    async def parse_browser_actions(self, memory: list[Message]) -> list[Message]:
        if not self.browser.is_empty_page:
            pattern = re.compile(r"Command Browser\.(\w+) executed")
            for index, msg in zip(range(len(memory), 0, -1), memory[::-1]):
                if pattern.search(msg.content):
                    memory.insert(index, UserMessage(cause_by="browser", content=await self.browser.view()))
                    break
        return memory

    async def parse_editor_result(self, memory: list[Message], keep_latest_count=5) -> list[Message]:
        """保留最新的编辑结果，删除过时的编辑结果。"""
        pattern = re.compile(r"Command Editor\.(\w+?) executed")
        new_memory = []
        i = 0
        for msg in reversed(memory):
            matches = pattern.findall(msg.content)
            if matches:
                i += 1
                if i > keep_latest_count:
                    new_content = msg.content[: msg.content.find("Command Editor")]
                    new_content += "\n".join([f"Command Editor.{match} executed." for match in matches])
                    msg = UserMessage(content=new_content)
            new_memory.append(msg)
        # 反转新内存列表，使最新消息在最后
        new_memory.reverse()
        return new_memory

    def parse_images(self, memory: list[Message]) -> list[Message]:
        if not self.llm.support_image_input():
            return memory
        for msg in memory:
            if IMAGES in msg.metadata or msg.role != "user":
                continue
            images = extract_and_encode_images(msg.content)
            if images:
                msg.add_metadata(IMAGES, images)
        return memory

    def _get_prefix(self) -> str:
        time_info = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return super()._get_prefix() + f" 当前时间是 {time_info}。"

    async def _act(self) -> Message:
        if self.use_fixed_sop:
            return await super()._act()

        commands, ok, self.command_rsp = await self._parse_commands(self.command_rsp)
        self.rc.memory.add(AIMessage(content=self.command_rsp))
        if not ok:
            error_msg = commands
            self.rc.memory.add(UserMessage(content=error_msg, cause_by=RunCommand))
            return error_msg
        logger.info(f"命令: \n{commands}")
        outputs = await self._run_commands(commands)
        logger.info(f"命令输出: \n{outputs}")
        self.rc.memory.add(UserMessage(content=outputs, cause_by=RunCommand))

        return AIMessage(
            content=f"我已完成任务，请标记任务为已完成。输出结果: {outputs}",
            sent_from=self.name,
            cause_by=RunCommand,
        )

    async def _react(self) -> Message:
        # 注意：差异1：每次到这里意味着有新的信息被观察，设置todo以允许处理新信息
        self._set_state(0)

        # 快速思考可以解决的问题无需进行正式的think-act周期
        quick_rsp, _ = await self._quick_think()
        if quick_rsp:
            return quick_rsp

        actions_taken = 0
        rsp = AIMessage(content="尚未采取任何行动", cause_by=Action)  # 后续会被 Role _act 覆盖
        while actions_taken < self.rc.max_react_loop:
            # 注意：差异2：继续观察，新的信息将进入内存，允许根据新信息进行适应
            await self._observe()

            # 思考
            has_todo = await self._think()
            if not has_todo:
                break
            # 执行
            logger.debug(f"{self._setting}: {self.rc.state=}, 将执行 {self.rc.todo}")
            rsp = await self._act()
            actions_taken += 1

            # 后期检查
            if self.rc.max_react_loop >= 10 and actions_taken >= self.rc.max_react_loop:
                # 如果 max_react_loop 是一个较小的值（例如 < 10），那么到达它是预期行为，并且使代理停止
                logger.warning(f"达到了最大操作轮次: {actions_taken}")
                human_rsp = await self.ask_human(
                    "我已经达到最大操作轮次，您希望我继续吗？是或否"
                )
                if "yes" in human_rsp.lower():
                    actions_taken = 0
        return rsp  # 返回最后一轮操作的输出

    def format_quick_system_prompt(self) -> str:
        """格式化快速思考的系统提示。"""
        return QUICK_THINK_SYSTEM_PROMPT.format(examples=QUICK_THINK_EXAMPLES, role_info=self._get_prefix())

    async def _quick_think(self) -> Tuple[Message, str]:
        answer = ""
        rsp_msg = None
        if self.rc.news[-1].cause_by != any_to_str(UserRequirement):
            # 代理本身不会生成快速问题，使用此规则减少多余的LLM调用
            return rsp_msg, ""

        # 路由
        memory = self.get_memories(k=self.memory_k)
        context = self.llm.format_msg(memory + [UserMessage(content=QUICK_THINK_PROMPT)])
        async with ThoughtReporter() as reporter:
            await reporter.async_report({"type": "classify"})
            intent_result = await self.llm.aask(context, system_msgs=[self.format_quick_system_prompt()])

        if "QUICK" in intent_result or "AMBIGUOUS" in intent_result:  # 使用原始上下文调用LLM
            async with ThoughtReporter(enable_llm_stream=True) as reporter:
                await reporter.async_report({"type": "quick"})
                answer = await self.llm.aask(
                    self.llm.format_msg(memory),
                    system_msgs=[QUICK_RESPONSE_SYSTEM_PROMPT.format(role_info=self._get_prefix())],
                )
            # 如果答案包含'[Message] from A to B:'，则移除
            pattern = r"\[Message\] from .+? to .+?:\s*"
            answer = re.sub(pattern, "", answer, count=1)
            if "command_name" in answer:
                # 将实际的任务意图误分类为快速思考，这里进行纠正
                answer = ""
                intent_result = "TASK"
        elif "SEARCH" in intent_result:
            query = "\n".join(str(msg) for msg in memory)
            answer = await SearchEnhancedQA().run(query)

        if answer:
            self.rc.memory.add(AIMessage(content=answer, cause_by=QUICK_THINK_TAG))
            await self.reply_to_human(content=answer)
            rsp_msg = AIMessage(
                content=answer,
                sent_from=self.name,
                cause_by=QUICK_THINK_TAG,
            )

        return rsp_msg, intent_result

    async def _check_duplicates(self, req: list[dict], command_rsp: str, check_window: int = 10):
        past_rsp = [mem.content for mem in self.rc.memory.get(check_window)]
        if command_rsp in past_rsp and '"command_name": "end"' not in command_rsp:
            # 一般情况下，正常的响应包含思考内容不太可能重复
            # 如果检测到相同的响应，说明是坏响应，通常是LLM重复生成的内容
            # 在这种情况下，向人类请求帮助并重新生成
            # TODO: 改为使用llm_cached_aask

            # 硬规则，向人类请求帮助
            if past_rsp.count(command_rsp) >= 3:
                if '"command_name": "Plan.finish_current_task",' in command_rsp:
                    # 检测到重复的'Plan.finish_current_task'命令，并使用'end'命令结束任务
                    logger.warning(f"检测到重复响应: {command_rsp}")
                    return END_COMMAND
                problem = await self.llm.aask(
                    req + [UserMessage(content=SUMMARY_PROBLEM_WHEN_DUPLICATE.format(language=self.respond_language))]
                )
                ASK_HUMAN_COMMAND[0]["args"]["question"] = ASK_HUMAN_GUIDANCE_FORMAT.format(problem=problem).strip()
                ask_human_command = "```json\n" + json.dumps(ASK_HUMAN_COMMAND, indent=4, ensure_ascii=False) + "\n```"
                return ask_human_command
            # 尝试自我修正
            logger.warning(f"检测到重复响应: {command_rsp}")
            regenerate_req = req + [UserMessage(content=REGENERATE_PROMPT)]
            regenerate_req = self.llm.format_msg(regenerate_req)
            command_rsp = await self.llm.aask(regenerate_req)
        return command_rsp

    async def _parse_commands(self, command_rsp) -> Tuple[List[Dict], bool]:
        """从大型语言模型（LLM）中检索命令。

        该函数通过处理响应（`self.command_rsp`）从LLM中检索命令列表。
        它还处理解析过程中的潜在错误和LLM响应格式问题。

        返回:
            - 一个元组，第一个元素是布尔值，表示成功（True）或失败（False）。
        """
        try:
            # 使用 CodeParser 解析代码，假设返回的是 JSON 格式的命令
            commands = CodeParser.parse_code(block=None, lang="json", text=command_rsp)

            # 如果解析结果不完整，则尝试补充格式
            if commands.endswith("]") and not commands.startswith("["):
                commands = "[" + commands

            # 修复 LLM 输出并解析 JSON
            commands = json.loads(repair_llm_raw_output(output=commands, req_keys=[None], repair_type=RepairType.JSON))
        except json.JSONDecodeError as e:
            logger.warning(f"无法解析JSON: {command_rsp}。尝试修复...")

            # 如果解析失败，尝试修复 LLM 返回的数据
            commands = await self.llm.aask(
                msg=JSON_REPAIR_PROMPT.format(json_data=command_rsp, json_decode_error=str(e))
            )
            try:
                # 重新解析修复后的代码
                commands = json.loads(CodeParser.parse_code(block=None, lang="json", text=commands))
            except json.JSONDecodeError:
                # 如果再次解析失败，修复转义错误并尝试解析
                commands = CodeParser.parse_code(block=None, lang="json", text=command_rsp)
                new_command = repair_escape_error(commands)
                commands = json.loads(
                    repair_llm_raw_output(output=new_command, req_keys=[None], repair_type=RepairType.JSON)
                )
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            error_msg = str(e)
            return error_msg, False, command_rsp

        # 为了容错处理 LLM 格式不正确的情况
        if isinstance(commands, dict):
            commands = commands["commands"] if "commands" in commands else [commands]

        # 设置独占命令标志
        command_flag = [command["command_name"] not in self.exclusive_tool_commands for command in commands]
        if command_flag.count(False) > 1:
            # 如果存在多个独占命令，仅保留第一个独占命令
            index_of_first_exclusive = command_flag.index(False)
            commands = commands[: index_of_first_exclusive + 1]
            command_rsp = "```json\n" + json.dumps(commands, indent=4, ensure_ascii=False) + "\n```"
            logger.info(
                "当前命令列表中包含多个独占命令。已更改命令列表。\n" + command_rsp
            )
        return commands, True, command_rsp

    async def _run_commands(self, commands) -> str:
        """执行命令并返回输出"""
        outputs = []
        for cmd in commands:
            output = f"命令 {cmd['command_name']} 执行"

            # 先处理特殊命令
            if self._is_special_command(cmd):
                special_command_output = await self._run_special_command(cmd)
                outputs.append(output + ":" + special_command_output)
                continue

            # 执行指定的工具命令
            if cmd["command_name"] in self.tool_execution_map:
                tool_obj = self.tool_execution_map[cmd["command_name"]]
                try:
                    if inspect.iscoroutinefunction(tool_obj):
                        tool_output = await tool_obj(**cmd["args"])
                    else:
                        tool_output = tool_obj(**cmd["args"])
                    if tool_output:
                        output += f": {str(tool_output)}"
                    outputs.append(output)
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.exception(str(e) + tb)
                    outputs.append(output + f": {tb}")
                    break  # 如果命令失败，停止执行
            else:
                outputs.append(f"命令 {cmd['command_name']} 未找到。")
                break
        outputs = "\n\n".join(outputs)

        return outputs

    def _is_special_command(self, cmd) -> bool:
        """检查命令是否为特殊命令"""
        return cmd["command_name"] in self.special_tool_commands

    async def _run_special_command(self, cmd) -> str:
        """处理特殊命令"""
        command_output = ""

        if cmd["command_name"] == "Plan.finish_current_task":
            if not self.planner.plan.is_plan_finished():
                self.planner.plan.finish_current_task()
            command_output = (
                "当前任务已完成。如果不再需要执行操作，请使用‘end’命令停止。"
            )

        elif cmd["command_name"] == "end":
            command_output = await self._end()
        elif cmd["command_name"] == "RoleZero.ask_human":
            # 向用户请求帮助
            human_response = await self.ask_human(**cmd["args"])
            if human_response.strip().lower().endswith(("stop", "<stop>")):
                human_response += "用户要求我停止，因为遇到问题。"
                self.rc.memory.add(UserMessage(content=human_response, cause_by=RunCommand))
                end_output = "\n执行 end 命令："
                end_output += await self._end()
                return end_output
            return human_response
        elif cmd["command_name"] == "Terminal.run_command":
            # 执行终端命令
            tool_obj = self.tool_execution_map[cmd["command_name"]]
            tool_output = await tool_obj(**cmd["args"])
            if len(tool_output) <= 10:
                command_output += (
                    f"\n[命令]: {cmd['args']['cmd']} \n[命令输出] : {tool_output} (请注意此输出)"
                )
            else:
                command_output += f"\n[命令]: {cmd['args']['cmd']} \n[命令输出] : {tool_output}"

        return command_output

    def _get_plan_status(self) -> Tuple[str, str]:
        """获取计划状态"""
        plan_status = self.planner.plan.model_dump(include=["goal", "tasks"])
        current_task = (
            self.planner.plan.current_task.model_dump(exclude=["code", "result", "is_success"])
            if self.planner.plan.current_task
            else ""
        )
        # 格式化计划状态
        formatted_plan_status = f"[目标] {plan_status['goal']}\n"
        if len(plan_status["tasks"]) > 0:
            formatted_plan_status += "[计划]\n"
            for task in plan_status["tasks"]:
                formatted_plan_status += f"[任务ID {task['task_id']}] ({'完成' if task['is_finished'] else '    '}){task['instruction']} 该任务依赖任务{task['dependent_task_ids']}。 [指派给 {task['assignee']}]\n"
        else:
            formatted_plan_status += "没有计划\n"
        return formatted_plan_status, current_task

    def _retrieve_experience(self) -> str:
        """默认实现经验检索，可以在子类中重写"""
        context = [str(msg) for msg in self.rc.memory.get(self.memory_k)]
        context = "\n\n".join(context)
        example = self.experience_retriever.retrieve(context=context)
        return example

    async def ask_human(self, question: str) -> str:
        """当任务失败或不确定当前遇到的情况时，向人类请求帮助"""
        from metagpt.environment.mgx.mgx_env import MGXEnv  # 避免循环导入

        if not isinstance(self.rc.env, MGXEnv):
            return "不在MGXEnv环境中，命令无法执行。"
        return await self.rc.env.ask_human(question, sent_from=self)

    async def reply_to_human(self, content: str) -> str:
        """当有明确的答案或解决方案时，回复人类用户"""
        from metagpt.environment.mgx.mgx_env import MGXEnv  # 避免循环导入

        if not isinstance(self.rc.env, MGXEnv):
            return "不在MGXEnv环境中，命令无法执行。"
        return await self.rc.env.reply_to_human(content, sent_from=self)

    async def _end(self, **kwarg):
        """结束当前任务并进行总结"""
        self._set_state(-1)
        memory = self.rc.memory.get(self.memory_k)
        # 确保在执行 "end" 命令前有回复人类
        if not any(["reply_to_human" in memory.content for memory in self.get_memories(k=5)]):
            logger.info("手动回复人类")
            reply_to_human_prompt = REPORT_TO_HUMAN_PROMPT.format(respond_language=self.respond_language)
            async with ThoughtReporter(enable_llm_stream=True) as reporter:
                await reporter.async_report({"type": "quick"})
                reply_content = await self.llm.aask(self.llm.format_msg(memory + [UserMessage(reply_to_human_prompt)]))
            await self.reply_to_human(content=reply_content)
            self.rc.memory.add(AIMessage(content=reply_content, cause_by=RunCommand))
        outputs = ""
        # 总结已完成的任务和交付物
        if self.use_summary:
            logger.info("结束当前运行并总结")
            outputs = await self.llm.aask(self.llm.format_msg(memory + [UserMessage(SUMMARY_PROMPT)]))
        return outputs
