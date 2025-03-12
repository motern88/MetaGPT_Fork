#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import typer

from metagpt.const import CONFIG_ROOT

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def generate_repo(
    idea,
    investment=3.0,
    n_round=5,
    code_review=True,
    run_tests=False,
    implement=True,
    project_name="",
    inc=False,
    project_path="",
    reqa_file="",
    max_auto_summarize_code=0,
    recover_path=None,
):
    """运行启动逻辑，可以从 CLI 或其他 Python 脚本调用。"""
    from metagpt.config2 import config
    from metagpt.context import Context
    from metagpt.roles import (
        Architect,
        DataAnalyst,
        Engineer2,
        ProductManager,
        TeamLeader,
    )
    from metagpt.team import Team

    config.update_via_cli(project_path, project_name, inc, reqa_file, max_auto_summarize_code)  # 更新配置
    ctx = Context(config=config)  # 创建上下文对象

    if not recover_path:  # 如果没有恢复路径，则初始化一个新的公司团队
        company = Team(context=ctx)
        company.hire(
            [
                TeamLeader(),
                ProductManager(),
                Architect(),
                Engineer2(),
                # ProjectManager(),
                DataAnalyst(),
            ]
        )

        # 如果需要实现或代码评审，可以聘请更多的工程师
        # if implement or code_review:
        #     company.hire([Engineer(n_borg=5, use_code_review=code_review)]))
        #
        # 如果需要运行测试，则聘请 QA 工程师
        # if run_tests:
        #     company.hire([QaEngineer()])
        #     if n_round < 8:
        #         n_round = 8  # 如果启用了 `--run-tests`，则至少需要 8 轮才能完成所有 QA 操作。
    else:  # 如果有恢复路径，则从存储中恢复项目
        stg_path = Path(recover_path)
        if not stg_path.exists() or not str(stg_path).endswith("team"):
            raise FileNotFoundError(f"{recover_path} 不存在或不是以 `team` 结尾")

        company = Team.deserialize(stg_path=stg_path, context=ctx)  # 从存储反序列化团队对象
        idea = company.idea  # 恢复项目的创意

    company.invest(investment)  # 投资资金
    asyncio.run(company.run(n_round=n_round, idea=idea))  # 运行团队的任务

    return ctx.kwargs.get("project_path")  # 返回项目路径


@app.command("", help="启动一个新的项目。")
def startup(
    idea: str = typer.Argument(None, help="你的创新想法，例如 '创建一个2048游戏'。"),
    investment: float = typer.Option(default=3.0, help="投资金额，用于支持 AI 公司。"),
    n_round: int = typer.Option(default=5, help="模拟的轮次数量。"),
    code_review: bool = typer.Option(default=True, help="是否启用代码评审。"),
    run_tests: bool = typer.Option(default=False, help="是否启用 QA 以添加和运行测试。"),
    implement: bool = typer.Option(default=True, help="是否启用代码实现。"),
    project_name: str = typer.Option(default="", help="项目的唯一名称，例如 'game_2048'。"),
    inc: bool = typer.Option(default=False, help="增量模式，用于处理现有的仓库。"),
    project_path: str = typer.Option(
        default="",
        help="指定旧版本项目的目录路径，用于满足增量需求。",
    ),
    reqa_file: str = typer.Option(
        default="", help="指定源文件名以重写质量保证代码。"
    ),
    max_auto_summarize_code: int = typer.Option(
        default=0,
        help="自动总结代码的最大次数，-1 表示无限制。此参数用于调试工作流。",
    ),
    recover_path: str = typer.Option(default=None, help="从现有的序列化存储中恢复项目"),
    init_config: bool = typer.Option(default=False, help="初始化 MetaGPT 的配置文件。"),
):
    """启动一个项目，成为老板。"""
    if init_config:
        copy_config_to()  # 如果初始化配置文件，调用复制配置函数
        return

    if idea is None:  # 如果没有提供创意，显示错误信息并退出
        typer.echo("缺少参数 'IDEA'。运行 'metagpt --help' 获取更多信息。")
        raise typer.Exit()

    return generate_repo(
        idea,
        investment,
        n_round,
        code_review,
        run_tests,
        implement,
        project_name,
        inc,
        project_path,
        reqa_file,
        max_auto_summarize_code,
        recover_path,
    )


DEFAULT_CONFIG = """# 完整示例: https://github.com/geekan/MetaGPT/blob/main/config/config2.example.yaml
# 代码参考: https://github.com/geekan/MetaGPT/blob/main/metagpt/config2.py
# 配置文档: https://docs.deepwisdom.ai/main/en/guide/get_started/configuration.html
llm:
  api_type: "openai"  # 或 azure / ollama / groq 等
  model: "gpt-4-turbo"  # 或 gpt-3.5-turbo
  base_url: "https://api.openai.com/v1"  # 或转发 URL / 其他 LLM URL
  api_key: "YOUR_API_KEY"
"""


def copy_config_to():
    """初始化 MetaGPT 的配置文件。"""
    target_path = CONFIG_ROOT / "config2.yaml"

    # 创建目标目录（如果不存在）
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果目标文件已经存在，则重命名为 .bak
    if target_path.exists():
        backup_path = target_path.with_suffix(".bak")
        target_path.rename(backup_path)
        print(f"已将现有配置文件备份到 {backup_path}")

    # 复制默认配置文件
    target_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    print(f"配置文件已初始化，路径为 {target_path}")


if __name__ == "__main__":
    app()
