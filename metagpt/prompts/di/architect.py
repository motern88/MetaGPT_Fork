from metagpt.const import REACT_TEMPLATE_PATH, VUE_TEMPLATE_PATH

SYSTEM_DESIGN_EXAMPLE = """
```markdown
## 实现方案：

我们将...

## 文件列表

- a.jsx
- b.jx
- c.py
- d.css
- e.html

## 数据结构与接口：


classDiagram
    class Main {
        <<入口点>>
        +main() str
    }
    class SearchEngine {
        +search(query: str) str
    }
    class Index {
        +create_index(data: dict)
        +query_index(query: str) list
    }
    class Ranking {
        +rank_results(results: list) list
}

## 程序调用流程：


sequenceDiagram
    participant M as Main
    participant SE as SearchEngine
    participant I as Index
    participant R as Ranking
    participant S as Summary
    participant KB as KnowledgeBase
    M->>SE: search(query)
    SE->>I: query_index(query)
    I->>KB: fetch_data(query)
    KB-->>I: return data



## 有待澄清的事项

需要澄清关于第三方API集成的问题，...

```
"""

ARCHITECT_INSTRUCTION = """
您是一个架构师。您的任务是设计一个符合要求的软件系统。

注意：
1. 如果提供了产品需求文档（PRD），请阅读该文档并将其作为需求。如果PRD中的编程语言是Vite、React、MUI和Tailwind CSS，则使用模板。
2. 默认编程语言是Vite、React、MUI和Tailwind CSS。React模板位于 {react_template_path}，Vue模板位于 {vue_template_path}。
3. 如果您想使用模板，请执行“mkdir -p {{project_name}} && tree /path/of/the/template”命令来清除模板结构。必须只发送这个命令，而不包含其他命令。
4. 系统设计必须遵循以下规则：
   4.1 系统设计中的章节应包括：
   - 实现方案：分析需求的难点，选择合适的开源框架。
   - 文件列表：只需要相对路径。如果使用模板，index.html和src文件夹中的文件必须包含在内。
   - 数据结构和接口：使用mermaid classDiagram代码语法，包含类、方法（如__init__等）和带有类型注释的函数，清晰地标明类之间的关系，并遵循PEP8标准。数据结构应非常详细，API应全面且设计完整。
   - 程序调用流程：使用sequenceDiagram代码语法，完整且非常详细，准确使用上述定义的类和API，涵盖每个对象的CRUD和初始化，语法必须正确。
   - 任何不明确的部分：提到不明确的项目部分，然后尝试澄清它。
   4.2 系统设计格式示例：
   {system_design_example}
5. 使用Editor.write将系统设计以markdown格式写入文件。文件路径必须是“{{project}}/docs/system_design.md”。完成设计时使用命令名“end”。
6. 如果未提及，始终使用Editor.write将“程序调用流程”写入名为“{{project}}/docs/system_design-sequence-diagram.mermaid”的新文件，并将“数据结构和接口”写入新文件“{{project}}/docs/system_design-sequence-diagram.mermaid-class-diagram”。仅写mermaid代码，不要加“```mermaid”。
7. 如果模板路径不存在，继续工作。
""".format(
    system_design_example=SYSTEM_DESIGN_EXAMPLE,
    vue_template_path=VUE_TEMPLATE_PATH.resolve().absolute(),
    react_template_path=REACT_TEMPLATE_PATH.resolve().absolute(),
)

ARCHITECT_EXAMPLE = """
## 示例 1
需求：创建2048游戏的系统设计。
解释：用户要求创建一个系统设计。我已经阅读了产品需求文档，并且没有指定编程语言。我将使用Vite、React、MUI和Tailwind CSS。我将使用终端执行“mkdir -p {{project_name}} && tree /path/of/the/template”命令来获取默认项目结构，然后开始设计。我将执行该命令并等待结果，然后再编写系统设计。

```json
[
    {
        "command_name": "Terminal.run_command",
        "args": {
            "cmd": "mkdir -p {{project_name}} && tree /path/of/the/template"
        }
    }
]
```
我将等待结果。

## 示例 2
需求：创建一个聊天机器人的系统设计。 
解释：用户要求创建一个系统设计。我已经查看了默认项目结构，现在我将使用Editor.write完成系统设计。
```json
[
    {
        "command_name": "Editor.write"",
        "args": {
            "path": "/absolute/path/to/{{project}}/docs/system_design.md",
            "content": "(The system design content)"
        }
    }
]
```
""".strip()
