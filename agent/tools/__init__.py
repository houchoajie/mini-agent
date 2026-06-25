"""
============================================================
工具注册表 — 所有 Agent 可调用工具的集中管理
============================================================

注册表模式：所有工具实例集中在一个列表中管理。
新增工具只需：1. 在 tools/ 下创建文件 2. 在此处导入并加入 ALL_TOOLS。

为什么用注册表而非动态发现：
- 显式导入：IDE 可以静态分析，支持跳转和重构
- 可控的顺序：注册顺序影响 to_openai_schema() 返回的工具列表顺序
- 明确的依赖：工具间的依赖关系一目了然
- 编译时检查：导入错误在启动时就能发现，而非运行时

使用方式：
    # 根据名称查找工具
    tool = get_tool_by_name("calculator")
    if tool:
        result = tool.execute(expression="2 + 3")

    # 获取所有工具的 OpenAI schema（用于传递给 LLM）
    schemas = get_all_tool_schemas()
"""

from agent.tools.base import BaseTool, ToolResult
from agent.tools.calculator import CalculatorTool
from agent.tools.search import SearchTool
from agent.tools.todo_manager import TodoManagerTool
from agent.tools.file_reader import FileReaderTool
from agent.tools.file_writer import FileWriterTool
from agent.tools.datetime_tool import DateTimeTool

# ============================================================
# 工具注册表：所有可用工具的实例列表
# 每个工具实例是单例的（工具通常无状态或状态由上下文管理）
# ============================================================
ALL_TOOLS: list[BaseTool] = [
    CalculatorTool(),
    SearchTool(),
    TodoManagerTool(),
    FileReaderTool(),
    FileWriterTool(),
    DateTimeTool(),
]


def get_tool_by_name(name: str) -> BaseTool | None:
    """根据工具名称查找工具实例。未找到返回 None。"""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def get_all_tool_schemas() -> list[dict]:
    """
    获取所有工具的 OpenAI function calling schema。

    用于传递给 LLM API 的 tools 参数，让 LLM 知道有哪些工具可用。
    每个 schema 包含工具的名称、描述、参数定义。
    """
    return [tool.to_openai_schema() for tool in ALL_TOOLS]
