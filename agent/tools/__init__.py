"""
工具模块 - 包含所有 Agent 可调用的工具
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
# Agent Runtime 通过此列表获取工具描述和执行能力
# 新增工具只需：1. 在 tools/ 下创建文件 2. 在此处导入并加入列表
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
    """根据名称查找工具实例"""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def get_all_tool_schemas() -> list[dict]:
    """
    获取所有工具的 OpenAI function calling schema
    用于传递给 LLM API 的 tools 参数
    """
    return [tool.to_openai_schema() for tool in ALL_TOOLS]
