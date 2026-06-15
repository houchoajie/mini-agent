"""
工具基类定义

所有 Agent 可调用的工具都必须继承 BaseTool 并实现以下接口：
- name: 工具唯一标识名
- description: 工具功能描述（供 LLM 理解）
- parameters: JSON Schema 格式的参数定义
- execute(): 实际执行逻辑
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class ToolResult:
    """
    工具执行结果封装

    Attributes:
        success: 是否执行成功
        result: 执行结果内容（字符串形式，会反馈给 LLM）
        error: 如果失败，错误信息
        metadata: 额外元数据（如执行耗时等）
    """
    success: bool
    result: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        """将结果转为 LLM 可读的字符串"""
        if self.success:
            return self.result
        return f"[ERROR] {self.error}"


class BaseTool(ABC):
    """
    工具基类 - 所有工具必须继承此类

    子类需要实现：
    - name (property): 工具名称
    - description (property): 工具描述
    - parameters (property): 参数 JSON Schema
    - execute(**kwargs) -> ToolResult: 执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具的唯一标识名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具的功能描述，会展示给 LLM"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数的 JSON Schema 定义"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具的核心逻辑

        Args:
            **kwargs: 工具参数，由 LLM 根据 parameters schema 生成

        Returns:
            ToolResult: 包含执行结果或错误信息
        """
        ...

    def to_openai_schema(self) -> dict:
        """
        转换为 OpenAI function calling 格式

        Returns:
            符合 OpenAI tools API 的字典格式：
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "...",
                    "parameters": { ... }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def safe_execute(self, arguments: str | dict) -> ToolResult:
        """
        安全执行工具 - 包装异常处理

        负责：
        1. 解析 JSON 字符串参数为字典
        2. 调用 execute() 并捕获异常
        3. 返回统一的 ToolResult

        Args:
            arguments: JSON 字符串或字典格式的参数

        Returns:
            ToolResult: 执行结果或错误信息
        """
        try:
            # 如果 arguments 是字符串，解析为字典
            if isinstance(arguments, str):
                args_dict = json.loads(arguments) if arguments.strip() else {}
            else:
                args_dict = arguments

            # 执行工具
            return self.execute(**args_dict)

        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"参数解析失败: {e}. 原始参数: {arguments}",
            )
        except TypeError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"参数类型错误: {e}. 请检查参数格式。",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result="",
                error=f"工具执行异常: {type(e).__name__}: {e}",
            )
