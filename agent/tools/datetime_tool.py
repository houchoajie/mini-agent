"""
DateTime 工具 - 获取当前日期时间

让 Agent 具备时间感知能力，支持：
- 获取当前日期和时间
- 自定义输出格式
- 时间戳转换

使用示例：
    format: "%Y-%m-%d %H:%M:%S"
    返回: "当前时间: 2024-01-15 14:30:00"
"""

from datetime import datetime
from agent.tools.base import BaseTool, ToolResult


class DateTimeTool(BaseTool):
    """
    日期时间工具 - 让 Agent 具备时间感知

    工作原理：
    1. 获取系统当前时间
    2. 按指定格式格式化输出
    3. 同时返回 Unix 时间戳

    使用场景：
    - 用户问"现在几点了"
    - 需要记录任务的时间戳
    - 需要计算时间差
    - 需要在文件名中加入时间信息
    """

    @property
    def name(self) -> str:
        return "datetime_tool"

    @property
    def description(self) -> str:
        return (
            "获取当前日期和时间信息。"
            "支持自定义输出格式（strftime 格式），如 '%Y-%m-%d' 仅日期，'%H:%M:%S' 仅时间。"
            "同时返回 Unix 时间戳。可用于时间相关查询、任务记录、文件命名等。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": (
                        "输出格式（Python strftime 格式字符串）。"
                        "常用: '%Y-%m-%d %H:%M:%S' (完整), '%Y-%m-%d' (仅日期), "
                        "'%H:%M:%S' (仅时间), '%Y年%m月%d日' (中文日期)"
                    ),
                    "default": "%Y-%m-%d %H:%M:%S",
                },
            },
        }

    def execute(self, format: str = "%Y-%m-%d %H:%M:%S") -> ToolResult:
        """
        获取当前时间

        执行流程：
        1. 获取 datetime.now()
        2. 按 format 格式化
        3. 计算 Unix 时间戳
        4. 返回格式化结果 + 时间戳

        Args:
            format: strftime 格式字符串

        Returns:
            ToolResult: 格式化的时间信息
        """
        now = datetime.now()

        try:
            formatted = now.strftime(format)
        except (ValueError, TypeError) as e:
            return ToolResult(
                success=False, result="",
                error=f"时间格式错误: {e}。请使用有效的 strftime 格式。",
            )

        # 构建详细输出
        result = (
            f"当前时间: {formatted}\n"
            f"  Unix 时间戳: {int(now.timestamp())}\n"
            f"  星期: {['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}\n"
            f"  ISO 格式: {now.isoformat()}"
        )

        return ToolResult(
            success=True,
            result=result,
            metadata={
                "formatted": formatted,
                "timestamp": int(now.timestamp()),
                "weekday": now.weekday(),
            },
        )
