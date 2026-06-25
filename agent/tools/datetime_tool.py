"""
============================================================
DateTime 工具 — 获取当前日期时间，让 Agent 具备时间感知
============================================================

为什么 Agent 需要时间感知：
    - 用户问"现在几点了"——需要实时时间
    - 创建任务时记录时间戳——需要一致的格式
    - 生成文件名时包含时间——需要自定义格式
    - 计算时间差——需要 Unix 时间戳

虽然每次 LLM 调用时系统 prompt 可以注入当前时间，
但让 Agent 主动调用工具获取时间更灵活：
    - 可以自定义格式（日期/时间/完整）
    - 可以多次获取（计算耗时）
    - 不依赖 system prompt 的注入机制

缓存策略：TTL=1 秒。虽然很短，但在同一次 ReAct 循环中
连续两次调用 get_time 能返回毫秒级精准的结果。
"""

from datetime import datetime
from agent.tools.base import BaseTool, ToolResult


class DateTimeTool(BaseTool):
    """
    日期时间工具 — 让 Agent 具备时间感知。

    工作原理：
    1. 获取系统当前时间（datetime.now()）
    2. 按指定格式（strftime）格式化输出
    3. 同时返回 Unix 时间戳、星期、ISO 格式
    """

    @property
    def name(self) -> str:
        return "datetime_tool"

    @property
    def timeout(self) -> float:
        return 5.0

    @property
    def cache_ttl(self) -> float:
        """时间每秒都在变，缓存时间很短（1 秒）。"""
        return 1.0

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
        获取当前时间。

        执行流程：
        1. 获取 datetime.now()
        2. 按 format 格式化
        3. 计算 Unix 时间戳
        4. 获取星期信息
        5. 返回完整的时间信息

        Args:
            format: strftime 格式字符串

        Returns:
            ToolResult: 格式化的时间信息（包含格式化时间、Unix 时间戳、星期、ISO 格式）
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
