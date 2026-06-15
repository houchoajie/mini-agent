"""
执行日志 / Trace 系统

记录 Agent 的完整执行过程，包括：
- LLM 请求与响应
- 工具调用与结果
- 系统事件（初始化、错误等）
- 步数追踪

日志同时输出到控制台和文件。
"""

import json
import os
from datetime import datetime
from pathlib import Path


class TraceLogger:
    """
    Agent 执行追踪日志

    功能：
    - 控制台实时输出（带颜色和时序）
    - 文件持久化（JSON Lines 格式，便于分析）
    - 支持不同级别的日志（system/llm/tool/error）

    Attributes:
        log_file: 日志文件路径
        session_id: 当前会话 ID
        _entries: 内存中的日志条目列表
    """

    def __init__(self, log_dir: str | None = None, session_id: str = "default"):
        """
        初始化日志追踪器

        Args:
            log_dir: 日志文件目录，默认为项目根目录下的 .agent_data/logs
            session_id: 会话 ID，用于日志文件命名
        """
        self.session_id = session_id
        self._entries: list[dict] = []

        # 设置日志目录
        if log_dir is None:
            log_dir_path = Path(__file__).parent.parent / ".agent_data" / "logs"
        else:
            log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # 日志文件名包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir_path / f"trace_{session_id}_{timestamp}.jsonl"

        self._print(f"📋 Trace 日志已启动: {self.log_file}")

    def _print(self, message: str) -> None:
        """控制台输出"""
        print(message)

    def _write_entry(self, entry: dict) -> None:
        """
        写入一条日志条目

        同时写入内存列表和文件。

        Args:
            entry: 日志条目字典
        """
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._entries.append(entry)

        # 追加写入 JSONL 文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_system(self, message: str) -> None:
        """记录系统级事件"""
        self._print(f"🔧 [SYSTEM] {message}")
        self._write_entry({"level": "system", "message": message})

    def log_user_input(self, message: str) -> None:
        """记录用户输入"""
        self._print(f"👤 [USER] {message}")
        self._write_entry({"level": "user", "message": message})

    def log_llm_request(self, messages: list[dict], tools: list[dict] | None = None) -> None:
        """
        记录发送给 LLM 的请求

        Args:
            messages: 对话消息列表
            tools: 工具定义列表
        """
        msg_count = len(messages)
        tool_count = len(tools) if tools else 0
        self._print(f"🤖 [LLM→请求] 发送 {msg_count} 条消息, {tool_count} 个工具可用")
        self._write_entry({
            "level": "llm_request",
            "message_count": msg_count,
            "tool_count": tool_count,
            "messages_summary": [
                {"role": m.get("role"), "content_preview": str(m.get("content", ""))[:100]}
                for m in messages
            ],
        })

    def log_llm_response(self, response: dict, usage: dict | None = None) -> None:
        """
        记录 LLM 的响应

        Args:
            response: LLM 响应消息
            usage: token 使用统计
        """
        has_tool_calls = response.get("tool_calls") is not None
        content_preview = (response.get("content") or "")[:200]

        if has_tool_calls:
            tool_names = [tc["function"]["name"] for tc in response["tool_calls"]]
            self._print(f"🤖 [LLM←响应] 请求调用工具: {', '.join(tool_names)}")
        else:
            self._print(f"🤖 [LLM←响应] 文本回复: {content_preview[:80]}...")

        if usage:
            self._print(f"   Token 使用: {usage.get('total_tokens', '?')} (prompt: {usage.get('prompt_tokens', '?')}, completion: {usage.get('completion_tokens', '?')})")

        self._write_entry({
            "level": "llm_response",
            "has_tool_calls": has_tool_calls,
            "content_preview": content_preview,
            "usage": usage,
        })

    def log_tool_call(self, tool_name: str, arguments: str | dict, step: int) -> None:
        """
        记录工具调用

        Args:
            tool_name: 工具名称
            arguments: 调用参数
            step: 当前步数
        """
        args_preview = str(arguments)[:200]
        self._print(f"🔨 [TOOL→调用] Step {step}: {tool_name}({args_preview})")
        self._write_entry({
            "level": "tool_call",
            "step": step,
            "tool_name": tool_name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
        })

    def log_tool_result(self, tool_name: str, success: bool, result: str, step: int) -> None:
        """
        记录工具执行结果

        Args:
            tool_name: 工具名称
            success: 是否成功
            result: 结果文本
            step: 当前步数
        """
        status = "✅" if success else "❌"
        result_preview = result[:200]
        self._print(f"{status} [TOOL←结果] Step {step}: {tool_name} → {result_preview}")
        self._write_entry({
            "level": "tool_result",
            "step": step,
            "tool_name": tool_name,
            "success": success,
            "result_preview": result_preview,
        })

    def log_error(self, message: str) -> None:
        """记录错误信息"""
        self._print(f"❌ [ERROR] {message}")
        self._write_entry({"level": "error", "message": message})

    def log_step(self, step: int, max_steps: int, action: str) -> None:
        """
        记录步数信息

        Args:
            step: 当前步数
            max_steps: 最大步数
            action: 当前动作描述
        """
        self._print(f"📍 [STEP] {step}/{max_steps}: {action}")
        self._write_entry({
            "level": "step",
            "step": step,
            "max_steps": max_steps,
            "action": action,
        })

    def get_all_entries(self) -> list[dict]:
        """获取所有日志条目"""
        return self._entries.copy()

    def summary(self) -> str:
        """生成日志摘要"""
        tool_calls = sum(1 for e in self._entries if e["level"] == "tool_call")
        errors = sum(1 for e in self._entries if e["level"] == "error")
        llm_calls = sum(1 for e in self._entries if e["level"] == "llm_request")
        return f"Trace 摘要: {llm_calls} 次 LLM 调用, {tool_calls} 次工具调用, {errors} 个错误"
