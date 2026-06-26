"""
============================================================
执行追踪 / Trace 日志系统
============================================================

功能：
    记录 Agent 的完整执行过程，包括 LLM 请求/响应、工具调用/结果、
    系统事件、错误信息、步数追踪、token 消耗等。

双重输出：
    1. 控制台实时输出（带颜色 emoji 标记，人类可读）
    2. JSONL 文件持久化（每行一个 JSON 对象，机器可分析）

为什么用 JSONL 而非 JSON：
    - JSONL 是逐行追加的，不需要将整个文件读入内存再修改
    - 天然的 append-only 日志格式，不会因写入冲突损坏
    - 每行可独立解析，grep/sed 等工具可直接处理
    - 日志分析工具（如 ELK、Datadog）原生支持 JSONL

为什么同时输出到控制台：
    - 开发调试时即时看到执行过程
    - 生产环境中可以通过重定向 stdout 收集日志

记录级别：
    system      系统事件（初始化、压缩、保存）
    user        用户输入
    llm_request LLM 请求（消息数量、工具数量）
    llm_response LLM 响应（文本预览、工具名称、token 统计）
    tool_call   工具调用（名称、参数、步数）
    tool_result 工具执行结果（成功/失败、耗时、截断标记）
    error       错误信息
    step        步数信息
    token_usage Token 消耗明细
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class TraceLogger:
    """
    Agent 执行追踪日志。

    功能：
    - 控制台实时输出（带颜色和时序）
    - 文件持久化（JSON Lines 格式，便于分析）
    - 支持不同级别的日志（system/llm/tool/error）

    Attributes:
        log_file: 日志文件路径
        session_id: 当前会话 ID
        _entries: 内存中的日志条目列表
    """

    # emoji 映射表：每种日志级别对应的显示图标
    LEVEL_ICONS = {
        "system": "🔧",
        "user": "👤",
        "llm_request": "🤖",
        "llm_response": "🤖",
        "tool_call": "🔨",
        "tool_result": "🔨",
        "error": "❌",
        "step": "📍",
        "token_usage": "💰",
    }

    def __init__(self, log_dir: str | None = None, session_id: str = "default",
                 user_dir: Path | None = None):
        """
        初始化日志追踪器。

        日志目录优先级：user_dir/log > log_dir > 默认 .agent_data/logs/
        文件命名：trace_{session_id}_{timestamp}.jsonl

        Args:
            log_dir: 日志文件目录（手动指定路径）
            session_id: 会话 ID，用于日志文件命名
            user_dir: 用户数据目录，有则日志写到 user_dir/log/
                      这是最推荐的模式，实现数据隔离
        """
        self.session_id = session_id
        self._entries: list[dict] = []

        # 设置日志目录
        # 优先级：user_dir > log_dir > 默认全局路径
        if user_dir:
            log_dir_path = user_dir / "log"
        elif log_dir is None:
            log_dir_path = Path(__file__).parent.parent / ".agent_data" / "logs"
        else:
            log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # 日志文件名包含时间戳，避免同 session 多次运行时文件覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir_path / f"trace_{session_id}_{timestamp}.jsonl"

        self._print(f"📋 Trace 日志已启动: {self.log_file}")

    def _print(self, message: str) -> None:
        """
        控制台输出（写入 stderr，不污染 stdout）。

        为什么用 stderr：
        - stdout 留给 LLM 的流式回复文本（main.py 直接 print 到 stdout）
        - trace 日志（💰 Token、📍 [STEP] 等）走 stderr，
          两者在终端上看起来一样，但重定向 stdout 时可分离
        - 例如：python main.py > response.txt 只保存 LLM 回复，不含日志
        """
        print(message, file=sys.stderr)

    def _write_entry(self, entry: dict) -> None:
        """
        写入一条日志条目（同时写入内存和文件）。

        文件写入使用追加模式（"a"），符合 JSONL 格式要求。
        每行末尾的换行符确保文件可被逐行读取。

        错误处理：
        写入日志文件失败不会影响主流程——trace 模块只负责记录，
        不应对调用方产生副作用。因此所有 IO 异常在此被吞掉并打印警告。
        """
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self._entries.append(entry)

        # 追加写入 JSONL 文件
        # 写入失败不抛异常，只打印警告（trace 日志不能阻塞主流程）
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (IOError, OSError, TypeError, ValueError) as e:
            self._print(f"⚠️ [TRACE] 日志写入失败: {e}")

    def log_system(self, message: str) -> None:
        """记录系统级事件（如初始化完成、会话切换、记忆压缩等）。"""
        self._print(f"🔧 [SYSTEM] {message}")
        self._write_entry({"level": "system", "message": message})

    def log_user_input(self, message: str) -> None:
        """记录用户输入。"""
        self._print(f"👤 [USER] {message}")
        self._write_entry({"level": "user", "message": message})

    def log_llm_request(self, messages: list[dict], tools: list[dict] | None = None) -> None:
        """
        记录发送给 LLM 的请求。

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
                {"role": m.get("role"), "content_preview": str(m.get("content", ""))[:500]}
                for m in messages
            ],
        })

    def log_llm_response(self, response: dict, usage: dict | None = None) -> None:
        """
        记录 LLM 的响应。

        Args:
            response: LLM 响应消息（包含 content 和/或 tool_calls）
            usage: token 使用统计（prompt_tokens, completion_tokens, total_tokens）
        """
        has_tool_calls = response.get("tool_calls") is not None
        content_text = response.get("content") or ""
        content_preview = content_text[:2000]

        if has_tool_calls:
            tool_names = [tc["function"]["name"] for tc in response["tool_calls"]]
            self._print(f"🤖 [LLM←响应] 请求调用工具: {', '.join(tool_names)}")
        else:
            self._print(f"🤖 [LLM←响应] 文本回复: {content_text}")

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
        记录工具调用。

        Args:
            tool_name: 工具名称
            arguments: 调用参数（JSON 字符串或字典）
            step: 当前 ReAct 步数
        """
        args_str = str(arguments)
        self._print(f"🔨 [TOOL→调用] Step {step}: {tool_name}({args_str})")
        self._write_entry({
            "level": "tool_call",
            "step": step,
            "tool_name": tool_name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
        })

    def log_tool_result(self, tool_name: str, success: bool, result: str, step: int,
                        elapsed_ms: float = 0, truncated: bool = False) -> None:
        """
        记录工具执行结果。

        Args:
            tool_name: 工具名称
            success: 是否执行成功
            result: 结果文本
            step: 当前 ReAct 步数
            elapsed_ms: 执行耗时（毫秒）
            truncated: 结果是否被截断
        """
        status = "✅" if success else "❌"
        time_info = f" [{elapsed_ms:.0f}ms]" if elapsed_ms else ""
        trunc_info = " [截断]" if truncated else ""

        self._print(f"{status} [TOOL←结果] Step {step}: {tool_name}{time_info}{trunc_info} → {result}")
        self._write_entry({
            "level": "tool_result",
            "step": step,
            "tool_name": tool_name,
            "success": success,
            "result_preview": result[:2000],
            "elapsed_ms": elapsed_ms,
            "truncated": truncated,
        })

    def log_error(self, message: str) -> None:
        """记录错误信息。"""
        self._print(f"❌ [ERROR] {message}")
        self._write_entry({"level": "error", "message": message})

    def log_step(self, step: int, max_steps: int, action: str) -> None:
        """
        记录步数信息。

        Args:
            step: 当前步数
            max_steps: 最大步数
            action: 当前动作描述（如"调用 LLM 推理"、"执行工具"）
        """
        self._print(f"📍 [STEP] {step}/{max_steps}: {action}")
        self._write_entry({
            "level": "step",
            "step": step,
            "max_steps": max_steps,
            "action": action,
        })

    def log_step_token(self, step: int, tokens: int, cumulative: int,
                       prompt_tokens: int = 0, completion_tokens: int = 0,
                       tool_name: str = "") -> None:
        """
        记录每一步的 token 使用量。

        用于跟踪会话整体的 token 消耗趋势，帮助判断是否需要
        压缩对话历史或触发限额检查。

        Args:
            step: 当前步数
            tokens: 本次使用的 token 数
            cumulative: 累计 token 使用量
            prompt_tokens: prompt token 数（可选）
            completion_tokens: completion token 数（可选）
            tool_name: 工具名称（可选，如果是工具调用产生的消耗）
        """
        tool_info = f" [{tool_name}]" if tool_name else ""
        detail = f"prompt: {prompt_tokens}, completion: {completion_tokens}" if prompt_tokens or completion_tokens else ""
        self._print(
            f"  💰 Token{ tool_info }: +{tokens} (累计: {cumulative})"
            + (f" [{detail}]" if detail else "")
        )
        self._write_entry({
            "level": "token_usage",
            "step": step,
            "tokens": tokens,
            "cumulative": cumulative,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_name": tool_name,
        })

    def get_all_entries(self) -> list[dict]:
        """获取所有日志条目（返回副本，防止外部修改）。"""
        return self._entries.copy()

    def summary(self) -> str:
        """生成日志摘要（调用次数统计）。"""
        tool_calls = sum(1 for e in self._entries if e["level"] == "tool_call")
        errors = sum(1 for e in self._entries if e["level"] == "error")
        llm_calls = sum(1 for e in self._entries if e["level"] == "llm_request")
        return f"Trace 摘要: {llm_calls} 次 LLM 调用, {tool_calls} 次工具调用, {errors} 个错误"
