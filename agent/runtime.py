"""
============================================================
Agent Runtime — 核心 ReAct 循环引擎
============================================================

这是整个 Agent 系统的心脏，负责协调 LLM、工具、会话、记忆之间的交互。

ReAct 循环流程：
    用户输入 → LLM推理(带工具定义) → 有工具调用?
        ├── 是 → 执行工具 → 结果加入历史 → 继续 LLM 推理
        └── 否 → 输出文本回复 → 结束本轮

兜底机制（保障系统不会无限运行）：
    1. 最大步数限制（默认 10 步）→ 强制总结终止
    2. 连续失败检测（默认 3 次）→ 放弃工具调用，直接文本回复
    3. Token 限额检查 → 超出时强制压缩/清空历史
    4. 对话记忆压缩 → 消息数超过 50 条时自动摘要
    5. 线程安全保存 → 使用 threading.Lock 保护会话文件写入

设计原则：
    - 不假设 LLM 或工具一定能成功（所有调用都有 try/except）
    - 不假设用户输入一定有效（空输入跳过）
    - 不假设会话文件一定完整（加载失败自动重建）
    - 每步都保存状态（崩溃时最多丢失一步数据）
"""

import json
import asyncio
from pathlib import Path
from agent.llm import LLMClient
from agent.session import Session, SessionManager
from agent.trace import TraceLogger
from agent.memory import ConversationMemory
from agent.tools import get_tool_by_name, get_all_tool_schemas, ALL_TOOLS
from agent.tools.base import ToolContext
from agent.tool_executor import execute_async
from agent import config
from typing import AsyncGenerator


class AgentRuntime:
    """
    Agent 运行时 — 核心 ReAct 循环引擎。

    负责协调 LLM、工具、会话管理器之间的交互。
    每次用户输入触发一次 run_stream() 调用，执行完整的 ReAct 循环。

    Attributes:
        llm: LLM 客户端实例（封装 OpenAI API 调用）
        session: 当前会话（消息历史 + 元数据）
        session_manager: 会话管理器（持久化到文件）
        trace: 执行追踪日志（控制台 + JSONL 文件）
        memory: 对话记忆管理器（长对话自动压缩）
        max_steps: 单轮最大步数（工具调用次数），默认 10
        tool_schemas: 所有工具的 OpenAI schema（传递给 LLM 的工具定义）

    使用示例：
        runtime = AgentRuntime(username="zhangsan")

        # 流式调用（逐 token 输出）
        for event in runtime.run_stream("搜索 Python 教程"):
            if event["type"] == "text_chunk":
                print(event["data"], end="")
    """

    def __init__(
        self,
        max_steps: int = 10,
        session_id: str | None = None,
        trace: TraceLogger | None = None,
        memory_max_messages: int = 50,
        memory_keep_recent: int = 20,
        username: str | None = None,
    ):
        """
        初始化 Agent Runtime。

        初始化顺序很重要（已多次优化）：
        1. 计算用户目录（用于数据隔离）
        2. 初始化会话管理器 + 记忆管理器
        3. 确定会话（加载已有或创建新会话）
        4. 初始化 TraceLogger（使用实际的 session_id，确保日志文件名正确）
        5. 设置 todo_manager 的用户作用域
        6. 初始化 LLM 客户端
        7. 获取工具 schema
        8. 初始化 Token 追踪

        Args:
            max_steps: 单轮对话中允许的最大工具调用步数，默认 10。
                       超过此限制会强制终止，防止无限循环消耗 token。
            session_id: 要恢复的会话 ID，为 None 则创建新会话。
            trace: 日志追踪器，为 None 则自动创建（建议不传，让 Runtime 自动管理）。
            memory_max_messages: 对话记忆压缩阈值，默认 50。
                                 当消息数超过此值时自动压缩早期历史。
            memory_keep_recent: 压缩后保留的最近消息数，默认 20。
            username: 当前用户名，为 None 时使用全局路径（向后兼容）。
        """
        self.max_steps = max_steps
        self.username = username
        self._save_lock = asyncio.Lock()

        # 获取用户目录
        user_dir = None
        if username:
            from agent.user_manager import UserManager
            um = UserManager()
            user_dir = um.get_user_dir(username)

        # ================================================================
        # 先初始化会话管理器 + 记忆管理器
        # ================================================================
        self.session_manager = SessionManager(user_dir=user_dir)
        self.memory = ConversationMemory(
            max_messages=memory_max_messages,
            keep_recent=memory_keep_recent,
        )

        # ================================================================
        # 先确定会话（加载已有或创建新会话）
        # ================================================================
        if session_id:
            self.session = self.session_manager.load_session(session_id)
            if self.session:
                print(f"🔧 [SYSTEM] 恢复会话: {session_id}")
            else:
                print(f"🔧 [SYSTEM] 会话 {session_id} 不存在，创建新会话")
                self.session = self.session_manager.create_session()
        else:
            self.session = self.session_manager.create_session()

        # ================================================================
        # 再初始化 TraceLogger — 使用实际的 session_id
        # 确保日志文件命名与会话 ID 一致
        # ================================================================
        self.trace = trace or TraceLogger(
            session_id=self.session.session_id,
            user_dir=user_dir,
        )

        # 设置 todo_manager 的用户作用域
        self._init_todo_scope(user_dir)

        self.trace.log_system(f"Agent Runtime 初始化完成 | Session: {self.session.session_id} | 最大步数: {max_steps}")


        # 初始化 LLM 客户端
        self.llm = LLMClient(trace=self.trace)

        # 获取所有工具的 schema（用于传递给 LLM）
        self.tool_schemas = get_all_tool_schemas()

        # 连续失败检测
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3

        # Token 限额追踪（从已保存的会话中恢复用量）
        self._total_tokens_used = self.session.total_tokens_used
        self._token_limit = config.get_total_token_limit()
        self._token_limit_enabled = config.is_token_limit_enabled()
        if self._token_limit > 0:
            self.trace.log_system(f"Token 限额已启用: 累计上限 {self._token_limit} tokens")
        if self._total_tokens_used > 0:
            self.trace.log_system(
                f"从会话恢复 token 用量: {self._total_tokens_used} tokens, "
                f"共 {len(self.session.token_log)} 条记录"
            )

        # 重置所有工具的配额计数器（新会话从零开始）
        self._reset_all_tool_quotas()

    @property
    def session_id(self) -> str:
        """获取当前会话 ID"""
        return self.session.session_id

    # ============================================================
    # Token 限额管理
    # ============================================================

    def _update_token_usage(self, step: int = 0, tool_name: str = "") -> None:
        """
        从 LLM 客户端更新累计 token 用量，并记录到 session 和 trace

        Args:
            step: 当前 ReAct 步数
            tool_name: 如果本次消耗是工具调用相关，传入工具名称
        """
        if self.llm.last_usage:
            used = self.llm.last_usage.get("total_tokens", 0)
            prompt_tokens = self.llm.last_usage.get("prompt_tokens", 0)
            completion_tokens = self.llm.last_usage.get("completion_tokens", 0)
            if used > 0:
                self._total_tokens_used += used
                # 记录到 session
                label = f"LLM推理{' -> ' + tool_name if tool_name else ''}"
                self.session.add_token_usage(
                    step=step, tokens=used, label=label, tool_name=tool_name
                )
                self.trace.log_step_token(
                    step=step, tokens=used, cumulative=self._total_tokens_used,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    tool_name=tool_name,
                )

    def _check_token_limit(self) -> bool:
        """
        检查 token 限额是否已超限

        Returns:
            True = 可以继续（未超限或不限制）
            False = 已超限，需要强制干预
        """
        if not self._token_limit_enabled or self._token_limit <= 0:
            return True
        if self._total_tokens_used >= self._token_limit:
            self.trace.log_error(
                f"Token 限额已达: {self._total_tokens_used}/{self._token_limit}，"
                f"将强制压缩对话历史"
            )
            return False
        return True

    def _enforce_token_limit(self) -> None:
        """
        强制压缩对话历史以降低 token 消耗

        当累计 token 超过限额时：
        1. 清空非关键的消息历史
        2. 在 system prompt 中添加提示
        """
        # 清空除 system 外的所有消息
        old_count = len(self.session.messages)
        self.session.messages = [
            m for m in self.session.messages
            if m["role"] == "system"
        ]
        # 添加提示
        reminder = (
            f"[系统提示：累计 token 使用量已达 {self._total_tokens_used}/{self._token_limit}，"
            f"已清空之前的对话历史以节省 token。如果需要之前的上下文，请重述你的需求。]"
        )
        self.session.add_message("system", reminder)
        self.trace.log_system(
            f"Token 限额强制执行: 清空 {old_count} 条消息中的 {old_count - 1} 条"
        )

    # 注意：删除了同步非流式 run() 方法。
    # 所有调用场景统一使用 run_stream() 流式方法。
    # 流式 Generator 可通过收集事件拼回完整字符串，覆盖原 run() 的所有用途。

    async def run_stream(self, user_input: str) -> AsyncGenerator[dict, None]:
        """
        流式处理用户输入 — 逐步 yield 事件供前端实时展示（异步流式模式）。

        与 run() 的区别：
        - run() 使用非流式 chat()，所有 LLM 调用等待完整响应
        - run_stream() 使用 chat_stream_detect_tools_async()，边输出文本边检测工具调用
        - 异步版本使用 asyncio 而非线程，避免 GIL 限制，支持并发工具执行

        事件类型：
            text_chunk  — LLM 文本片段（逐 token）
            tool_call   — LLM 请求调用工具
            tool_result — 工具执行结果（按完成顺序逐条输出）
            step        — 当前步数信息
            ask_user    — 工具需要向用户提问（多轮交互）
            done        — 本轮处理完成
            error       — 错误信息

        工作原理：
        1. 将用户输入加入历史并持久化
        2. 检查记忆压缩（同步调用 LLM 时使用 asyncio.to_thread）
        3. ReAct 主循环（LLM 流式调用 + 异步并发工具执行）
        4. 完成后 yield done

        Args:
            user_input: 用户输入文本

        Yields:
            dict: 事件字典
        """
        # 记录用户输入
        self.trace.log_user_input(user_input)
        self.session.add_message("user", user_input)
        await self._save()  # 用户输入后立即持久化

        # ============================================================
        # 对话记忆压缩检查
        # ============================================================
        if self.memory.should_compress(self.session.messages):
            self.trace.log_system("对话历史过长，正在压缩早期消息...")
            # compress 内部调用 llm.chat()（同步），在线程中执行避免阻塞事件循环
            loop = asyncio.get_event_loop()
            compressed = await loop.run_in_executor(
                None, self.memory.compress, self.session.messages, self.llm
            )
            self.session.messages = compressed
            self.trace.log_system(f"压缩完成: {len(compressed)} 条消息")

        # ============================================================
        # ReAct 主循环 — 全部使用异步流式 API
        # ============================================================
        step = 0
        while step < self.max_steps:
            step += 1
            self.trace.log_step(step, self.max_steps, "调用 LLM 推理（异步流式）")
            yield {"type": "step", "data": {"step": step, "max_steps": self.max_steps}}

            # ============================================================
            # Token 限额检查：如果超限则强制压缩对话
            # ============================================================
            if not self._check_token_limit():
                self._enforce_token_limit()
                if not self._check_token_limit():
                    error_msg = f"Token 使用量 {self._total_tokens_used} 已超过限额 {self._token_limit}"
                    self.trace.log_error(error_msg)
                    yield {"type": "error", "data": "token 限额已耗尽，请开始新的会话。"}
                    yield {"type": "done"}
                    return

            try:
                # 异步流式调用 LLM，边输出文本边检测工具调用
                collected_content = []
                tool_calls = None

                async for event in self.llm.chat_stream_detect_tools_async(
                    messages=self.session.get_messages(),
                    tools=self.tool_schemas,
                ):
                    if event["type"] == "content":
                        collected_content.append(event["data"])
                        yield {"type": "text_chunk", "data": event["data"]}
                    elif event["type"] == "tool_calls_done":
                        tool_calls = event["data"]

                # 更新 token 累计用量（含步数和日志记录）
                self._update_token_usage(step=step)
            except Exception as e:
                error_msg = f"LLM 调用出错: {e}"
                self.trace.log_error(error_msg)
                yield {"type": "error", "data": error_msg}
                self.session.add_message("assistant", f"抱歉，我遇到了一些问题: {e}")
                await self._save()
                yield {"type": "done"}
                return

            # ============================================================
            # 处理流式结果
            # ============================================================
            if tool_calls:
                # ---- 有工具调用 ----
                full_content = "".join(collected_content)

                # 将 assistant 的响应（包含 tool_calls）添加到历史
                self.session.add_message(
                    "assistant",
                    full_content,
                    tool_calls=tool_calls,
                )
                await self._save()  # 立即持久化，防止窗口期丢失

                # ============================================================
                # 异步并发执行所有工具调用 + 流式事件输出
                # 使用 asyncio.wait(FIRST_COMPLETED) 替代 as_completed，
                # 避免 as_completed 内部 _wait_for_one 协程悬挂导致 RuntimeWarning
                # ============================================================
                task_map = {
                    asyncio.create_task(
                        self._execute_single_tool_async(tc, step)
                    ): tc for tc in tool_calls
                }

                pending = set(task_map.keys())
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for coro in done:
                        tc = task_map[coro]
                        try:
                            data = await coro
                        except Exception as e:
                            data = {
                                "tc_id": tc.get("id", ""),
                                "func_name": tc.get("function", {}).get("name", "?"),
                                "func_args": tc.get("function", {}).get("arguments", "{}"),
                                "success": False,
                                "result_text": f"工具执行异步任务异常: {e}",
                                "elapsed_ms": 0,
                                "truncated": False,
                                "error": str(e),
                                "tool_not_found": False,
                            }

                        # 记录 trace 日志
                        self.trace.log_tool_call(data["func_name"], data["func_args"], step)
                        self.trace.log_tool_result(
                            data["func_name"], data["success"], data["result_text"], step,
                            elapsed_ms=data["elapsed_ms"],
                            truncated=data["truncated"],
                        )

                        # 将工具结果添加到对话历史
                        self.session.add_message("tool", data["result_text"], tool_call_id=data["tc_id"])

                        # 流式事件：先输出工具调用信息，再输出结果
                        yield {"type": "tool_call", "data": {"tool": data["func_name"], "args": data["func_args"]}}
                        yield {
                            "type": "tool_result",
                            "data": {
                                "tool": data["func_name"],
                                "success": data["success"],
                                "result": data["result_text"][:300],
                            },
                        }

                        # ============================================================
                        # 多轮交互：工具需要向用户提问
                        # ============================================================
                        ask_question = data.get("ask_user")
                        if ask_question:
                            self.trace.log_system(f"工具请求用户输入: {ask_question}")
                            yield {
                                "type": "ask_user",
                                "data": {
                                    "tool": data["func_name"],
                                    "question": ask_question,
                                },
                            }
                            # 取消其余未完成的任务
                            for t in pending:
                                t.cancel()
                            # 等待被取消的任务真正完成清理
                            if pending:
                                await asyncio.wait(pending)
                            # 此时流结束，下一轮用户输入会作为答案
                            yield {"type": "done"}
                            return

                        # 每次工具结果写入后立即持久化会话
                        await self._save()

                # 继续循环，让 LLM 根据工具结果继续推理
                continue

            else:
                # ---- 没有工具调用，已通过流式输出完成回复 ----
                full_content = "".join(collected_content)

                self.trace.log_step(step, self.max_steps, "LLM 给出最终回复（异步流式）")

                # 将完整回复添加到历史
                self.session.add_message("assistant", full_content)

                # 保存会话
                await self._save()
                yield {"type": "done"}
                return

        # ============================================================
        # 超过最大步数限制
        # ============================================================
        self.trace.log_error(f"达到最大步数限制 ({self.max_steps})，强制终止")
        yield {"type": "error", "data": f"达到最大步数限制 ({self.max_steps})"}

        # 尝试让 LLM 给出总结性回复
        self.session.add_message(
            "user",
            "你已经执行了很多步骤，请直接给出你的最终结论或总结。",
        )
        try:
            async for event in self.llm.chat_stream_detect_tools_async(
                messages=self.session.get_messages(),
                tools=None,  # 不再传递工具，强制文本回复
            ):
                if event["type"] == "content":
                    yield {"type": "text_chunk", "data": event["data"]}
            self._update_token_usage()
        except Exception:
            final = "抱歉，执行步骤过多，无法继续处理。请简化问题后重试。"
            yield {"type": "text_chunk", "data": final}

        await self._save()
        yield {"type": "done"}

    def _build_context(self) -> ToolContext:
        """
        构建工具执行上下文。

        每次工具调用前创建，注入当前用户、会话信息到工具中。
        上下文会传递给 tool.safe_execute()，供权限检查、路径校验等使用。

        上下文包含：
        - username: 当前操作用户名（用于数据隔离）
        - user_dir: 用户数据目录（用于文件操作路径检查）
        - session_id: 当前会话 ID（用于日志追踪）
        - runtime_ref: Runtime 实例引用（供高级工具获取全局状态）
        """
        # 获取用户目录
        user_dir = None
        if self.username:
            from agent.user_manager import UserManager
            um = UserManager()
            user_dir = um.get_user_dir(self.username)

        return ToolContext(
            username=self.username or "",
            user_dir=user_dir,
            session_id=self.session.session_id,
            runtime_ref=self,
        )

    async def _execute_single_tool_async(self, tc: dict, step: int) -> dict:
        """
        执行单个工具调用（异步版本）

        将解析、查找、执行集中到一个异步方法中，返回结果字典。
        构建 ToolContext 并传递给工具，支持鉴权和上下文感知。
        内部使用 tool_executor.execute_async 进行实际执行，支持：
        - 超时控制
        - 自动重试
        - 异步执行（工具重写了 execute_async 时）
        - 同步回退（工具仅有 execute 时在线程池中执行）

        Args:
            tc: LLM 返回的 tool_call 字典
            step: 当前 ReAct 步数

        Returns:
            dict: {
                "tc_id": str,
                "func_name": str,
                "func_args": str,
                "success": bool,
                "result_text": str,
                "elapsed_ms": float,
                "truncated": bool,
                "error": str | None,
                "tool_not_found": bool,
            }
        """
        # ================================================================
        # 防御性解析 tool_calls 结构
        # ================================================================
        try:
            tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
            func_info = tc.get("function", {}) if isinstance(tc, dict) else {}
            func_name = func_info.get("name", "") if isinstance(func_info, dict) else ""
            func_args = func_info.get("arguments", "{}") if isinstance(func_info, dict) else "{}"

            if not tc_id:
                raise ValueError("tool_call 缺少 'id' 字段")
            if not func_name:
                raise ValueError("tool_call 缺少 'function.name' 字段")

        except (KeyError, TypeError, ValueError) as e:
            return {
                "tc_id": "",
                "func_name": "?",
                "func_args": "{}",
                "success": False,
                "result_text": f"[INVALID_PARAMS] 解析 tool_call 失败: {e}",
                "elapsed_ms": 0,
                "truncated": False,
                "error": str(e),
                "tool_not_found": False,
            }

        # ================================================================
        # 查找并执行工具
        # ================================================================
        tool = get_tool_by_name(func_name)

        if tool is None:
            available = ", ".join(t.name for t in ALL_TOOLS)
            return {
                "tc_id": tc_id,
                "func_name": func_name,
                "func_args": func_args,
                "success": False,
                "result_text": f"[TOOL_NOT_FOUND] 未知工具: '{func_name}'。可用工具: {available}",
                "elapsed_ms": 0,
                "truncated": False,
                "error": None,
                "tool_not_found": True,
            }

        # 构建上下文并异步执行
        context = self._build_context()
        result = await execute_async(tool, func_args, context=context)

        # ================================================================
        # 工具执行后：更新会话级 token 累计用量
        # ================================================================
        if result.success:
            tool_result_tokens = config.get_estimated_tokens(result.result)
            if tool_result_tokens > 0:
                self._total_tokens_used += tool_result_tokens
                self.session.add_token_usage(
                    step=step,
                    tokens=tool_result_tokens,
                    label=f"工具结果: {func_name}",
                    tool_name=func_name,
                )

        return {
            "tc_id": tc_id,
            "func_name": func_name,
            "func_args": func_args,
            "success": result.success,
            "result_text": result.result if result.success else f"[ERROR] {result.error}",
            "elapsed_ms": result.metadata.get("elapsed_ms", 0),
            "truncated": result.metadata.get("truncated", False),
            "error": None,
            "tool_not_found": False,
            "ask_user": result.ask_user,
            "quota_used": result.metadata.get("quota_used", 0),
            "quota_limit": result.metadata.get("quota_limit", 0),
        }

    async def _save(self) -> None:
        """保存当前会话状态（异步安全，带锁）"""
        async with self._save_lock:
            self.session_manager.save_session(self.session)
            self.trace.log_system(f"会话已保存: {self.session.session_id}")

    async def save_session(self) -> None:
        """
        公开的会话保存方法（异步版本，供外部调用）

        用于 main.py 中在切换/新建/退出会话前确保数据持久化。
        """
        await self._save()

    def _init_todo_scope(self, user_dir: Path):
        """设置 todo_manager 的用户作用域，任务数据存储到 user_dir/task/（向后兼容）"""
        from agent.tools import get_tool_by_name
        todo = get_tool_by_name("todo_manager")
        if todo and hasattr(todo, "set_user_dir") and user_dir:
            todo.set_user_dir(user_dir)

    @staticmethod
    def _reset_all_tool_quotas():
        """
        重置所有工具的配额计数器。

        在新会话创建或切换会话时调用，使每个工具的 quota 从零开始计数。
        静态方法，直接操作工具注册表中的实例。
        """
        from agent.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            tool.reset_quota()

    def get_trace_summary(self) -> str:
        """
        获取执行追踪摘要（含工具调用统计）

        统计内容包括：
        - LLM 调用次数
        - 工具调用总次数
        - 各工具调用次数 / 成功数 / 失败数 / 平均耗时
        """
        entries = self.trace.get_all_entries()

        tool_calls = [e for e in entries if e["level"] == "tool_call"]
        tool_results = [e for e in entries if e["level"] == "tool_result"]
        errors = [e for e in entries if e["level"] == "error"]
        llm_calls = [e for e in entries if e["level"] == "llm_request"]

        # 统计各工具调用情况
        tool_stats = {}
        for r in tool_results:
            name = r.get("tool_name", "?")
            if name not in tool_stats:
                tool_stats[name] = {"calls": 0, "success": 0, "fail": 0, "total_ms": 0}
            tool_stats[name]["calls"] += 1
            if r.get("success"):
                tool_stats[name]["success"] += 1
            else:
                tool_stats[name]["fail"] += 1
            tool_stats[name]["total_ms"] += r.get("elapsed_ms", 0)

        # Token 使用统计
        token_entries = [e for e in entries if e["level"] == "token_usage"]

        lines = [
            f"Trace 摘要: {len(llm_calls)} 次 LLM 调用, "
            f"{len(tool_calls)} 次工具调用, "
            f"{len(errors)} 个错误, "
            f"总 Token: {self._total_tokens_used}"
        ]

        # 展示最近几步的 token 使用明细
        if token_entries:
            lines.append("")
            lines.append("Token 使用明细（最近 5 步）:")
            for te in token_entries[-5:]:
                t = te.get("tokens", 0)
                c = te.get("cumulative", 0)
                s = te.get("step", "?")
                tn = te.get("tool_name", "")
                tn_info = f" [{tn}]" if tn else ""
                lines.append(f"  Step {s}{tn_info}: +{t} (累计: {c})")
            # 总计
            last_cumulative = token_entries[-1].get("cumulative", self._total_tokens_used)
            lines.append(f"  会话累计 Token: {last_cumulative}")

        if tool_stats:
            lines.append("")
            lines.append("工具使用统计:")
            # 获取每个工具的配额信息
            from agent.tools import ALL_TOOLS
            tool_quota_map = {t.name: t for t in ALL_TOOLS}
            for name, stats in sorted(tool_stats.items()):
                avg_ms = stats["total_ms"] / stats["calls"] if stats["calls"] > 0 else 0
                # 附加用量信息（始终显示，有配额时显示比例）
                quota_info = ""
                t = tool_quota_map.get(name)
                if t and t.quota_used > 0:
                    if t.quota_limit > 0:
                        pct = min(100.0, round(t.quota_used / t.quota_limit * 100, 1))
                        quota_info = f" | 用量: {t.quota_used}/{t.quota_limit} ({pct}%)"
                    else:
                        quota_info = f" | 用量: {t.quota_used}（无限额）"
                lines.append(
                    f"  {name}: {stats['calls']} 次调用, "
                    f"{stats['success']} 成功/{stats['fail']} 失败, "
                    f"平均 {avg_ms:.0f}ms/次{quota_info}"
                )

        # Token 限额信息
        if self._token_limit > 0:
            pct = min(100.0, round(self._total_tokens_used / self._token_limit * 100, 1))
            lines.append(
                f"\n会话 Token 限额: {self._total_tokens_used}/{self._token_limit} ({pct}%)"
            )

        return "\n".join(lines)

    def switch_session(self, session_id: str) -> str:
        """
        切换到指定会话

        Args:
            session_id: 目标会话 ID

        Returns:
            切换结果信息
        """
        loaded = self.session_manager.load_session(session_id)
        if loaded:
            self.session = loaded
            # 切换会话时重置所有工具的配额计数
            self._reset_all_tool_quotas()
            self.trace.log_system(f"切换到会话: {session_id}，已重置工具配额")
            return f"已切换到会话 {session_id}（{len(loaded.messages)} 条消息，工具配额已重置）"
        return f"会话 {session_id} 不存在"
