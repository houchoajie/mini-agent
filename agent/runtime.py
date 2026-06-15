"""
Agent Runtime - 核心 ReAct 循环引擎

这是整个 Agent 系统的心脏，实现了：
1. ReAct 循环：接收输入 → LLM 判断 → 工具调用 → 结果反馈 → 继续/终止
2. 最大步数限制（防止无限循环）
3. 工具调用的完整生命周期管理
4. 多轮对话中的上下文传递

ReAct 循环流程：
┌─────────────────────────────────────┐
│  用户输入                            │
│     ↓                               │
│  [LLM 推理] ← 对话历史 + 工具定义    │
│     ↓                               │
│  有工具调用？                        │
│     ├── 是 → 执行工具 → 结果加入历史 → 回到 LLM 推理 │
│     └── 否 → 输出文本回复 → 结束本轮  │
└─────────────────────────────────────┘
"""

import json
from agent.llm import LLMClient
from agent.session import Session, SessionManager
from agent.trace import TraceLogger
from agent.memory import ConversationMemory
from agent.tools import get_tool_by_name, get_all_tool_schemas
from typing import Generator


class AgentRuntime:
    """
    Agent 运行时 - 核心 ReAct 循环引擎

    负责协调 LLM、工具、会话管理器之间的交互。
    每次用户输入触发一次 run() 调用，执行完整的 ReAct 循环。

    Attributes:
        llm: LLM 客户端实例
        session: 当前会话
        session_manager: 会话管理器
        trace: 执行追踪日志
        max_steps: 单轮最大步数（工具调用次数）
        tool_schemas: 所有工具的 OpenAI schema

    使用示例：
        runtime = AgentRuntime()

        # 第一轮对话
        response = runtime.run("帮我计算 2 的 10 次方")

        # 第二轮对话（自动保持上下文）
        response = runtime.run("再帮我搜一下 Python 教程")
    """

    def __init__(
        self,
        max_steps: int = 10,
        session_id: str | None = None,
        trace: TraceLogger | None = None,
        memory_max_messages: int = 50,
        memory_keep_recent: int = 20,
    ):
        """
        初始化 Agent Runtime

        Args:
            max_steps: 单轮对话中允许的最大工具调用步数，默认 10
                       超过此限制会强制终止，防止无限循环
            session_id: 要恢复的会话 ID，为 None 则创建新会话
            trace: 日志追踪器，为 None 则自动创建
            memory_max_messages: 对话记忆压缩阈值，默认 50
                                 当消息数超过此值时自动压缩早期历史
            memory_keep_recent: 压缩后保留的最近消息数，默认 20
        """
        self.max_steps = max_steps
        self.trace = trace or TraceLogger(session_id=session_id or "default")
        self.session_manager = SessionManager()

        # 初始化对话记忆管理器（用于长对话的自动压缩）
        self.memory = ConversationMemory(
            max_messages=memory_max_messages,
            keep_recent=memory_keep_recent,
        )

        # 尝试恢复已有会话，或创建新会话
        if session_id:
            self.session = self.session_manager.load_session(session_id)
            if self.session:
                self.trace.log_system(f"恢复会话: {session_id}")
            else:
                self.trace.log_system(f"会话 {session_id} 不存在，创建新会话")
                self.session = self.session_manager.create_session()
        else:
            self.session = self.session_manager.create_session()

        self.trace.log_system(f"Agent Runtime 初始化完成 | Session: {self.session.session_id} | 最大步数: {max_steps}")

        # 初始化 LLM 客户端
        self.llm = LLMClient(trace=self.trace)

        # 获取所有工具的 schema（用于传递给 LLM）
        self.tool_schemas = get_all_tool_schemas()

    @property
    def session_id(self) -> str:
        """获取当前会话 ID"""
        return self.session.session_id

    def run(self, user_input: str) -> str:
        """
        处理用户输入 - 执行完整的 ReAct 循环

        这是 Agent 的核心方法，执行流程如下：

        1. 将用户输入添加到对话历史
        2. 调用 LLM 获取响应
        3. 如果 LLM 请求调用工具：
           a. 解析工具名称和参数
           b. 查找并执行对应工具
           c. 将工具结果添加到对话历史
           d. 步数 +1，检查是否超过限制
           e. 回到步骤 2
        4. 如果 LLM 直接回复文本：
           a. 将回复添加到对话历史
           b. 返回回复内容
        5. 如果超过最大步数：
           a. 强制生成终止回复

        Args:
            user_input: 用户输入文本

        Returns:
            Agent 的最终回复文本
        """
        # 记录用户输入
        self.trace.log_user_input(user_input)
        self.session.add_message("user", user_input)

        # ============================================================
        # 对话记忆压缩检查
        # 当消息历史过长时，自动压缩早期消息为摘要
        # ============================================================
        if self.memory.should_compress(self.session.messages):
            self.trace.log_system("对话历史过长，正在压缩早期消息...")
            compressed = self.memory.compress(self.session.messages, self.llm)
            self.session.messages = compressed
            self.trace.log_system(f"压缩完成: {len(compressed)} 条消息")

        # ============================================================
        # ReAct 主循环
        # ============================================================
        step = 0
        while step < self.max_steps:
            step += 1
            self.trace.log_step(step, self.max_steps, "调用 LLM 推理")

            try:
                # 调用 LLM，传递对话历史和工具定义
                response = self.llm.chat(
                    messages=self.session.get_messages(),
                    tools=self.tool_schemas,
                )
            except Exception as e:
                error_msg = f"LLM 调用出错: {e}"
                self.trace.log_error(error_msg)
                self.session.add_message("assistant", f"抱歉，我遇到了一些问题: {e}")
                self._save()
                return f"抱歉，我遇到了一些问题: {e}"

            # ============================================================
            # 检查 LLM 是否请求调用工具
            # ============================================================
            tool_calls = response.get("tool_calls")

            if tool_calls:
                # ---- 有工具调用请求 ----

                # 先将 assistant 的响应（包含 tool_calls）添加到历史
                # 注意：content 可能为 None，需要处理
                self.session.add_message(
                    "assistant",
                    response.get("content") or "",
                    tool_calls=response["tool_calls"],
                )

                # 逐个执行工具调用
                for tc in tool_calls:
                    tc_id = tc["id"]
                    func_name = tc["function"]["name"]
                    func_args = tc["function"]["arguments"]

                    self.trace.log_tool_call(func_name, func_args, step)

                    # 查找工具
                    tool = get_tool_by_name(func_name)
                    if tool is None:
                        # 工具不存在
                        result_text = f"[ERROR] 未知工具: {func_name}"
                        self.trace.log_tool_result(func_name, False, result_text, step)
                    else:
                        # 执行工具
                        result = tool.safe_execute(func_args)
                        result_text = result.to_string()
                        self.trace.log_tool_result(func_name, result.success, result_text, step)

                    # 将工具结果添加到对话历史
                    # role 必须是 "tool"，且包含 tool_call_id
                    self.session.add_message(
                        "tool",
                        result_text,
                        tool_call_id=tc_id,
                    )

                # 继续循环，让 LLM 根据工具结果继续推理
                continue

            else:
                # ---- 没有工具调用，LLM 直接回复 ----
                content = response.get("content") or ""

                self.trace.log_step(step, self.max_steps, f"LLM 给出最终回复")

                # 将回复添加到历史
                self.session.add_message("assistant", content)

                # 保存会话
                self._save()

                return content

        # ============================================================
        # 超过最大步数限制
        # ============================================================
        self.trace.log_error(f"达到最大步数限制 ({self.max_steps})，强制终止")

        # 尝试让 LLM 给出总结性回复
        self.session.add_message(
            "user",
            "你已经执行了很多步骤，请直接给出你的最终结论或总结。",
        )
        try:
            response = self.llm.chat(
                messages=self.session.get_messages(),
                tools=None,  # 不再传递工具，强制文本回复
            )
            final = response.get("content") or "抱歉，我执行了太多步骤但未能得出结论。"
        except Exception:
            final = "抱歉，执行步骤过多，无法继续处理。请简化问题后重试。"

        self.session.add_message("assistant", final)
        self._save()
        return final

    def run_stream(self, user_input: str) -> Generator[dict, None, None]:
        """
        流式处理用户输入 - 逐步 yield 事件用于实时展示

        相比 run() 方法，支持逐 token 输出 LLM 回复。
        用于 CLI 场景的实时打印，用户可即时看到 AI "打字" 效果。

        工作原理：
        1. 记忆压缩检查（同 run()）
        2. ReAct 主循环
           - 工具调用阶段：使用非流式 chat() 检测，yield tool_call/tool_result 事件
           - 文本回复阶段：使用 chat_stream() 逐 token 流式输出，yield text_chunk 事件
        3. 完成后 yield done 事件

        Args:
            user_input: 用户输入文本

        Yields:
            dict: 事件字典，包含 type 和 data 字段
                - type: "text_chunk", data: str - LLM 文本片段
                - type: "tool_call", data: dict - 工具调用信息
                - type: "tool_result", data: dict - 工具执行结果
                - type: "step", data: dict - 步数信息
                - type: "done" - 完成
                - type: "error", data: str - 错误信息

        使用示例：
            for event in runtime.run_stream(user_input):
                if event["type"] == "text_chunk":
                    print(event["data"], end="", flush=True)
        """
        # 记录用户输入
        self.trace.log_user_input(user_input)
        self.session.add_message("user", user_input)

        # ============================================================
        # 对话记忆压缩检查
        # ============================================================
        if self.memory.should_compress(self.session.messages):
            self.trace.log_system("对话历史过长，正在压缩早期消息...")
            compressed = self.memory.compress(self.session.messages, self.llm)
            self.session.messages = compressed
            self.trace.log_system(f"压缩完成: {len(compressed)} 条消息")

        # ============================================================
        # ReAct 主循环
        # ============================================================
        step = 0
        while step < self.max_steps:
            step += 1
            self.trace.log_step(step, self.max_steps, "调用 LLM 推理")
            yield {"type": "step", "data": {"step": step, "max_steps": self.max_steps}}

            try:
                # 先使用非流式调用判断 LLM 意图（工具调用还是文本回复）
                response = self.llm.chat(
                    messages=self.session.get_messages(),
                    tools=self.tool_schemas,
                )
            except Exception as e:
                error_msg = f"LLM 调用出错: {e}"
                self.trace.log_error(error_msg)
                yield {"type": "error", "data": error_msg}
                self.session.add_message("assistant", f"抱歉，我遇到了一些问题: {e}")
                self._save()
                yield {"type": "done"}
                return

            # ============================================================
            # 检查 LLM 是否请求调用工具
            # ============================================================
            tool_calls = response.get("tool_calls")

            if tool_calls:
                # ---- 有工具调用请求 ----
                self.session.add_message(
                    "assistant",
                    response.get("content") or "",
                    tool_calls=response["tool_calls"],
                )

                for tc in tool_calls:
                    tc_id = tc["id"]
                    func_name = tc["function"]["name"]
                    func_args = tc["function"]["arguments"]

                    self.trace.log_tool_call(func_name, func_args, step)

                    # yield 工具调用事件
                    try:
                        args_parsed = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except json.JSONDecodeError:
                        args_parsed = {"raw": func_args}
                    yield {"type": "tool_call", "data": {"tool": func_name, "args": args_parsed}}

                    # 查找并执行工具
                    tool = get_tool_by_name(func_name)
                    if tool is None:
                        result_text = f"[ERROR] 未知工具: {func_name}"
                        self.trace.log_tool_result(func_name, False, result_text, step)
                        yield {"type": "tool_result", "data": {"tool": func_name, "success": False, "result": result_text}}
                    else:
                        result = tool.safe_execute(func_args)
                        result_text = result.to_string()
                        self.trace.log_tool_result(func_name, result.success, result_text, step)
                        yield {"type": "tool_result", "data": {"tool": func_name, "success": result.success, "result": result_text[:300]}}

                    # 将工具结果添加到对话历史
                    self.session.add_message(
                        "tool",
                        result_text,
                        tool_call_id=tc_id,
                    )

                # 继续循环
                continue

            else:
                # ---- 没有工具调用，LLM 直接回复 - 流式输出 ----
                self.trace.log_step(step, self.max_steps, f"LLM 给出最终回复（流式输出）")

                # 使用流式 API 逐 token 输出
                collected_content = []
                for chunk in self.llm.chat_stream(
                    self.session.get_messages(),
                    tools=self.tool_schemas,
                ):
                    collected_content.append(chunk)
                    yield {"type": "text_chunk", "data": chunk}

                full_content = "".join(collected_content)

                # 将完整回复添加到历史
                self.session.add_message("assistant", full_content)

                # 保存会话
                self._save()
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
            response = self.llm.chat(
                messages=self.session.get_messages(),
                tools=None,
            )
            final = response.get("content") or "抱歉，我执行了太多步骤但未能得出结论。"
        except Exception:
            final = "抱歉，执行步骤过多，无法继续处理。请简化问题后重试。"

        self.session.add_message("assistant", final)
        self._save()
        yield {"type": "text_chunk", "data": final}
        yield {"type": "done"}

    def _save(self) -> None:
        """保存当前会话状态"""
        self.session_manager.save_session(self.session)
        self.trace.log_system(f"会话已保存: {self.session.session_id}")

    def get_trace_summary(self) -> str:
        """获取执行追踪摘要"""
        return self.trace.summary()

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
            self.trace.log_system(f"切换到会话: {session_id}")
            return f"已切换到会话 {session_id}（{len(loaded.messages)} 条消息）"
        return f"会话 {session_id} 不存在"
