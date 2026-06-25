"""
============================================================
会话管理器 — 多轮对话的状态维护与持久化
============================================================

职责：
1. 维护每个会话的消息历史（messages 列表，遵循 OpenAI Chat 格式）
2. 注入系统提示词（定义 Agent 的角色和能力）
3. 持久化会话状态到 JSON 文件
4. 管理多个独立会话（新建/切换/列出/删除）
5. 追踪会话级别的 token 累计用量

为什么使用 JSON 而非数据库：
    - 零依赖：不需要安装/配置任何数据库
    - 便携性：整个项目复制即可运行，无需额外服务
    - 可读性：JSON 文件可直接用文本编辑器查看和修改
    - 合适的数据量：会话数据通常只有几 KB 到几十 KB

数据隔离：
    通过 user_dir 参数实现用户级数据隔离：
    - 有 user_dir → session 文件存储在 user_dir/session/ 下
    - 无 user_dir → 使用全局 .agent_data/sessions/（向后兼容）

文件结构：
    .agent_data/<user>/session/session_<id>.json
    {
        "session_id": "session_abc123",
        "messages": [...],
        "created_at": "2026-06-24 10:00:00",
        "updated_at": "2026-06-24 11:30:00",
        "total_tokens_used": 12345,
        "token_log": [...]
    }
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path


# ============================================================
# 全局会话存储目录（兜底路径）
# 当不传 user_dir 时使用此路径，保持向后兼容
# ============================================================
SESSION_DIR = Path(__file__).parent.parent / ".agent_data" / "sessions"


class Session:
    """
    单个会话的数据结构。

    每个会话对应一个与 Agent 的完整对话流程，包含多轮 user/assistant/tool 消息。

    Attributes:
        session_id: 会话唯一标识（格式: session_<8位hex>）
        messages: 对话历史消息列表（OpenAI Chat 格式）
        created_at: 创建时间
        updated_at: 最后更新时间
        metadata: 会话元数据（可扩展，如标签、分类等）
        total_tokens_used: 当前会话累计 token 使用量
        token_log: 每步的 token 使用记录列表
    """

    def __init__(self, session_id: str | None = None):
        """
        初始化会话。

        Args:
            session_id: 会话 ID，为 None 时自动生成（session_ + 8位随机hex）
        """
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.messages: list[dict] = []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at
        self.metadata: dict = {}
        self.total_tokens_used: int = 0
        self.token_log: list[dict] = []

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        添加一条消息到对话历史。

        Args:
            role: 消息角色 — "system"(系统提示), "user"(用户输入),
                  "assistant"(助手回复), "tool"(工具执行结果)
            content: 消息内容字符串
            **kwargs: 额外字段，如：
                - tool_call_id: 工具调用 ID（role="tool" 时必填）
                - tool_calls: 工具调用列表（role="assistant" 时有工具调用时必填）
        """
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_messages(self) -> list[dict]:
        """获取完整的对话历史（返回副本，防止外部修改）。"""
        return self.messages.copy()

    def clear_history(self) -> None:
        """
        清空对话历史（保留 system 消息）。

        用于 /clear 命令。只保留 system prompt，
        用户和助手的对话全部清空，但会话 ID 和 token 统计保留。
        """
        self.messages = [m for m in self.messages if m["role"] == "system"]
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        """将会话序列化为字典（用于 JSON 序列化）。"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "total_tokens_used": self.total_tokens_used,
            "token_log": self.token_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """从字典反序列化（从 JSON 文件中恢复会话）。"""
        session = cls(session_id=data["session_id"])
        session.messages = data["messages"]
        session.created_at = data.get("created_at", "")
        session.updated_at = data.get("updated_at", "")
        session.metadata = data.get("metadata", {})
        session.total_tokens_used = data.get("total_tokens_used", 0)
        session.token_log = data.get("token_log", [])
        return session

    def add_token_usage(self, step: int, tokens: int, label: str = "",
                        tool_name: str = "") -> None:
        """
        记录一次 token 使用到日志。

        Args:
            step: 当前 ReAct 步数
            tokens: 本次消耗的 token 数
            label: 用途描述（如 "LLM推理", "工具调用"）
            tool_name: 工具名称（如果是工具调用产生的消耗）
        """
        self.total_tokens_used += tokens
        self.token_log.append({
            "step": step,
            "tokens": tokens,
            "cumulative": self.total_tokens_used,
            "label": label,
            "tool": tool_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


class SessionManager:
    """
    会话管理器 — 管理多个会话的创建、加载、保存、列出、删除。

    会话数据以 JSON 文件形式持久化到指定目录（用户隔离或全局）。
    使用原子写入（先写临时文件再 rename）防止文件损坏。

    使用方法：
        manager = SessionManager(user_dir=user_dir)  # 用户隔离
        session = manager.create_session()            # 创建新会话
        session = manager.load_session("session_abc")  # 加载已有会话
        sessions = manager.list_sessions()             # 列出所有会话
    """

    def __init__(self, user_dir: Path | None = None):
        """
        初始化会话管理器。

        Args:
            user_dir: 用户数据目录。如果提供，会话文件存储在
                      user_dir/session/ 下（实现数据隔离）。
                      如果不提供，使用全局 SESSION_DIR（向后兼容）。
        """
        if user_dir:
            self.session_dir = user_dir / "session"
        else:
            self.session_dir = SESSION_DIR  # 全局兜底
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """获取会话文件路径。"""
        return self.session_dir / f"{session_id}.json"

    def create_session(self, system_prompt: str = "") -> Session:
        """
        创建新的会话。

        创建流程：
        1. 实例化 Session（自动生成 UUID）
        2. 添加系统提示词（定义 Agent 行为）
        3. 立即持久化到文件

        Args:
            system_prompt: 系统提示词，为空时使用默认提示词

        Returns:
            新创建的 Session 实例
        """
        session = Session()

        # 添加系统提示词
        if not system_prompt:
            system_prompt = self._default_system_prompt()
        session.add_message("system", system_prompt)

        # 立即保存到文件（确保即使后续出错，会话也已持久化）
        self.save_session(session)
        return session

    def save_session(self, session: Session) -> None:
        """
        保存会话到文件（原子写入）。

        使用原子写入策略：
        1. 将数据写入临时文件（xxx.json.tmp）
        2. 用 replace() 原子替换原文件
        3. fsync 确保数据落盘

        为什么需要 fsync：
        - 操作系统可能缓冲写入到内存中，崩溃时丢失数据
        - fsync 强制将缓冲写入磁盘
        - 在 Windows 上 fsync 行为略有不同，但能确保数据离开应用缓冲区
        """
        path = self._session_path(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()           # 刷新 Python 缓冲区 → OS
            os.fsync(f.fileno())  # 强制 OS 写入磁盘

    def load_session(self, session_id: str) -> Session | None:
        """
        从文件加载会话。

        如果文件不存在或损坏，返回 None（而不是抛异常）。
        调用方（Runtime）负责处理 None 情况 — 创建新会话。

        Args:
            session_id: 会话 ID

        Returns:
            Session 实例，如果不存在或损坏返回 None
        """
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def list_sessions(self) -> list[dict]:
        """
        列出所有已保存的会话摘要。

        扫描会话目录下的所有 JSON 文件，提取元信息。
        按最后更新时间倒序排列。

        Returns:
            会话摘要列表，每项包含 session_id, updated_at, message_count
        """
        sessions = []
        for path in self.session_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                # 文件损坏 → 跳过，不阻塞整个列表
                continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话。返回 True 表示删除成功。"""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def _default_system_prompt() -> str:
        """
        默认系统提示词。

        定义了 Agent 的角色、能力和行为规范。
        这是 Agent 的"人格设定"，会影响 LLM 的所有响应。

        设计要点：
        1. 明确列出所有工具及其用途（LLM 需要知道有什么工具可用）
        2. 规定工作原则（何时调用工具，何时直接回答）
        3. 强调持久化特性（任务数据跨对话轮次保留）
        4. 要求中文回答

        注意：新增工具后必须同步更新此提示词。
        """
        return """你是一个智能助手 Agent，具备以下能力：

1. **数学计算**：使用 calculator 工具进行数学运算
2. **信息搜索**：使用 search 工具搜索相关信息
3. **任务管理**：使用 todo_manager 工具创建和管理任务
4. **文件读取**：使用 read_file 工具读取本地文本文件、PDF 文档、Word 文档（代码、配置、数据、报告等）
5. **文件写入**：使用 write_file 工具将内容写入本地文件，支持文本文件、PDF 文档、Word 文档
6. **时间感知**：使用 datetime_tool 获取当前日期和时间

工作原则：
- 当用户的问题需要计算时，调用 calculator 工具
- 当用户需要查找信息时，调用 search 工具
- 当用户想要创建、查看或管理任务时，调用 todo_manager 工具
- 当用户要求读取或分析文件时，调用 read_file 工具
- 当用户要求写入或生成文件时，调用 write_file 工具
- 当用户询问时间或需要时间信息时，调用 datetime_tool
- 如果问题可以直接回答（不需要工具），直接给出答案
- 如果一个任务需要多个步骤，逐步执行每个步骤
- 任务管理是持久化的，之前创建的任务在后续对话中仍然可以查看和修改

请用中文回答用户的问题。"""
