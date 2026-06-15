"""
Session 管理器 - 多轮对话和状态维护

负责：
1. 维护对话历史（messages 列表）
2. 持久化会话状态到文件
3. 管理多个独立会话
4. 注入系统提示词和上下文信息
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


# ============================================================
# 会话数据存储目录
# ============================================================
SESSION_DIR = Path(__file__).parent.parent / ".agent_data" / "sessions"


class Session:
    """
    单个会话的数据结构

    Attributes:
        session_id: 会话唯一标识
        messages: 对话历史消息列表（OpenAI Chat 格式）
        created_at: 创建时间
        updated_at: 最后更新时间
        metadata: 会话元数据（可扩展）
    """

    def __init__(self, session_id: str | None = None):
        """
        初始化会话

        Args:
            session_id: 会话 ID，为 None 时自动生成
        """
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.messages: list[dict] = []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at
        self.metadata: dict = {}

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """
        添加一条消息到对话历史

        Args:
            role: 消息角色 ("system", "user", "assistant", "tool")
            content: 消息内容
            **kwargs: 额外字段（如 tool_call_id, tool_calls）
        """
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_messages(self) -> list[dict]:
        """获取完整的对话历史"""
        return self.messages.copy()

    def clear_history(self) -> None:
        """清空对话历史（保留 system 消息）"""
        self.messages = [m for m in self.messages if m["role"] == "system"]
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """从字典反序列化"""
        session = cls(session_id=data["session_id"])
        session.messages = data["messages"]
        session.created_at = data.get("created_at", "")
        session.updated_at = data.get("updated_at", "")
        session.metadata = data.get("metadata", {})
        return session


class SessionManager:
    """
    会话管理器 - 管理多个会话的创建、加载、保存

    会话数据以 JSON 文件形式持久化到 SESSION_DIR 目录。

    使用方法：
        manager = SessionManager()

        # 创建新会话
        session = manager.create_session()

        # 加载已有会话
        session = manager.load_session("session_abc123")

        # 列出所有会话
        sessions = manager.list_sessions()
    """

    def __init__(self):
        """初始化会话管理器，确保存储目录存在"""
        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """获取会话文件路径"""
        return SESSION_DIR / f"{session_id}.json"

    def create_session(self, system_prompt: str = "") -> Session:
        """
        创建新的会话

        Args:
            system_prompt: 系统提示词，为 None 时使用默认

        Returns:
            新创建的 Session 实例
        """
        session = Session()

        # 添加系统提示词
        if not system_prompt:
            system_prompt = self._default_system_prompt()
        session.add_message("system", system_prompt)

        # 保存会话
        self.save_session(session)
        return session

    def save_session(self, session: Session) -> None:
        """
        保存会话到文件

        Args:
            session: 要保存的会话实例
        """
        path = self._session_path(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Session | None:
        """
        从文件加载会话

        Args:
            session_id: 会话 ID

        Returns:
            Session 实例，如果不存在返回 None
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
        列出所有已保存的会话

        Returns:
            会话摘要列表 [{"session_id": ..., "updated_at": ..., "message_count": ...}, ...]
        """
        sessions = []
        for path in SESSION_DIR.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话"""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def _default_system_prompt() -> str:
        """
        默认系统提示词

        定义了 Agent 的角色、能力和行为规范。
        这是 Agent 的"人格设定"，会影响 LLM 的所有响应。
        新增工具后必须同步更新此提示词。
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
