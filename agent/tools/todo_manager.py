"""
Todo Manager 工具 - 跨轮次任务管理器

支持创建、查看、更新、删除任务。
所有任务状态持久化到 JSON 文件，实现跨对话轮次的状态保持。

这是实现"跨轮次继续执行"场景的核心工具：
- 第一轮：用户请求创建任务，Agent 调用 create_task
- 第二轮：用户询问进度，Agent 调用 list_tasks 获取已有状态
"""

import json
import os
from datetime import datetime
from pathlib import Path
from agent.tools.base import BaseTool, ToolResult


# ============================================================
# 任务持久化文件路径
# 使用项目根目录下的 .agent_data 目录存储
# ============================================================
DATA_DIR = Path(__file__).parent.parent.parent / ".agent_data"
TODO_FILE = DATA_DIR / "todos.json"


def _load_todos() -> dict:
    """
    从 JSON 文件加载所有任务

    Returns:
        任务字典，格式为 {task_id: {title, status, detail, created_at, updated_at}}
        如果文件不存在则返回空字典
    """
    if not TODO_FILE.exists():
        return {}
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_todos(todos: dict) -> None:
    """
    将任务字典持久化到 JSON 文件

    Args:
        todos: 任务字典
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)


def _next_id(todos: dict) -> str:
    """生成下一个任务 ID（自增格式: task_001, task_002, ...）"""
    if not todos:
        return "task_001"
    max_num = 0
    for tid in todos:
        try:
            num = int(tid.split("_")[1])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            continue
    return f"task_{max_num + 1:03d}"


class TodoManagerTool(BaseTool):
    """
    任务管理器工具 - 支持跨轮次状态持久化

    支持的操作（通过 action 参数指定）：
    - create: 创建新任务
    - list: 列出所有任务（可按状态过滤）
    - update: 更新任务状态或详情
    - delete: 删除指定任务
    - get: 获取单个任务详情

    使用示例：
        action: "create", title: "实现登录功能", detail: "需要 OAuth2.0 认证"
        action: "list", status: "pending"
        action: "update", task_id: "task_001", status: "in_progress"
        action: "delete", task_id: "task_002"
    """

    @property
    def name(self) -> str:
        return "todo_manager"

    @property
    def description(self) -> str:
        return (
            "任务管理器，用于创建、查看、更新和删除任务。任务状态会持久化保存，跨对话轮次可用。"
            "支持的操作: create(创建), list(列出), update(更新), delete(删除), get(查看详情)。"
            "任务状态有: pending(待处理), in_progress(进行中), done(已完成)。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "update", "delete", "get"],
                    "description": "要执行的操作类型",
                },
                "title": {
                    "type": "string",
                    "description": "任务标题（create 时必填）",
                },
                "detail": {
                    "type": "string",
                    "description": "任务详细描述（可选）",
                },
                "task_id": {
                    "type": "string",
                    "description": "任务 ID（update/delete/get 时必填），如 task_001",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done"],
                    "description": "任务状态（list 时用于过滤，update 时用于修改）",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        title: str = "",
        detail: str = "",
        task_id: str = "",
        status: str = "",
    ) -> ToolResult:
        """
        执行任务管理操作

        根据 action 分发到不同的处理方法：
        - create: 创建新任务并持久化
        - list: 加载并过滤任务列表
        - update: 修改任务状态/详情并持久化
        - delete: 删除任务并持久化
        - get: 获取单个任务详情

        Args:
            action: 操作类型
            title: 任务标题（create 用）
            detail: 任务详情（create/update 用）
            task_id: 任务 ID（update/delete/get 用）
            status: 任务状态（list 过滤 / update 修改用）

        Returns:
            ToolResult: 操作结果
        """
        # 根据 action 分发处理
        if action == "create":
            return self._create_task(title, detail)
        elif action == "list":
            return self._list_tasks(status)
        elif action == "update":
            return self._update_task(task_id, status, detail)
        elif action == "delete":
            return self._delete_task(task_id)
        elif action == "get":
            return self._get_task(task_id)
        else:
            return ToolResult(
                success=False,
                result="",
                error=f"不支持的操作: {action}。支持: create, list, update, delete, get",
            )

    def _create_task(self, title: str, detail: str = "") -> ToolResult:
        """创建新任务"""
        if not title:
            return ToolResult(success=False, result="", error="创建任务需要提供 title 参数")

        todos = _load_todos()
        task_id = _next_id(todos)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        todos[task_id] = {
            "title": title,
            "detail": detail,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        _save_todos(todos)

        return ToolResult(
            success=True,
            result=f"任务创建成功！\n  ID: {task_id}\n  标题: {title}\n  状态: pending",
            metadata={"task_id": task_id},
        )

    def _list_tasks(self, status_filter: str = "") -> ToolResult:
        """列出任务（可按状态过滤）"""
        todos = _load_todos()

        if not todos:
            return ToolResult(success=True, result="当前没有任何任务。")

        # 按状态过滤
        filtered = todos
        if status_filter:
            filtered = {k: v for k, v in todos.items() if v["status"] == status_filter}

        if not filtered:
            return ToolResult(
                success=True,
                result=f"没有状态为 '{status_filter}' 的任务。",
            )

        # 格式化输出
        lines = [f"任务列表（共 {len(filtered)} 个）："]
        lines.append("-" * 50)
        for tid, task in sorted(filtered.items()):
            status_icon = {"pending": "⬜", "in_progress": "🔄", "done": "✅"}.get(
                task["status"], "❓"
            )
            lines.append(f"  {status_icon} [{tid}] {task['title']}")
            lines.append(f"     状态: {task['status']} | 创建: {task['created_at']}")
            if task.get("detail"):
                lines.append(f"     详情: {task['detail']}")
            lines.append("")

        return ToolResult(
            success=True,
            result="\n".join(lines),
            metadata={"count": len(filtered)},
        )

    def _update_task(self, task_id: str, status: str = "", detail: str = "") -> ToolResult:
        """更新任务状态或详情"""
        if not task_id:
            return ToolResult(success=False, result="", error="更新任务需要提供 task_id")

        todos = _load_todos()
        if task_id not in todos:
            return ToolResult(success=False, result="", error=f"任务不存在: {task_id}")

        updated = False
        if status:
            todos[task_id]["status"] = status
            updated = True
        if detail:
            todos[task_id]["detail"] = detail
            updated = True

        if not updated:
            return ToolResult(
                success=False, result="", error="请提供要更新的内容（status 或 detail）"
            )

        todos[task_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_todos(todos)

        task = todos[task_id]
        return ToolResult(
            success=True,
            result=f"任务更新成功！\n  [{task_id}] {task['title']}\n  状态: {task['status']}",
        )

    def _delete_task(self, task_id: str) -> ToolResult:
        """删除任务"""
        if not task_id:
            return ToolResult(success=False, result="", error="删除任务需要提供 task_id")

        todos = _load_todos()
        if task_id not in todos:
            return ToolResult(success=False, result="", error=f"任务不存在: {task_id}")

        title = todos[task_id]["title"]
        del todos[task_id]
        _save_todos(todos)

        return ToolResult(
            success=True,
            result=f"任务已删除: [{task_id}] {title}",
        )

    def _get_task(self, task_id: str) -> ToolResult:
        """获取单个任务详情"""
        if not task_id:
            return ToolResult(success=False, result="", error="查看任务需要提供 task_id")

        todos = _load_todos()
        if task_id not in todos:
            return ToolResult(success=False, result="", error=f"任务不存在: {task_id}")

        task = todos[task_id]
        lines = [
            f"任务详情: [{task_id}]",
            f"  标题: {task['title']}",
            f"  状态: {task['status']}",
            f"  详情: {task.get('detail', '无')}",
            f"  创建时间: {task['created_at']}",
            f"  更新时间: {task['updated_at']}",
        ]

        return ToolResult(success=True, result="\n".join(lines))

