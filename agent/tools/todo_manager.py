"""
============================================================
Todo Manager 工具 — 跨轮次持久化任务管理器
============================================================

这是实现"跨轮次继续执行"场景的核心工具：
    - 第一轮：用户请求创建任务，Agent 调用 create_task
    - 第二轮（新会话或同一会话）：用户询问进度，Agent 调用 list_tasks

数据隔离：任务数据按用户存储到 user_dir/task/。
    不同用户的任务完全隔离，互不可见。
    这是通过 Runtime._init_todo_scope() 在初始化时设置用户目录实现的。

数据安全：
    - 文件锁：使用 O_CREAT | O_EXCL 原子操作实现文件锁
    - 原子写入：先写临时文件再 rename，防止写崩溃导致文件损坏
    - 锁过期：超过 10 秒的锁视为过期，自动清理

支持的操作：
    create  — 创建新任务（需 title）
    list    — 列出所有任务（可按 status 过滤）
    update  — 更新任务状态或详情（需 task_id）
    delete  — 删除指定任务（需 task_id）
    get     — 获取单个任务详情（需 task_id）
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from agent.tools.base import BaseTool, ToolResult, ToolError, ErrorCode


class TodoManagerTool(BaseTool):
    """
    任务管理器工具 — 支持跨轮次状态持久化。

    所有任务数据按用户隔离存储，通过 set_user_dir() 设置用户目录。
    支持 CRUD 操作：create, list, update, delete, get。

    Attributes:
        _user_dir: 用户数据目录（由 Runtime 初始化时设置）
    """

    def __init__(self):
        super().__init__()
        self._user_dir: Path | None = None
        self._timeout = 5.0

    def set_user_dir(self, user_dir: Path):
        """设置用户目录，任务数据将存储到 user_dir/task/。"""
        self._user_dir = user_dir
        (user_dir / "task").mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "todo_manager"

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def description(self) -> str:
        return (
            "任务管理器，用于创建、查看、更新和删除任务。任务状态会持久化保存，跨对话轮次可用。"
            "支持的操作: create(创建), list(列出), update(更新), delete(删除), get(查看详情)。"
            "任务状态有: pending(待处理), in_progress(进行中), done(已完成)。"
            "注意：删除任务时，需要先不加 confirm 参数让工具询问用户确认，"
            "用户同意后再加 confirm=true 执行删除。"
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
                "confirm": {
                    "type": "boolean",
                    "description": "是否确认执行删除。delete 操作时，需要先不加此参数让工具询问用户确认，用户同意后再加 confirm=true 执行删除。默认为 false。",
                    "default": False,
                },
            },
            "required": ["action"],
        }

    def before_execute(self, kwargs: dict, context) -> dict | None:
        """执行前检查用户认证。未登录时拒绝使用任务管理。"""
        if not self.get_context().is_authenticated and self._user_dir is None:
            raise ToolError(
                ErrorCode.PERMISSION_DENIED,
                "任务管理需要登录后才能使用",
            )
        return None

    # ================================================================
    # 用户目录解析
    # 优先级：set_user_dir() > ToolContext.user_dir
    # ================================================================

    def _resolve_user_dir(self) -> Path:
        """
        解析用户目录。

        优先级：
        1. set_user_dir() 设置的目录（由 Runtime._init_todo_scope 在初始化时设置）
        2. ToolContext 中注入的 user_dir
        3. 报错

        Returns:
            用户的任务数据目录
        """
        # 优先使用 _user_dir（由 Runtime 初始化时设置）
        if self._user_dir is not None:
            return self._user_dir

        # 回退：使用 ToolContext 中的 user_dir
        ctx = self.get_context()
        if ctx.user_dir is not None:
            return ctx.user_dir

        raise RuntimeError(
            f"[{ErrorCode.PERMISSION_DENIED}] TodoManagerTool: 无法确定用户目录，"
            f"请先登录或调用 set_user_dir()"
        )

    def _ensure_task_dir(self) -> Path:
        """获取并确保任务目录存在。"""
        user_dir = self._resolve_user_dir()
        task_dir = user_dir / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _get_todo_file(self) -> Path:
        """获取用户任务数据文件路径: user_dir/task/todos.json。"""
        return self._ensure_task_dir() / "todos.json"

    def _get_lock_file(self) -> Path:
        """获取用户锁文件路径: user_dir/task/todos.lock。"""
        return self._ensure_task_dir() / "todos.lock"

    def _get_data_dir(self) -> Path:
        """获取用户任务数据目录: user_dir/task/。"""
        return self._ensure_task_dir()

    # ================================================================
    # 文件锁
    # 使用 O_CREAT | O_EXCL 原子操作确保并发安全
    # ================================================================

    def _acquire_lock(self, timeout: float = 3.0) -> bool:
        """
        获取文件锁（阻塞，最多等待 timeout 秒）。

        锁机制：
        - 创建锁文件作为互斥信号量
        - 使用 O_CREAT | O_EXCL 保证原子性（同时只有一个进程能创建成功）
        - 重试直到超时
        - 超过 10 秒的锁视为过期（防止进程崩溃导致死锁）

        Args:
            timeout: 最大等待秒数

        Returns:
            True 成功获得锁，False 超时
        """
        lock_file = self._get_lock_file()
        data_dir = self._get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        while time.time() - start < timeout:
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                # 检查锁文件是否过期（超过 10 秒的锁视为过期）
                if lock_file.exists():
                    age = time.time() - lock_file.stat().st_mtime
                    if age > 10:
                        try:
                            lock_file.unlink()
                        except OSError:
                            pass
                time.sleep(0.05)

        return False

    def _release_lock(self):
        """释放文件锁。"""
        lock_file = self._get_lock_file()
        try:
            if lock_file.exists():
                lock_file.unlink()
        except (OSError, PermissionError):
            pass

    # ================================================================
    # 数据持久化
    # 使用"临时文件 → 原子重命名"模式防止文件损坏
    # ================================================================

    def _load_todos(self) -> dict:
        """
        从 JSON 文件加载所有任务（带文件锁）。

        Returns:
            任务字典 {task_id: task_data, ...}，文件不存在或损坏返回空字典
        """
        todo_file = self._get_todo_file()
        if not todo_file.exists():
            return {}

        self._acquire_lock()
        try:
            with open(todo_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
        finally:
            self._release_lock()

    def _save_todos(self, todos: dict) -> None:
        """
        将任务字典持久化到 JSON 文件（原子写入 + 文件锁）。

        原子写入流程：
        1. 获取文件锁（防止并发写入）
        2. 先写入临时文件 todos.json.tmp
        3. 用 replace() 原子替换原文件
        4. 释放文件锁

        为什么不用直接写入：
        - 直接写入原文件时，如果进程崩溃会留下半截 JSON
        - 原子替换保证：要么全部成功，要么原文件不变

        Raises:
            RuntimeError: 获取锁超时或写入失败时抛出，携带具体原因
        """
        todo_file = self._get_todo_file()
        data_dir = self._get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        # 获取文件锁（最多等 3 秒）
        acquired = self._acquire_lock()
        if not acquired:
            raise RuntimeError(
                "无法获取任务文件锁，数据写入失败。"
                "可能原因：另一个进程正在同时修改任务，请稍后重试。"
            )

        try:
            # 先写入临时文件
            tmp_file = todo_file.with_suffix(".json.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(todos, f, ensure_ascii=False, indent=2)
            tmp_file.replace(todo_file)  # 原子替换
        except (IOError, OSError, json.JSONEncodeError) as e:
            # 写入失败时清理残留的临时文件
            tmp_file = todo_file.with_suffix(".json.tmp")
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"任务数据写入磁盘失败: {e}")
        finally:
            self._release_lock()

    def _next_id(self, todos: dict) -> str:
        """生成下一个任务 ID（自增格式: task_001, task_002, ...）。"""
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

    # ================================================================
    # 核心操作
    # ================================================================

    def execute(
        self,
        action: str,
        title: str = "",
        detail: str = "",
        task_id: str = "",
        status: str = "",
        confirm: bool = False,
    ) -> ToolResult:
        """
        执行任务管理操作。

        根据 action 分发到不同的处理方法。

        Args:
            action: 操作类型 (create/list/update/delete/get)
            title: 任务标题（create 用）
            detail: 任务详情（create/update 用）
            task_id: 任务 ID（update/delete/get 用）
            status: 任务状态（list 过滤 / update 修改用）
            confirm: 确认标记。delete 操作时需先询问用户，用户同意后再传 confirm=true

        Returns:
            ToolResult: 操作结果
        """
        if action == "create":
            return self._create_task(title, detail)
        elif action == "list":
            return self._list_tasks(status)
        elif action == "update":
            return self._update_task(task_id, status, detail)
        elif action == "delete":
            return self._delete_task(task_id, confirm)
        elif action == "get":
            return self._get_task(task_id)
        else:
            return ToolResult(
                success=False,
                result="",
                error=f"不支持的操作: {action}。支持: create, list, update, delete, get",
            )

    def _create_task(self, title: str, detail: str = "") -> ToolResult:
        """创建新任务。"""
        if not title:
            return ToolResult(success=False, result="", error="创建任务需要提供 title 参数")

        todos = self._load_todos()
        task_id = self._next_id(todos)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        todos[task_id] = {
            "title": title,
            "detail": detail,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }

        # 持久化到磁盘；如果写入失败，内存中的修改自然丢弃（未落盘）
        try:
            self._save_todos(todos)
        except RuntimeError as e:
            return ToolResult(success=False, result="", error=str(e))

        return ToolResult(
            success=True,
            result=f"任务创建成功！\n  ID: {task_id}\n  标题: {title}\n  状态: pending",
            metadata={"task_id": task_id},
        )

    def _list_tasks(self, status_filter: str = "") -> ToolResult:
        """列出任务（可按状态过滤）。"""
        todos = self._load_todos()

        if not todos:
            return ToolResult(success=True, result="当前没有任何任务。")

        filtered = todos
        if status_filter:
            filtered = {k: v for k, v in todos.items() if v["status"] == status_filter}

        if not filtered:
            return ToolResult(
                success=True,
                result=f"没有状态为 '{status_filter}' 的任务。",
            )

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
        """
        更新任务状态或详情。

        注意：如果磁盘写入失败，会返回错误信息，内存中的修改自然丢弃。
        用户看到错误后知道需要重试，不会出现"显示成功但实际没存上"的问题。
        """
        if not task_id:
            return ToolResult(success=False, result="", error="更新任务需要提供 task_id")

        todos = self._load_todos()
        if task_id not in todos:
            return ToolResult(success=False, result="", error=f"任务不存在: {task_id}")

        # 先把旧数据存一份快照，方便将来可能的回滚（目前不需要回滚，
        # 因为 _save_todos 失败时直接返回错误，内存修改自然丢弃）
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

        # 持久化到磁盘
        # _save_todos() 内部使用原子写入 + 文件锁，失败时会抛出 RuntimeError
        # 这里捕获并转为 ToolResult 返回给 LLM
        try:
            self._save_todos(todos)
        except RuntimeError as e:
            return ToolResult(success=False, result="", error=str(e))

        task = todos[task_id]
        return ToolResult(
            success=True,
            result=f"任务更新成功！\n  [{task_id}] {task['title']}\n  状态: {task['status']}",
        )

    def _delete_task(self, task_id: str, confirm: bool = False) -> ToolResult:
        """
        删除任务（需要用户确认）。

        安全策略：delete 操作不可撤销，必须先询问用户确认。
        用户同意后，LLM 再传 confirm=true 执行实际删除。

        Args:
            task_id: 要删除的任务 ID
            confirm: 是否已确认删除。默认为 False，此时会返回 ask_user 询问用户。

        Returns:
            ToolResult: 操作结果或确认请求
        """
        if not task_id:
            return ToolResult(success=False, result="", error="删除任务需要提供 task_id")

        todos = self._load_todos()
        if task_id not in todos:
            return ToolResult(success=False, result="", error=f"任务不存在: {task_id}")

        title = todos[task_id]["title"]

        # 如果未确认，向用户提问，等待确认后再执行
        if not confirm:
            return ToolResult(
                success=True,
                result=(
                    f"准备删除任务:\n"
                    f"  [{task_id}] {title}\n"
                    f"请在确认后重新调用 todo_manager 并设置 confirm=true 执行删除。"
                ),
                ask_user=f"确定要删除任务 '{title}'（{task_id}）吗？此操作不可撤销。",
            )

        del todos[task_id]

        # 持久化到磁盘；写入失败时内存中的删除不生效
        try:
            self._save_todos(todos)
        except RuntimeError as e:
            return ToolResult(success=False, result="", error=str(e))

        return ToolResult(
            success=True,
            result=f"任务已删除: [{task_id}] {title}",
        )

    def _get_task(self, task_id: str) -> ToolResult:
        """获取单个任务详情。"""
        if not task_id:
            return ToolResult(success=False, result="", error="查看任务需要提供 task_id")

        todos = self._load_todos()
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
