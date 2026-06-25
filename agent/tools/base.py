"""
============================================================
工具基类定义 — 所有 Agent 可调用工具的基类和基础设施
============================================================

本模块定义了工具系统的全部基础设施：

核心组件：
    1. BaseTool — 工具基类，所有工具必须继承
    2. ToolResult — 工具执行结果封装
    3. ToolContext — 工具执行上下文（用户/会话信息）
    4. ToolError — 结构化异常（携带标准错误码）
    5. ErrorCode — 标准错误码常量

扩展机制：
    - 生命周期钩子：before_execute / after_execute / on_error
    - 结果缓存：自动缓存纯函数工具的结果（如计算器）
    - 动态参数：根据已填参数调整参数 Schema
    - 异步执行：execute_async() 原生协程支持
    - 自文档化：examples + usage_guide 属性
    - 依赖检查装饰器：require_import() 友好提示缺失依赖

设计理念：
    每个工具是一个自包含的类，继承 BaseTool 并实现以下接口：
    - name: 工具唯一标识名（LLM 通过此名称调用）
    - description: 工具功能描述（供 LLM 理解何时使用）
    - parameters: 参数 JSON Schema（LLM 据此生成参数）
    - execute(): 实际执行逻辑
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import json
import time
import hashlib
import importlib
from functools import wraps
from pathlib import Path
from agent import config


# ============================================================
# 结构化错误码
# ============================================================
# 使用常量而非枚举，方便在 json 序列化中直接使用字符串值

class ErrorCode:
    """
    标准工具错误码常量。

    为什么要用错误码而非直接抛异常：
    1. LLM 需要理解错误类型来决定下一步行动
    2. 结构化的错误信息比"出错了"更有帮助
    3. 错误码可以携带上下文（哪个文件、哪个参数）
    4. Runtime 可以根据错误码采取不同的策略（重试/跳过/终止）

    错误码分类：
    - INVALID_*: 参数/输入相关
    - FILE_*: 文件操作相关
    - TOOL_*: 工具执行相关
    - SESSION_*: 会话相关
    - USER_*: 用户交互相关
    """
    # 参数相关
    INVALID_PARAMS = "INVALID_PARAMS"
    MISSING_PARAM = "MISSING_PARAM"
    INVALID_PARAM_TYPE = "INVALID_PARAM_TYPE"
    PARAM_OUT_OF_RANGE = "PARAM_OUT_OF_RANGE"

    # 文件相关
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    WRITE_ERROR = "WRITE_ERROR"
    READ_ERROR = "READ_ERROR"

    # 执行相关
    TIMEOUT = "TIMEOUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    NOT_FOUND = "TOOL_NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"

    # 用户交互
    ASK_USER = "ASK_USER"
    USER_CANCELLED = "USER_CANCELLED"

    # 会话相关
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_ACCESS_DENIED = "SESSION_ACCESS_DENIED"

    @classmethod
    def all_codes(cls) -> list[str]:
        """获取所有错误码。"""
        return [v for k, v in vars(cls).items() if not k.startswith("_") and isinstance(v, str)]

    @classmethod
    def description(cls, code: str) -> str:
        """获取错误码的说明文本。"""
        descriptions = {
            cls.INVALID_PARAMS: "参数校验失败，请检查参数格式和类型",
            cls.MISSING_PARAM: "缺少必须参数",
            cls.INVALID_PARAM_TYPE: "参数类型错误",
            cls.PARAM_OUT_OF_RANGE: "参数值超出允许范围",
            cls.FILE_NOT_FOUND: "指定的文件不存在",
            cls.FILE_TOO_LARGE: "文件大小超出限制",
            cls.UNSUPPORTED_TYPE: "不支持的文件类型",
            cls.PERMISSION_DENIED: "权限不足，无法执行此操作",
            cls.WRITE_ERROR: "文件写入失败",
            cls.READ_ERROR: "文件读取失败",
            cls.TIMEOUT: "操作执行超时",
            cls.EXECUTION_ERROR: "工具执行过程中发生异常",
            cls.NOT_FOUND: "请求的工具不存在",
            cls.RATE_LIMITED: "操作频率过高，请稍后重试",
            cls.DEPENDENCY_MISSING: "缺少必要的 Python 依赖库",
            cls.ASK_USER: "需要向用户确认或询问更多信息",
            cls.USER_CANCELLED: "用户取消了操作",
            cls.SESSION_NOT_FOUND: "指定的会话不存在",
            cls.SESSION_ACCESS_DENIED: "无权访问此会话",
        }
        return descriptions.get(code, "未知错误")


class ToolError(Exception):
    """
    结构化工具异常。

    携带标准错误码，供 Runtime 和 LLM 理解错误类型并采取不同策略。
    支持嵌套原始异常信息。

    与普通 Exception 的区别：
    - 普通 Exception："出错了"（LLM 不知道错在哪）
    - ToolError："[FILE_NOT_FOUND] 文件 /path/to/file 不存在"
      （LLM 可以知道是文件问题，建议用户检查路径）

    用法:
        raise ToolError(ErrorCode.FILE_NOT_FOUND, f"文件不存在: {path}")
        raise ToolError(ErrorCode.PERMISSION_DENIED, "无权写入此路径")
    """

    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")

    def to_tool_result(self) -> "ToolResult":
        """将异常转为 ToolResult（方便在钩子中返回）。"""
        return ToolResult(
            success=False,
            result="",
            error=f"[{self.code}] {self.message}",
            metadata={"error_code": self.code, "error_details": self.details},
        )


# ============================================================
# 工具上下文（注入给每个工具调用）
# ============================================================

@dataclass
class ToolContext:
    """
    工具执行上下文。

    每次工具调用时，Runtime 会将此上下文注入给工具。
    工具可通过 self.context 访问当前用户、会话等信息。

    Attributes:
        username: 当前操作用户名
        user_dir: 用户数据目录 (.agent_data/<username>/)
        session_id: 当前会话 ID
        runtime_ref: Runtime 实例引用（供高级工具使用，谨慎访问）
    """
    username: str = ""
    user_dir: Path | None = None
    session_id: str = ""
    runtime_ref: Any = None  # AgentRuntime 实例（高级功能用，需注意循环引用）

    @property
    def is_authenticated(self) -> bool:
        """是否有有效的用户身份（用户名和目录都存在）。"""
        return bool(self.username) and self.user_dir is not None

    def assert_file_path_allowed(self, file_path: str, operation: str = "read") -> Path:
        """
        检查文件路径是否在用户允许范围内。

        安全策略：
        1. 如果用户没有上下文（未登录），只允许操作当前项目目录
        2. 如果已登录，文件必须在 user_dir 下（或 user_dir 的子目录）
        3. 禁止访问 .agent_data 根目录和其他用户目录
        4. 系统临时目录也允许（用于下载临时文件等场景）

        Args:
            file_path: 文件路径
            operation: 操作类型 ("read" / "write")

        Returns:
            Path: 解析后的绝对路径

        Raises:
            ToolError: 路径不在允许范围内
        """
        path = Path(file_path).expanduser().resolve()

        if not self.is_authenticated:
            # 未登录用户：只允许操作当前项目目录
            if not self._is_in_public_dir(path):
                raise ToolError(
                    ErrorCode.PERMISSION_DENIED,
                    f"未登录用户无权操作路径: {path}",
                    {"path": str(path), "operation": operation},
                )
            return path

        # 已登录用户：检查路径是否在本用户目录下
        assert self.user_dir is not None
        user_dir_resolved = self.user_dir.resolve()

        try:
            path.relative_to(user_dir_resolved)
            return path  # 路径在用户目录下，允许
        except ValueError:
            pass

        # 检查是否是公共目录（如系统临时目录）
        if self._is_in_public_dir(path):
            return path

        raise ToolError(
            ErrorCode.PERMISSION_DENIED,
            f"操作路径 {path} 不在用户目录 {user_dir_resolved} 下，"
            f"且不属于公共路径",
            {
                "path": str(path),
                "user_dir": str(user_dir_resolved),
                "operation": operation,
            },
        )

    @staticmethod
    def _is_in_public_dir(path: Path) -> bool:
        """
        检查路径是否在公共目录下。

        允许的公共目录：
        1. 当前项目目录（方便开发调试）
        2. 系统临时目录（方便下载临时文件）
        """
        path_str = str(path).lower()
        # 当前项目目录下的文件允许
        cwd = Path.cwd().resolve()
        try:
            path.relative_to(cwd)
            return True
        except ValueError:
            pass
        # 系统临时目录
        import tempfile
        tmp = Path(tempfile.gettempdir()).resolve()
        try:
            path.relative_to(tmp)
            return True
        except ValueError:
            pass
        return False


# ============================================================
# 工具结果封装
# ============================================================

@dataclass
class ToolResult:
    """
    工具执行结果封装。

    这是工具与 Runtime 之间的标准数据交换格式。
    所有工具执行（无论成功与否）都返回 ToolResult 实例。

    Attributes:
        success: 是否执行成功
        result: 执行结果内容（成功时，反馈给 LLM）
        error: 如果失败，错误信息
        ask_user: 多轮交互字段，非空时表示需要向用户提问
        continuation: 结果被截断时，描述如何继续获取后续内容的提示。
                      格式: {"tool": "工具名", "params": {...}, "hint": "给LLM的提示文本"}
                      如果不为 None，BaseTool 截断时会在末尾追加 hint 信息。
                      如果为 None，BaseTool 会使用通用提示说明结果已丢失。
        metadata: 额外元数据（执行耗时、错误码、截断标记等）
    """
    success: bool
    result: str
    error: str | None = None
    ask_user: str | None = None  # 多轮交互：向用户提出的问题
    continuation: dict | None = None  # 截断恢复指引，详见类文档
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def error_code(self) -> str | None:
        """获取结构化错误码（如果有）。"""
        return self.metadata.get("error_code")


# ============================================================
# 工具缓存（LRU）
# ============================================================

class _ToolCache:
    """
    工具结果 LRU 缓存。

    用于纯函数工具（计算器、时间等），相同参数直接返回缓存结果。
    非纯函数工具（文件读写、任务管理等）不应启用缓存。

    为什么用 LRU：
    - 缓存空间有限（默认 128 条），LRU 淘汰最久未使用的条目
    - 工具调用通常有局部性（一段时间内反复调用相似参数）
    """
    def __init__(self, max_size: int = 128, ttl: float = 300.0):
        """
        Args:
            max_size: 最大缓存条目数（超过时淘汰最久未使用的）
            ttl: 缓存生存时间（秒），过期后自动失效
        """
        self._max_size = max_size
        self._ttl = ttl
        self._cache: dict[str, tuple[float, ToolResult]] = {}
        self._lru_order: list[str] = []

    def _make_key(self, args_dict: dict) -> str:
        """生成缓存键：参数 dict 的 JSON 序列化后的 MD5 哈希。"""
        raw = json.dumps(args_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, args_dict: dict) -> ToolResult | None:
        """获取缓存，过期或不存在返回 None。"""
        key = self._make_key(args_dict)
        if key not in self._cache:
            return None
        timestamp, result = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._lru_order.remove(key)
            return None
        # 更新 LRU 顺序（移到末尾，表示最近使用过）
        self._lru_order.remove(key)
        self._lru_order.append(key)
        return result

    def set(self, args_dict: dict, result: ToolResult):
        """写入缓存。"""
        key = self._make_key(args_dict)
        self._cache[key] = (time.time(), result)
        self._lru_order.append(key)

        # LRU 淘汰：移除最久未使用的条目
        while len(self._cache) > self._max_size:
            oldest = self._lru_order.pop(0)
            del self._cache[oldest]

    def clear(self):
        """清空缓存。"""
        self._cache.clear()
        self._lru_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ============================================================
# 钩子注册表
# ============================================================

class HookRegistry:
    """
    工具生命周期钩子注册表。

    支持全局钩子（所有工具生效）和工具级钩子（仅对指定工具生效）。

    钩子类型：
    - "before": 执行前，可修改参数或中止执行
        def hook(context: ToolContext, kwargs: dict) -> dict | ToolResult | None
          返回 dict → 修改后的参数
          返回 ToolResult → 直接返回该结果，跳过 execute
          返回 None → 继续正常执行

    - "after": 执行后，可修改结果
        def hook(context: ToolContext, result: ToolResult) -> ToolResult | None
          返回 ToolResult → 替换原结果
          返回 None → 使用原结果

    - "on_error": 出错时，可处理或转换错误
        def hook(context: ToolContext, error: ToolError) -> ToolResult | None
          返回 ToolResult → 替换错误信息
          返回 None → 继续使用默认错误处理

    使用场景：
    - 审计日志：before 钩子记录每次工具调用
    - 参数脱敏：after 钩子过滤敏感信息（如密码、密钥）
    - 权限校验：before 钩子检查用户是否有权执行该操作
    """

    def __init__(self):
        self._global_hooks: dict[str, list[callable]] = {
            "before": [],
            "after": [],
            "on_error": [],
        }
        self._tool_hooks: dict[str, dict[str, list[callable]]] = {}

    def register_global(self, hook_type: str, hook_fn: callable):
        """注册全局钩子（所有工具都生效）。"""
        if hook_type not in self._global_hooks:
            raise ValueError(f"未知钩子类型: {hook_type}，支持: before, after, on_error")
        self._global_hooks[hook_type].append(hook_fn)

    def register(self, tool_name: str, hook_type: str, hook_fn: callable):
        """注册工具级钩子（仅对指定工具生效）。"""
        if hook_type not in ("before", "after", "on_error"):
            raise ValueError(f"未知钩子类型: {hook_type}")
        if tool_name not in self._tool_hooks:
            self._tool_hooks[tool_name] = {"before": [], "after": [], "on_error": []}
        self._tool_hooks[tool_name][hook_type].append(hook_fn)

    def run_before(self, tool_name: str, context: ToolContext,
                   kwargs: dict) -> dict | ToolResult | None:
        """运行所有 before 钩子（先全局，后工具级）。"""
        for hook in self._global_hooks["before"]:
            result = hook(context, kwargs)
            if isinstance(result, (dict, ToolResult)):
                return result
        for hook in self._tool_hooks.get(tool_name, {}).get("before", []):
            result = hook(context, kwargs)
            if isinstance(result, (dict, ToolResult)):
                return result
        return None

    def run_after(self, tool_name: str, context: ToolContext,
                  result: ToolResult) -> ToolResult | None:
        """运行所有 after 钩子。"""
        for hook in self._global_hooks["after"]:
            r = hook(context, result)
            if isinstance(r, ToolResult):
                result = r
        for hook in self._tool_hooks.get(tool_name, {}).get("after", []):
            r = hook(context, result)
            if isinstance(r, ToolResult):
                result = r
        return result

    def run_on_error(self, tool_name: str, context: ToolContext,
                     error: ToolError) -> ToolResult | None:
        """运行所有 on_error 钩子。"""
        for hook in self._global_hooks["on_error"]:
            result = hook(context, error)
            if isinstance(result, ToolResult):
                return result
        for hook in self._tool_hooks.get(tool_name, {}).get("on_error", []):
            result = hook(context, error)
            if isinstance(result, ToolResult):
                return result
        return None


# 全局钩子注册表实例（单例）
HOOKS = HookRegistry()


# ============================================================
# 工具基类
# ============================================================

class BaseTool(ABC):
    """
    工具基类 — 所有工具必须继承此类。

    子类需要实现以下抽象方法：
    - name (property): 工具名称（LLM 通过此名称调用工具）
    - description (property): 工具功能描述（LLM 据此决定何时调用）
    - parameters (property): 参数 JSON Schema（LLM 据此生成参数）
    - execute(**kwargs) -> ToolResult: 核心执行逻辑

    可选扩展：
    - execute_async(**kwargs) -> ToolResult: 异步执行逻辑
    - examples (property): 调用示例列表（注入到 LLM 提示词中）
    - usage_guide (property): 使用指南（何时/如何使用）
    - get_dynamic_parameters(kwargs) -> dict: 根据已填参数动态调整 Schema
    - before_execute(kwargs, context) -> dict|ToolResult|None: 前置钩子
    - after_execute(result, context) -> ToolResult|None: 后置钩子
    - on_error(error, context) -> ToolResult|None: 错误处理钩子
    - cache_ttl (property): 缓存 TTL（秒），0=不缓存

    安全执行流程（safe_execute）：
        参数解析 → 参数校验 → before_execute 钩子 → 缓存检查 →
        execute() → 缓存写入 → after_execute 钩子 → 结果截断

    为什么设计这么多钩子和属性：
    - 让子类可以精确控制工具的每个阶段行为
    - 大部分钩子有默认实现（返回 None），子类只需覆盖自己需要的
    - 不增加子类的实现负担
    """

    # 每个工具实例持有自己的上下文（由 Runtime 在执行前设置）
    # 注意：工具实例通常是单例（注册在 ALL_TOOLS 中），
    # 所以每次执行前都要重新 set_context，避免残留旧上下文
    _context: ToolContext | None = None

    # ============================================================
    # Per-tool 配额系统
    # 每个工具可以独立配置自己的 token 配额上限，
    # 默认 0 = 不限额，只统计使用量。
    # ============================================================
    _quota_used: int = 0  # 当前会话中该工具已使用的 token 数

    @property
    def quota_limit(self) -> int:
        """
        工具的 token 配额上限。

        控制该工具在单次会话中最多可以消耗多少 token（结果文本的估算值）。
        默认 0 表示不限额，仅统计使用量。
        子类可覆盖此属性设置具体限额。

        用途：
        - 计算器等轻量工具可设较小限额（500 tokens）
        - 文件读取等重量工具可设较大限额（5000 tokens）
        - 修改为 0 即取消对该工具的限额
        """
        return 0

    @property
    def quota_used(self) -> int:
        """获取当前会话中该工具的累计 token 使用量（结果 token 估算）。"""
        return self._quota_used

    def reset_quota(self):
        """
        重置工具的 quota 计数器。

        在新会话开始时调用，清空该工具的累计使用量。
        由 Runtime 在创建/切换会话时自动调用。
        """
        self._quota_used = 0

    def _check_quota(self, estimated_tokens: int = 0) -> bool:
        """
        检查工具配额是否还有剩余。

        Args:
            estimated_tokens: 本次调用预计消耗的 token 数（默认 0）

        Returns:
            True = 配额充足，可以执行
            False = 配额已超限，需要拒绝执行
        """
        if self.quota_limit <= 0:
            return True  # 不限额
        return (self._quota_used + estimated_tokens) <= self.quota_limit

    def _add_quota_usage(self, tokens: int):
        """增加工具的 quota 使用量。"""
        if tokens > 0:
            self._quota_used += tokens

    @property
    @abstractmethod
    def name(self) -> str:
        """
        工具的唯一标识名称。

        这个名称就是 LLM 在 tool_calls 中使用的名称，必须：
        - 唯一（不与其他工具重名）
        - 小写英文字母 + 下划线（如 calculator, read_file）
        - 语义明确（LLM 看到名称就知道干什么）
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        工具的功能描述，会展示给 LLM。

        描述的质量直接影响 LLM 是否能正确决定何时调用此工具。
        好的描述应该包含：
        - 工具的功能（"安全的数学表达式计算器"）
        - 适用场景（"当用户需要计算时"）
        - 使用限制（"不支持复数运算"）
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        参数的 JSON Schema 定义。

        LLM 根据此 Schema 生成函数调用的参数。
        格式遵循 OpenAI Function Calling 规范：
        {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }

        属性越多（description 越详细），LLM 生成的参数越准确。
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具的核心逻辑。

        这是工具唯一必须实现的业务方法。
        参数名必须与 parameters schema 中的属性名一致。

        Args:
            **kwargs: 工具参数，由 LLM 根据 parameters schema 生成

        Returns:
            ToolResult: 包含执行结果或错误信息
        """
        ...

    async def execute_async(self, **kwargs) -> ToolResult:
        """
        异步执行工具（可选重写）。

        对于 I/O 密集型的工具（如网络请求、大文件操作），
        重写此方法可获得更好的并发性能。
        默认回退到同步 execute() 在线程池中执行。
        """
        return self.execute(**kwargs)

    def before_execute(self, kwargs: dict, context: ToolContext) -> dict | ToolResult | None:
        """
        执行前置钩子（可选重写）。

        可用于：
        - 注入额外参数（如自动添加时间戳）
        - 参数安全检查（如路径合法性检查）
        - 鉴权校验（如检查用户是否有权限）
        - 跳过执行（返回 ToolResult 直接结束）

        Returns:
            dict: 修改后的参数字典（替换原参数）
            ToolResult: 直接返回此结果（跳过 execute）
            None: 继续正常执行
        """
        return None

    def after_execute(self, result: ToolResult, context: ToolContext) -> ToolResult | None:
        """
        执行后置钩子（可选重写）。

        可用于：
        - 结果脱敏（如隐藏文件路径中的用户名）
        - 结果格式化（如添加统一的前缀/后缀）
        - 结果缓存（如将结果写入额外的缓存文件）

        Returns:
            ToolResult: 修改后的结果（替换原结果）
            None: 使用原始结果
        """
        return None

    def on_error(self, error: ToolError, context: ToolContext) -> ToolResult | None:
        """
        错误处理钩子（可选重写）。

        可用于：
        - 错误信息美化（如将技术错误转为人话）
        - 降级返回缓存（如果之前有相同参数的缓存结果）

        Returns:
            ToolResult: 替换的错误结果
            None: 继续使用默认错误处理
        """
        return None

    # ============================================================
    # 自文档化
    # ============================================================

    @property
    def examples(self) -> list[dict]:
        """
        工具调用示例列表。

        返回格式:
        [
            {
                "description": "示例描述（LLM 理解这个示例在做什么）",
                "arguments": {"param1": "value1"},
                "result_preview": "预期结果预览（可选）",
            },
        ]

        这些示例会被注入到 OpenAI schema 的 description 中，
        帮助 LLM 更好理解工具的用法和参数格式。
        """
        return []

    @property
    def usage_guide(self) -> str:
        """
        工具使用指南（可选的详细说明）。

        描述何时使用、何时不使用此工具，以及最佳实践。
        会附加到工具描述中，增加 LLM 使用工具的准确性。
        """
        return ""

    # ============================================================
    # 动态参数
    # ============================================================

    def get_dynamic_parameters(self, current_params: dict) -> dict:
        """
        根据已填参数动态调整参数 Schema。

        例如：文件读取工具可根据 file_path 后缀调整 max_lines 的默认值。
        - file_path="data.csv" → 默认 max_lines=100（CSV 行多）
        - file_path="config.py" → 默认 max_lines=200（代码文件适中）

        Args:
            current_params: 当前已填写的参数字典

        Returns:
            dict: 与 parameters() 的 properties 格式相同的字典。
                  空 dict 表示不调整。
        """
        return {}

    # ============================================================
    # 缓存
    # ============================================================

    @property
    def cache_ttl(self) -> float:
        """
        缓存生存时间（秒）。

        0 表示不缓存。
        纯函数工具（计算器、时间）应启用缓存，减少重复 API 调用。
        有副作用的工具（文件读写、任务管理等）不应缓存或设置极短 TTL。
        """
        return 0.0

    _cache_instance: _ToolCache | None = None

    def _get_cache(self) -> _ToolCache | None:
        """获取工具缓存实例（懒加载）。"""
        if self.cache_ttl > 0:
            if self._cache_instance is None:
                self._cache_instance = _ToolCache(ttl=self.cache_ttl)
            return self._cache_instance
        return None

    # ============================================================
    # Schema & 执行
    # ============================================================

    @property
    def max_result_chars(self) -> int:
        """
        返回给 LLM 的结果最大字符数。

        0 表示不限制长度，工具返回完整内容。
        子类可按需覆盖此属性。
        """
        return 0

    @property
    def max_result_tokens(self) -> int:
        """
        返回给 LLM 的结果最大 token 数（估算值）。

        0 表示不限制长度，工具返回完整内容。
        子类可按需覆盖此属性。
        """
        return 0

    @property
    def timeout(self) -> float:
        """
        工具执行超时时间（秒）。

        超过此时间会强制中断并返回超时错误。
        子类应按执行耗时覆盖此属性：
        - 快速 CPU 操作（计算器、搜索、时间）：5 秒
        - 文件 I/O 操作：30 秒
        """
        return 30.0

    def set_context(self, context: ToolContext | None):
        """设置工具执行上下文（由 Runtime 在执行前调用）。"""
        self._context = context

    def get_context(self) -> ToolContext:
        """获取当前执行上下文。未设置时返回空上下文。"""
        if self._context is None:
            return ToolContext()
        return self._context

    def to_openai_schema(self) -> dict:
        """
        转换为 OpenAI function calling 格式。

        自文档化：如果存在 examples 或 usage_guide，会附加到描述中。
        这使得 LLM 能看到使用示例，不需要额外的 prompt 工程。

        Returns:
            符合 OpenAI tools API 的字典格式：
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "... + usage_guide + examples",
                    "parameters": {...}
                }
            }
        """
        desc = self.description
        if self.usage_guide:
            desc += f"\n\n使用说明: {self.usage_guide}"

        # 如果有示例，在描述中附加一个典型示例
        examples = self.examples
        if examples:
            ex = examples[0]
            desc += (
                f"\n\n调用示例: {ex.get('description', '')}\n"
                f"  参数: {json.dumps(ex.get('arguments', {}), ensure_ascii=False)}"
            )

        schema = self.parameters
        # 合并动态参数
        dyn_params = self.get_dynamic_parameters({})
        if dyn_params:
            props = schema.get("properties", {}).copy()
            props.update(dyn_params)
            schema = {**schema, "properties": props}

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": schema,
            },
        }

    def safe_execute(self, arguments: str | dict,
                     context: ToolContext | None = None) -> ToolResult:
        """
        安全执行工具 — 完整的执行流水线。

        这是工具调用的唯一入口，完整的执行流程：

        1. 设置上下文
        2. 参数解析（JSON 字符串 → dict）
        3. 参数校验（类型/必须/枚举值）
        4. before_execute 钩子（工具级 + 全局注册表）
        5. 缓存检查（命中则直接返回）
        6. execute() 核心逻辑
        7. 缓存写入
        8. after_execute 钩子（工具级 + 全局注册表）
        9. 结果截断（token 级别 + 字符级别双重保护）

        每一层都可能中断或修改结果，
        形成"洋葱模型"的多层保护架构。

        Args:
            arguments: JSON 字符串或字典格式的参数
            context: 工具执行上下文（由 Runtime 传入）

        Returns:
            ToolResult: 执行结果或错误信息（结果已截断）
        """
        # 设置上下文
        if context is not None:
            self._context = context
        ctx = self.get_context()

        try:
            # ---- 1. 参数解析 ----
            if isinstance(arguments, str):
                args_dict = json.loads(arguments) if arguments.strip() else {}
            else:
                args_dict = arguments

            # ---- 2. 参数校验 ----
            # 根据 parameters schema 校验参数的类型、必须项、枚举值
            cleaned, errors = self._validate_params(self.parameters, args_dict)
            if errors:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"[{ErrorCode.INVALID_PARAMS}] 参数校验失败:\n" + "\n".join(f"  - {e}" for e in errors),
                    metadata={"error_code": ErrorCode.INVALID_PARAMS, "validation_errors": errors},
                )

            # ---- 3. before_execute 钩子（工具级） ----
            try:
                hook_result = self.before_execute(cleaned, ctx)
            except ToolError as e:
                return e.to_tool_result()

            if isinstance(hook_result, ToolResult):
                return hook_result
            if isinstance(hook_result, dict):
                cleaned = hook_result  # 钩子修改了参数

            # ---- 4. before_execute 钩子（全局注册表） ----
            try:
                global_hook_result = HOOKS.run_before(self.name, ctx, cleaned)
            except ToolError as e:
                return e.to_tool_result()
            if isinstance(global_hook_result, ToolResult):
                return global_hook_result
            if isinstance(global_hook_result, dict):
                cleaned = global_hook_result

            # ---- 5. 配额检查（per-tool） ----
            # 估算本次调用可能消耗的 token 数（以最大结果 token 为参考）
            estimated_cost = self.max_result_tokens if self.max_result_tokens > 0 else 200
            if not self._check_quota(estimated_cost):
                return ToolResult(
                    success=False,
                    result="",
                    error=(
                        f"[{ErrorCode.RATE_LIMITED}] 工具 '{self.name}' 的 token 配额已耗尽"
                        f"（已用: {self._quota_used}, 限额: {self.quota_limit}）。"
                        f"如需继续使用，请调整该工具的 quota_limit 设置或开始新会话。"
                    ),
                    metadata={
                        "error_code": ErrorCode.RATE_LIMITED,
                        "quota_used": self._quota_used,
                        "quota_limit": self.quota_limit,
                        "tool_name": self.name,
                    },
                )

            # ---- 6. 缓存检查 ----
            cache = self._get_cache()
            if cache:
                cached = cache.get(cleaned)
                if cached is not None:
                    cached.metadata["cached"] = True
                    return cached

            # ---- 7. 执行核心逻辑 ----
            result = self.execute(**cleaned)

            # ---- 8. 缓存写入 ----
            if cache and result.success:
                cache.set(cleaned, result)

            # ---- 9. after_execute 钩子（工具级） ----
            try:
                after_result = self.after_execute(result, ctx)
                if isinstance(after_result, ToolResult):
                    result = after_result
            except ToolError as e:
                return e.to_tool_result()

            # ---- 10. after_execute 钩子（全局注册表） ----
            try:
                global_after = HOOKS.run_after(self.name, ctx, result)
                if isinstance(global_after, ToolResult):
                    result = global_after
            except ToolError as e:
                return e.to_tool_result()

            # ---- 11. 结果截断保护 ----
            # 双重截断：token 级别（精确控制上下文占用）+ 字符级别（绝对上限）
            if result.success:
                # Token 级别截断（估算值）
                if self.max_result_tokens > 0:
                    warn = config.is_token_truncate_warn_enabled()
                    truncated = config.truncate_by_tokens(
                        result.result, self.max_result_tokens, warn=warn
                    )
                    if truncated != result.result:
                        result.result = truncated
                        result.metadata["truncated"] = True

                # 字符级别截断（兜底）
                if self.max_result_chars > 0 and len(result.result) > self.max_result_chars:
                    # 在换行符处截断，保证行完整性
                    truncated = result.result[:self.max_result_chars]
                    last_newline = truncated.rfind("\n")
                    if last_newline > 0:
                        truncated = truncated[:last_newline]
                    elif last_newline == 0:
                        truncated = ""

                    total_lines = result.result.count("\n") + 1
                    shown_lines = truncated.count("\n") + 1 if truncated else 0

                    # 构建截断提示：优先使用工具声明的 continuation 指引
                    if result.continuation:
                        # 工具已声明如何恢复后续内容，使用工具给出的提示
                        hint = result.continuation.get("hint", "")
                        result.result = truncated + (
                            f"\n\n[...结果过长，已截断为前 {self.max_result_chars} 字符，"
                            f"当前显示 {shown_lines}/{total_lines} 行。\n"
                            f"{hint}]"
                        )
                    else:
                        # 工具未声明恢复方式，提示内容已丢失
                        result.result = truncated + (
                            f"\n\n[...结果过长，已截断为前 {self.max_result_chars} 字符，"
                            f"当前显示 {shown_lines}/{total_lines} 行，"
                            f"完整结果共 {len(result.result)} 字符。\n"
                            f"该工具不支持分页读取，被截断的部分无法继续获取。]"
                        )
                    result.metadata["truncated"] = True

            # ---- 11. 配额使用量统计 ----
            # 对成功执行的结果，估算其 token 消耗并计入工具的 quota
            if result.success:
                result_tokens = config.get_estimated_tokens(result.result)
                self._add_quota_usage(result_tokens)
                result.metadata["quota_used"] = self._quota_used
                result.metadata["quota_limit"] = self.quota_limit

            return result

        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"[{ErrorCode.INVALID_PARAMS}] 参数解析失败: {e}",
                metadata={"error_code": ErrorCode.INVALID_PARAMS},
            )
        except TypeError as e:
            return ToolResult(
                success=False,
                result="",
                error=f"[{ErrorCode.INVALID_PARAM_TYPE}] 参数类型错误: {e}",
                metadata={"error_code": ErrorCode.INVALID_PARAM_TYPE},
            )
        except ToolError as e:
            # 结构化错误 → 尝试 on_error 钩子
            try:
                on_error_result = self.on_error(e, ctx)
                if isinstance(on_error_result, ToolResult):
                    return on_error_result
                global_error = HOOKS.run_on_error(self.name, ctx, e)
                if isinstance(global_error, ToolResult):
                    return global_error
            except Exception:
                pass
            return e.to_tool_result()
        except Exception as e:
            # 非结构化异常 → 包装为 ToolError（统一错误格式）
            tool_error = ToolError(
                ErrorCode.EXECUTION_ERROR,
                f"{type(e).__name__}: {e}",
            )
            return tool_error.to_tool_result()

    def clear_cache(self):
        """手动清空工具缓存。"""
        if self._cache_instance:
            self._cache_instance.clear()

    # ============================================================
    # 参数校验
    # ============================================================

    @staticmethod
    def _validate_params(schema: dict, raw_args: dict) -> tuple[dict, list[str]]:
        """
        根据 JSON Schema 校验/清洗参数。

        处理项：
        - 检查必须参数是否存在
        - 注入可选参数的默认值
        - 类型校验与自动转换（如 "123" → 123）
        - enum 值校验（只能取允许的值）
        - 过滤 schema 未定义的冗余参数（防止 LLM 乱传参数）

        Args:
            schema: parameters JSON Schema
            raw_args: LLM 传入的原始参数字典

        Returns:
            (cleaned_args, errors) 清洗后的参数和错误列表
        """
        cleaned: dict = {}
        errors: list[str] = []
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        # 检查必须参数
        for name in required:
            if name not in raw_args or raw_args[name] is None:
                errors.append(f"缺少必须参数 '{name}'")

        # 清洗可选参数
        for name, prop in props.items():
            if name not in raw_args or raw_args[name] is None:
                if "default" in prop:
                    cleaned[name] = prop["default"]
                continue

            value = raw_args[name]
            expected_type = prop.get("type", "")
            prop_enum = prop.get("enum")

            converted, ok = BaseTool._coerce_type(value, expected_type, name, errors)
            if not ok:
                continue

            # enum 值校验
            if prop_enum is not None and converted not in prop_enum:
                errors.append(
                    f"参数 '{name}' 的值 '{converted}' 不在允许范围内: {prop_enum}"
                )
                continue

            cleaned[name] = converted

        return cleaned, errors

    @staticmethod
    def _coerce_type(value: Any, expected_type: str, param_name: str,
                     errors: list[str]) -> tuple[Any, bool]:
        """
        类型自动转换。

        JSON Schema → Python 类型的映射：
            string  → str
            integer → int
            number  → float
            boolean → bool
            array   → list
            object  → dict

        为什么做自动转换：
        - LLM 生成的参数可能类型不精确（如把 number 写成 string）
        - 自动转换让工具更鲁棒，减少因类型不匹配导致的失败
        - 转换失败时才报错，不增加工具的复杂度
        """
        TYPE_MAP = {
            "string": str, "integer": int, "number": float,
            "boolean": bool, "array": list, "object": dict,
        }

        target = TYPE_MAP.get(expected_type)
        if target is None:
            return value, True

        if isinstance(value, target):
            return value, True

        try:
            # 布尔值特殊处理：字符串 "true"/"false" 等
            if target is bool and isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    return True, True
                elif value.lower() in ("false", "0", "no"):
                    return False, True
                else:
                    raise ValueError(f"无法转为 bool: '{value}'")

            elif target is int:
                if isinstance(value, float):
                    return int(value), True
                return int(value), True

            elif target is float:
                return float(value), True

            elif target is str:
                return str(value), True

            elif target is list:
                if isinstance(value, (list, tuple)):
                    return list(value), True
                return [value], True

            elif target is dict:
                if isinstance(value, dict):
                    return value, True
                return {}, True

        except (ValueError, TypeError, OverflowError) as e:
            errors.append(
                f"参数 '{param_name}' 类型错误: 期望 {expected_type}, "
                f"收到 {type(value).__name__}({value!r}), 转换失败: {e}"
            )
            return None, False

        errors.append(
            f"参数 '{param_name}' 类型错误: 期望 {expected_type}, "
            f"收到 {type(value).__name__}({value!r})"
        )
        return None, False


# ============================================================
# 依赖检查装饰器
# ============================================================

def require_import(package_name: str, pip_name: str = None):
    """
    装饰器：检查可选依赖是否已安装，缺失时返回友好提示。

    用于需要在方法内部 import 可选依赖的工具方法。
    在 import 前检查依赖是否存在，避免抛出难以理解的 ModuleNotFoundError。
    支持 ToolError 结构化错误码。

    为什么不是直接 import 然后 except ImportError：
    - 那样需要在每个工具方法中写 try/except，重复代码
    - 装饰器方式声明式表达，一行搞定
    - 统一的错误信息格式（包含 pip install 命令）

    Args:
        package_name: import 时使用的包名（如 "fitz"）
        pip_name: pip install 时使用的包名（如 "PyMuPDF"），
                  不传则与 package_name 相同

    用法:
        class FileReaderTool(BaseTool):
            @require_import("fitz", "PyMuPDF")
            def _read_pdf(self, path, max_lines):
                import fitz
                ...
    """
    pip_name = pip_name or package_name

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if importlib.util.find_spec(package_name) is not None:
                return func(self, *args, **kwargs)

            try:
                importlib.import_module(package_name)
                return func(self, *args, **kwargs)
            except ImportError:
                return ToolResult(
                    success=False,
                    result="",
                    error=(
                        f"[{ErrorCode.DEPENDENCY_MISSING}] "
                        f"需要安装 '{pip_name}' 库才能处理此格式。\n"
                        f"请运行: pip install {pip_name}"
                    ),
                    metadata={"error_code": ErrorCode.DEPENDENCY_MISSING},
                )
        return wrapper
    return decorator
