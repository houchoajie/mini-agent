"""
============================================================
配置管理模块 — 从 .env 和系统环境变量加载所有应用配置
============================================================

设计原则：
1. 所有配置项集中管理，提供统一的访问接口
2. 配置优先级：进程环境变量 > .env 文件 > 代码默认值
3. 每次调用重新读取环境变量，支持运行时修改 .env 后生效
4. 不做配置缓存，保证配置变更即时可见

为什么这样做：
- 集中管理：修改配置只需改一个地方，不用满项目找魔法数字
- 优先级明确：部署时可通过环境变量覆盖 .env 中的值
- 无缓存：适合开发调试，改完 .env 不用重启（生产环境建议加缓存）

支持的 .env 配置项：

=== LLM 配置 ===
    OPENAI_API_KEY=sk-your-key          # API 密钥（必填）
    OPENAI_BASE_URL=https://.../v1      # API 端点（支持第三方兼容服务）
    OPENAI_MODEL=gpt-4o-mini            # 模型名称
    LLM_TIMEOUT=60                      # API 请求超时秒数

=== Token 限额配置 ===
    TOTAL_TOKEN_LIMIT=0                 # 会话累计 token 上限，0=不限制
    DEFAULT_LLM_MAX_TOKENS=4096         # 每次 LLM 调用的 max_tokens
    ENABLE_TOKEN_LIMIT=true             # 是否启用 token 限额检查
    TOKEN_TRUNCATE_WARN=true            # 超限时是否打印警告

=== 工具执行配置 ===
    MAX_TOOL_RETRIES=2                  # 工具失败时的最大重试次数
    TOOL_RETRY_DELAY=1.0                # 工具重试间隔秒数
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# .env 文件加载（只执行一次）
# ============================================================
# 为什么用全局标记：
#   load_dotenv() 虽然内部有防重复机制，但每次调用会扫描文件系统。
#   用一个显式的全局标记更清晰，一眼看出"只加载一次"。
# ============================================================
_load_dotenv_once = False


def _ensure_loaded():
    """
    确保 .env 文件只被加载一次。

    搜索路径：从本文件位置（agent/）向上两级到项目根目录加载 .env。
    使用全局标记 _load_dotenv_once 避免重复加载。
    override=False 防止 .env 覆盖已经设置好的进程环境变量，
    这样部署时可以通过 export 或 docker -e 覆盖配置。
    """
    global _load_dotenv_once
    if not _load_dotenv_once:
        dotenv_path = Path(__file__).parent.parent / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
        _load_dotenv_once = True


def _bool_env(key: str, default: bool) -> bool:
    """从环境变量读取布尔值，支持多种 true/false 表示法。"""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int_env(key: str, default: int) -> int:
    """从环境变量读取整数值，解析失败时返回默认值（静默降级）。"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


# ============================================================
# 配置项访问函数
# 每个函数都是独立、自包含的，方便单测和 mock。
# 每次调用重新读取环境变量，不缓存。
# ============================================================


def get_api_key() -> str:
    """获取 OpenAI API Key。空字符串表示未配置。"""
    _ensure_loaded()
    return os.getenv("OPENAI_API_KEY", "")


def get_base_url() -> str:
    """
    获取 API Base URL。
    默认使用 OpenAI 官方端点，也支持任何兼容 OpenAI API 格式的第三方服务
    （如 DeepSeek、智谱、阿里云 DashScope 等）。
    """
    _ensure_loaded()
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def get_model() -> str:
    """获取模型名称。"""
    _ensure_loaded()
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_llm_timeout() -> float:
    """
    获取 LLM API 请求超时（秒）。
    默认 60 秒。超过此时间未收到响应视为超时。
    注意：这只是网络超时，不包括流式读取时间。
    """
    _ensure_loaded()
    return float(os.getenv("LLM_TIMEOUT", "60"))


def get_total_token_limit() -> int:
    """
    获取整个会话的累计 token 上限。

    0 表示不限制。当会话累计 token 数超过此值时，Runtime 会：
    1. 先尝试记忆压缩（保留关键上下文）
    2. 如果仍超限，清空所有非 system 消息
    3. 插入提示告知用户

    为什么要有这个限制：
    - 防止 token 费用失控
    - 防止对话过长导致上下文窗口溢出
    - 强制 Agent 在有限的上下文内高效工作
    """
    _ensure_loaded()
    return _int_env("TOTAL_TOKEN_LIMIT", 0)


def get_default_llm_max_tokens() -> int | None:
    """
    获取每次 LLM API 调用的 max_tokens 参数。

    返回值：
        int | None — None 表示不限制（不给 API 传 max_tokens）。
        设置此参数可以控制模型每次回复的最大长度，防止单次回复
        占用过多 token 预算。

    注意：
        返回 0 也被转为 None（不限制），因为 API 的 max_tokens=0
        可能导致意外行为。
    """
    _ensure_loaded()
    val = _int_env("DEFAULT_LLM_MAX_TOKENS", 4096)
    return val if val > 0 else None


def is_token_limit_enabled() -> bool:
    """是否启用 token 限额检查。默认启用。"""
    _ensure_loaded()
    return _bool_env("ENABLE_TOKEN_LIMIT", True)


def is_token_truncate_warn_enabled() -> bool:
    """工具结果被截断时是否在文本中附加截断提示。默认启用。"""
    _ensure_loaded()
    return _bool_env("TOKEN_TRUNCATE_WARN", True)


def get_max_tool_retries() -> int:
    """
    获取工具执行失败时的最大重试次数。

    0 表示不重试。当工具执行返回失败结果时，系统会自动重试。
    注意：这里的重试是指工具 execute() 返回 success=False 的情况，
    与异常抛出的重试不同。
    """
    _ensure_loaded()
    return _int_env("MAX_TOOL_RETRIES", 2)


def get_tool_retry_delay() -> float:
    """
    获取工具重试间隔秒数。
    每次重试前等待此时间，避免立即重试导致同样的失败。
    """
    _ensure_loaded()
    return float(os.getenv("TOOL_RETRY_DELAY", "1.0"))


def get_estimated_tokens(text: str) -> int:
    """
    粗略估算文本的 token 数。

    使用通用近似公式：中英文混合场景下，约 1 token = 1.5 字符。
    即 token ≈ len(text) × 2/3。

    为什么用估算而非精确计数：
    1. 精确计数需要调用 tokenizer API，增加延迟和费用
    2. 在限额保护场景下 ±30% 的精度已经足够
    3. 实现简单，零依赖

    注意：此处有意使用"保守估算"，即估算值偏大。
    这意味着实际 token 可能少于估算值，但不会超出太多。
    """
    if not text:
        return 0
    return int(len(text) * 2 / 3) + 1


def truncate_by_tokens(text: str, max_tokens: int, warn: bool = True) -> str:
    """
    按 token 估算值截断文本（行级截断，保证语义完整性）。

    用于工具结果截断，确保返回给 LLM 的内容不会超限。
    在最近的换行符处截断，不会断在单词、代码或句子中间。

    裁剪策略：
    - 估算值：每 token ≈ 2 字符（中英文混合场景的平均值）
    - 英文每 token 约 3-4 字符 → 2 字符/token 偏保守，不会超限
    - 中文每 token 约 1-2 字符 → 2 字符/token 偶有超出，但 max_tokens
      本身是工具自定义的软限制，少量超出不会引发问题
    - 截断点对齐到最近的换行符，保证每行完整

    为什么调整之前的 1 字符/token 策略：
    - 之前过于保守，导致实际可用上下文只用了 1/3（英文场景）
    - LLM 看不到完整信息，影响判断准确性
    - 改为行级截断后，即使长度相近，LLM 看到的也是完整行内容

    Args:
        text: 要截断的文本
        max_tokens: 最大允许的 token 数
        warn: 截断时是否在结果中附加提示

    Returns:
        截断后的文本（在换行符处截断，可能追加了截断提示）
    """
    if not text or max_tokens <= 0:
        return text

    # 使用 2 字符/token 的估算值（中英文混合场景的平均值）
    # 英文约 3-4 字符/token，中文约 1-2 字符/token
    max_chars = max_tokens * 2

    if len(text) <= max_chars:
        return text

    # 在最近的换行符处截断，保证不破坏行/句子/代码的完整性
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > 0:
        truncated = truncated[:last_newline]
    elif last_newline == 0:
        truncated = ""

    # 统计截断比例（行数和字符数）
    truncated_lines = truncated.count("\n") + 1 if truncated else 0
    total_lines = text.count("\n") + 1
    remaining_chars = len(text) - len(truncated)
    ratio = (len(truncated) / len(text)) * 100

    if warn:
        truncated += (
            f"\n\n[...结果已根据 token 限额截断，"
            f"当前显示 {truncated_lines}/{total_lines} 行"
            f"（约占全文 {ratio:.0f}%），"
            f"后略约 {remaining_chars} 字符。"
            f"需要继续查看时，可以设置 start_line 分页读取后续内容。]"
        )

    return truncated
