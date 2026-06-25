"""
============================================================
工具执行器 — 超时控制、自动重试、异步执行、上下文注入
============================================================

本模块是对工具调用的"执行层"封装，提供：
1. 超时控制：使用 asyncio.wait_for 在指定时间内未完成则强制取消
2. 自动重试：工具返回失败结果时自动按配置次数重试
3. 异步执行：检测工具是否有 execute_async，优先用协程
4. 上下文注入：执行前设置 ToolContext（用户名、会话ID等）

执行策略：
    1. 检查工具是否有 execute_async（异步）
    2. 有 → 在事件循环中 await 执行
    3. 无 → 在 ThreadPoolExecutor 中执行同步方法
    4. 失败 → 按配置次数重试（默认 2 次）

为什么需要单独的 Executor 层：
    - Runtime 不需要关心"工具怎么执行"的细节
    - 超时、重试、异步检测等横切关注点集中处理
    - 便于单测：可以 mock Executor 来测试 Runtime 的重试行为
"""

import time
import asyncio
import concurrent.futures
from agent.tools.base import BaseTool, ToolResult, ToolContext, ErrorCode
from agent import config


# 全局线程池（复用线程，避免反复创建销毁的开销）
# max_workers=4：同时最多 4 个工具并行执行
# 为什么不是更多：工具通常是 IO 密集型（文件读写、API 调用），
# 4 个线程已经能充分利用 IO 等待时间
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="tool_exec",
)


def _is_async_override(tool: BaseTool) -> bool:
    """
    检测工具是否重写了 execute_async 方法。

    默认的 BaseTool.execute_async 只是同步调用 self.execute()，
    如果子类重写了它（即 cls.execute_async is not base_method），
    则优先使用异步执行。

    为什么用 is 检测而非 hasattr：
    - hasattr 无法区分"继承的方法"和"重写的方法"
    - is 检测精确判断子类是否真正重写了该方法
    """
    cls = type(tool)
    base_method = BaseTool.execute_async
    return cls.execute_async is not base_method


async def execute_async(
    tool: BaseTool,
    arguments: str | dict,
    timeout: float | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """
    异步执行工具（优先使用 execute_async）。

    执行策略：
    - 如果工具重写了 execute_async → 在原线程中 await
    - 如果工具只有 execute → 在 ThreadPoolExecutor 中执行

    重试策略：
    - 只有工具返回 success=False 时才重试
    - 抛出异常时直接返回（不走重试，因为异常可能不可恢复）
    - 重试间隔由 TOOL_RETRY_DELAY 控制

    Args:
        tool: 工具实例
        arguments: JSON 字符串或字典格式的参数
        timeout: 超时秒数。None 使用工具的默认 timeout 属性
        context: 工具执行上下文

    Returns:
        ToolResult: 执行结果（已包含耗时信息）
    """
    effective_timeout = timeout if timeout is not None else tool.timeout
    max_retries = config.get_max_tool_retries()
    retry_delay = config.get_tool_retry_delay()

    # 设置上下文
    if context is not None:
        tool.set_context(context)

    # 配额检查（per-tool）：如果工具设置了 quota_limit，先检查是否已超限
    if tool.quota_limit > 0 and tool.quota_used >= tool.quota_limit:
        return ToolResult(
            success=False,
            result="",
            error=(
                f"[{ErrorCode.RATE_LIMITED}] 工具 '{tool.name}' 的 token 配额已耗尽"
                f"（已用: {tool.quota_used}, 限额: {tool.quota_limit}）。"
                f"如需继续使用，请调整该工具的 quota_limit 设置或开始新会话。"
            ),
            metadata={
                "error_code": ErrorCode.RATE_LIMITED,
                "quota_used": tool.quota_used,
                "quota_limit": tool.quota_limit,
                "tool_name": tool.name,
            },
        )

    for attempt in range(max_retries + 1):
        attempt_start = time.time()

        try:
            if _is_async_override(tool):
                # 使用原生异步执行（工具自身提供了协程实现）
                result = await asyncio.wait_for(
                    tool.execute_async(
                        **(arguments if isinstance(arguments, dict)
                           else _parse_args(arguments))
                    ),
                    timeout=effective_timeout,
                )
            else:
                # 使用线程池执行同步方法
                # 将同步工具包装为协程，统一返回类型
                result = await asyncio.get_event_loop().run_in_executor(
                    _EXECUTOR,
                    tool.safe_execute,
                    arguments,
                    context,
                )

            # 记录耗时
            elapsed_ms = round((time.time() - attempt_start) * 1000)
            if isinstance(result, ToolResult):
                result.metadata["elapsed_ms"] = elapsed_ms

            # 成功 → 返回
            if result.success:
                if attempt > 0:
                    result.metadata["retried"] = True
                    result.metadata["retry_count"] = attempt
                return result

            # 失败 → 需要重试？
            if attempt < max_retries:
                _log_retry(tool, attempt, result.error, retry_delay)
                await asyncio.sleep(retry_delay)
                continue
            else:
                if max_retries > 0:
                    result.result += f"\n\n[已重试 {max_retries} 次仍失败]"
                    result.metadata["retried"] = True
                    result.metadata["retry_count"] = max_retries
                return result

        except asyncio.TimeoutError:
            elapsed_ms = round((time.time() - attempt_start) * 1000)
            return ToolResult(
                success=False,
                result="",
                error=f"[TIMEOUT] 工具 '{tool.name}' 执行超时（{effective_timeout} 秒）",
                metadata={"elapsed_ms": elapsed_ms, "timeout": effective_timeout, "error_code": "TIMEOUT"},
            )

        except Exception as e:
            elapsed_ms = round((time.time() - attempt_start) * 1000)
            return ToolResult(
                success=False,
                result="",
                error=f"[EXECUTION_ERROR] 工具执行异常: {type(e).__name__}: {e}",
                metadata={"elapsed_ms": elapsed_ms, "error_code": "EXECUTION_ERROR"},
            )

    # 兜底（理论上不会执行到这里）
    return ToolResult(
        success=False, result="",
        error="[EXECUTION_ERROR] 工具执行失败（所有尝试已耗尽）",
        metadata={"error_code": "EXECUTION_ERROR"},
    )


def execute_sync(
    tool: BaseTool,
    arguments: str | dict,
    timeout: float | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """
    同步执行工具（兼容现有代码的调用方式）。

    内部通过事件循环调用 execute_async。
    如果当前线程已有事件循环（如在 asyncio 环境中），
    则创建新线程和新事件循环来执行，避免阻塞当前循环。

    Args:
        tool: 工具实例
        arguments: 参数
        timeout: 超时
        context: 上下文

    Returns:
        ToolResult: 执行结果
    """
    try:
        loop = asyncio.get_running_loop()
        # 已有事件循环 → 创建新循环在线程中执行
        # 为什么：如果在异步上下文中直接 run_until_complete，
        # 会报 "Cannot run the event loop while it's running"
        import threading
        result_container = []

        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                r = new_loop.run_until_complete(
                    execute_async(tool, arguments, timeout, context)
                )
                result_container.append(r)
            finally:
                new_loop.close()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=(timeout or tool.timeout) + 10)
        if result_container:
            return result_container[0]
        return ToolResult(
            success=False, result="",
            error="[TIMEOUT] 工具执行超时",
            metadata={"error_code": "TIMEOUT"},
        )

    except RuntimeError:
        # 没有事件循环 → 直接创建并使用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                execute_async(tool, arguments, timeout, context)
            )
        finally:
            loop.close()


def _parse_args(arguments: str) -> dict:
    """解析 JSON 字符串为字典。解析失败返回空字典。"""
    import json
    try:
        return json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        return {}


def _log_retry(tool: BaseTool, attempt: int, error: str | None, delay: float):
    """记录重试信息到控制台。"""
    err_preview = error or "未知错误"
    print(
        f"🔄 [RETRY] 工具 '{tool.name}' 第 {attempt + 1} 次失败，"
        f"{delay}s 后重试... 错误: {err_preview}"
    )


def shutdown():
    """关闭线程池，释放资源。"""
    _EXECUTOR.shutdown(wait=True)
