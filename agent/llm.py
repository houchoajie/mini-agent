"""
============================================================
LLM 客户端 — 封装 OpenAI（及兼容 API）的 Chat Completions 调用
============================================================

本模块是 Agent 与 LLM 之间的桥梁，负责：
1. 调用 OpenAI Chat Completions API（支持 Function Calling）
2. 自动重试（指数退避策略）
3. 流式输出（逐 token 返回，提升用户体验）
4. 流式工具调用检测（边输出文本边判断是否需要调工具）

设计的核心考量：
- 对上层（Runtime）屏蔽 API 差异：无论是 OpenAI、DeepSeek 还是
  其他兼容服务，Runtime 看到的都是统一的 chat_async() / chat_stream_detect_tools_async() 接口
- 失败有兜底：网络波动、限流等瞬态异常由重试机制自动处理
- Token 统计：每次调用记录 token 用量，供限额检查和审计

使用示例：
    llm = LLMClient()
    response = await llm.chat_async(
        messages=[{"role": "user", "content": "你好"}],
        tools=[{"type": "function", ...}],
    )
    # response = {"role": "assistant", "content": "...", "tool_calls": [...]}
"""

import os
import json
import asyncio
from openai import AsyncOpenAI
from agent.trace import TraceLogger
from agent import config


class LLMClient:
    """
    LLM API 客户端封装。

    封装 OpenAI Chat Completions API，提供统一的调用接口。
    支持 function calling（工具调用）、自动重试、流式输出。

    为什么单独封装一个类而不是直接调用 OpenAI SDK：
    1. 统一的重试/错误处理逻辑
    2. Token 用量追踪集中管理
    3. 方便切换不同的 LLM 提供商（只需改 base_url）
    4. 便于单测时 mock

    Attributes:
        async_client: AsyncOpenAI SDK 客户端实例
        model: 使用的模型名称（如 gpt-4o-mini, deepseek-chat）
        trace: 日志追踪器（记录每次调用的请求/响应）
        last_usage: 最近一次 API 调用的 token 使用统计
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        trace: TraceLogger | None = None,
        timeout: float | None = None,
    ):
        """
        初始化 LLM 客户端。

        参数来源优先级：构造参数 > 环境变量 > 代码默认值。
        这样做的好处是：
        - 测试时可以轻松注入假数据
        - 生产环境通过 .env 配置
        - 默认值保底，不会因配置缺失而崩溃

        Args:
            api_key: OpenAI API Key
            base_url: API Base URL（支持兼容 OpenAI 的第三方服务）
            model: 模型名称
            trace: 日志追踪器实例
            timeout: API 请求超时秒数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.trace = trace or TraceLogger()
        self.timeout = timeout or float(os.getenv("LLM_TIMEOUT", "60"))

        # 初始化 OpenAI SDK 异步客户端
        # max_retries=0：关闭 SDK 内置重试，使用自定义重试逻辑
        # 为什么：SDK 的重试策略不可定制，自实现可以控制退避策略和日志
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

        self.trace.log_system(
            f"LLM Client 初始化完成 | 模型: {self.model} | "
            f"Base URL: {self.base_url} | 超时: {self.timeout}s"
        )

        # 最后一次调用的 token 使用统计，供 Runtime 做限额检查
        self.last_usage: dict | None = None


    # 主要用于记忆压缩
    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 3,
    ) -> dict:
        """
        发送聊天请求到 LLM（异步模式，带自动重试）。

        与 chat() 功能完全相同，但使用 AsyncOpenAI 客户端，
        支持在 asyncio 事件循环中非阻塞执行。

        Args:
            messages: 对话消息列表
            tools: 工具定义列表（可选）
            max_retries: 最大重试次数，默认 3

        Returns:
            LLM 响应消息字典，格式同 chat()
        """
        self.trace.log_llm_request(messages, tools)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message

                result = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": None,
                }

                if message.tool_calls:
                    result["tool_calls"] = []
                    for tc in message.tool_calls:
                        result["tool_calls"].append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                self.last_usage = usage
                self.trace.log_llm_response(result, usage=usage)

                return result

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.trace.log_error(
                        f"LLM 异步 API 调用失败 (第 {attempt + 1}/{max_retries} 次), "
                        f"{wait_time}s 后重试: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.trace.log_error(
                        f"LLM 异步 API 调用最终失败 (已重试 {max_retries} 次): "
                        f"{type(e).__name__}: {e}"
                    )

        raise last_exception

    # 逐文本推送，记忆压缩时不需要，只需要模型结果
    # 异步聊天 + 工具调用检测（只测不用）
    async def chat_stream_detect_tools_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 3,
    ):
        """
        （异步版本）流式聊天 + 工具调用检测。

        与 chat_stream_detect_tools 功能完全相同，但使用 AsyncOpenAI 客户端，
        支持在 asyncio 事件循环中非阻塞执行。

        使用示例：
            async for event in llm.chat_stream_detect_tools_async(messages, tools):
                if event["type"] == "content":
                    print(event["data"], end="")
                elif event["type"] == "tool_calls_done":
                    tool_calls = event["data"]
        """
        self.trace.log_llm_request(messages, tools)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # 工具调用累加器（流式 API 的 tool_calls 是增量到达的）
        tool_calls_acc: dict[int, dict] = {}

        last_exception = None
        for attempt in range(max_retries):
            tool_calls_acc.clear()
            _content_chunks: list[str] = []
            try:
                stream = await self.async_client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # ---- 文本内容：逐 token yield ----
                    if delta.content:
                        _content_chunks.append(delta.content)
                        yield {"type": "content", "data": delta.content}

                    # ---- 工具调用（流式增量）：累加拼接 ----
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

                # ---- 流正常结束 ----
                if tool_calls_acc:
                    tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
                    prompt_text = str(messages)
                    completion_text = json.dumps(tool_calls)
                    self.last_usage = {
                        "prompt_tokens": config.get_estimated_tokens(prompt_text),
                        "completion_tokens": config.get_estimated_tokens(completion_text),
                        "total_tokens": config.get_estimated_tokens(prompt_text) + config.get_estimated_tokens(completion_text),
                    }
                    self.trace.log_llm_response({
                        "content": None,
                        "tool_calls": tool_calls,
                    })
                    yield {"type": "tool_calls_done", "data": tool_calls}
                else:
                    collected = "".join(_content_chunks)
                    self.last_usage = {
                        "prompt_tokens": config.get_estimated_tokens(str(messages)),
                        "completion_tokens": config.get_estimated_tokens(collected),
                        "total_tokens": config.get_estimated_tokens(str(messages)) + config.get_estimated_tokens(collected),
                    }
                    self.trace.log_llm_response({
                        "content": collected,
                        "tool_calls": None,
                    })

                return  # 成功完成

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.trace.log_error(
                        f"LLM 异步流式调用失败 (第 {attempt + 1}/{max_retries} 次), "
                        f"{wait_time}s 后重试: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.trace.log_error(
                        f"LLM 异步流式调用最终失败 (已重试 {max_retries} 次): "
                        f"{type(e).__name__}: {e}"
                    )

        raise last_exception
