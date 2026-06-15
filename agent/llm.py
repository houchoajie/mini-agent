"""
LLM 客户端 - 封装 OpenAI API 调用

使用 OpenAI SDK 与 LLM 交互，支持：
- Function Calling（工具调用）
- 多轮对话消息传递
- 自动重试与错误处理
"""

import os
import json
import time
from openai import OpenAI
from agent.trace import TraceLogger


class LLMClient:
    """
    LLM API 客户端封装

    封装 OpenAI Chat Completions API，支持 function calling。

    Attributes:
        client: OpenAI SDK 客户端实例
        model: 使用的模型名称
        trace: 日志追踪器
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        trace: TraceLogger | None = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: OpenAI API Key，默认从环境变量 OPENAI_API_KEY 读取
            base_url: API Base URL（支持兼容 OpenAI 的第三方服务），默认从 OPENAI_BASE_URL 读取
            model: 模型名称，默认从 OPENAI_MODEL 读取，回退到 gpt-4o-mini
            trace: 日志追踪器实例
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.trace = trace or TraceLogger()

        # 初始化 OpenAI SDK 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.trace.log_system(f"LLM Client 初始化完成 | 模型: {self.model} | Base URL: {self.base_url}")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 3,
    ) -> dict:
        """
        发送聊天请求到 LLM（带自动重试）

        核心方法：将对话历史和工具定义发送给 LLM，获取响应。
        失败时采用指数退避策略自动重试。

        重试策略：
        - 第 1 次失败后等待 1 秒
        - 第 2 次失败后等待 2 秒
        - 第 3 次失败后等待 4 秒
        - 全部失败后抛出异常

        Args:
            messages: 对话消息列表，格式遵循 OpenAI Chat API：
                [
                    {"role": "system", "content": "系统提示"},
                    {"role": "user", "content": "用户消息"},
                    {"role": "assistant", "content": "助手回复"},
                    {"role": "tool", "tool_call_id": "xxx", "content": "工具结果"},
                ]
            tools: 工具定义列表（OpenAI function calling schema），可选
            max_retries: 最大重试次数，默认 3

        Returns:
            LLM 响应消息字典，格式为：
            {
                "role": "assistant",
                "content": "文本回复（可能为 None）",
                "tool_calls": [...]  # 工具调用请求（可能为 None）
            }

        Raises:
            Exception: 所有重试均失败后抛出最后一个异常
        """
        self.trace.log_llm_request(messages, tools)

        # 构建 API 请求参数（只需构建一次）
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"  # 让 LLM 自行决定是否调用工具

        # ============================================================
        # 重试循环：采用指数退避策略
        # ============================================================
        last_exception = None
        for attempt in range(max_retries):
            try:
                # 调用 OpenAI API
                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message = choice.message

                # 将响应转换为字典格式
                result = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": None,
                }

                # 处理工具调用请求
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

                # 记录响应日志
                self.trace.log_llm_response(result, usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                })

                return result

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # 指数退避：1s, 2s, 4s
                    wait_time = 2 ** attempt
                    self.trace.log_error(
                        f"LLM API 调用失败 (第 {attempt + 1}/{max_retries} 次), "
                        f"{wait_time}s 后重试: {type(e).__name__}: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    # 所有重试均失败
                    self.trace.log_error(
                        f"LLM API 调用最终失败 (已重试 {max_retries} 次): "
                        f"{type(e).__name__}: {e}"
                    )

        # 理论上不会走到这里，但作为安全兆底
        raise last_exception

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        """
        流式聊天 - 逐 token 返回内容

        用于最终文本回复的流式输出，提升用户体验。
        注意：流式模式不支持 tool_calls，仅用于纯文本回复场景。

        工作原理：
        1. 设置 stream=True 参数
        2. 逐 chunk 接收 LLM 响应
        3. 每个 chunk 包含一小段文本（通常 1 个 token）
        4. 通过 yield 逐个返回给调用方

        Args:
            messages: 对话消息列表
            tools: 工具定义列表（OpenAI function calling schema），可选

        Yields:
            str: 每个 chunk 的文本内容片段

        使用示例：
            for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
