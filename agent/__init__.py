"""
============================================================
Mini Agent — 从零实现的最小可用 Agent 框架
============================================================

本项目不依赖 LangChain / OpenHands 等现成 Agent 框架，
核心 runtime 完全自主实现。

版本: 0.1.0

核心模块：
    runtime      — ReAct 循环引擎（核心）
    llm          — LLM API 客户端（OpenAI 兼容）
    session      — 会话管理与持久化
    memory       — 对话记忆自动压缩
    trace        — 执行追踪日志
    config       — 配置管理
    user_manager — 用户注册/登录/数据隔离
    tool_executor— 工具执行器（超时/重试/异步）
    tools        — 内置工具注册表
"""

__version__ = "0.1.0"
