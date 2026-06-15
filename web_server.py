"""
Mini Agent Web Server - FastAPI 后端服务

提供完整的 Web API 接口：
- POST /api/chat - 对话接口（SSE 流式输出）
- POST /api/chat/non-stream - 非流式对话
- GET  /api/sessions - 列出所有会话
- POST /api/sessions/new - 创建新会话
- POST /api/sessions/switch - 切换会话
- GET  /api/trace - 查看执行追踪
- GET  /api/tools - 获取工具列表
"""

import os
import sys
from pathlib import Path
from typing import Optional

import json
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent / ".env")

# 确保项目路径在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.trace import TraceLogger
from agent.tools import get_all_tool_schemas, get_tool_by_name

# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title="Mini Agent API",
    description="从零实现的最小可用 Agent Web API",
    version="0.2.0",
)

# ============================================================
# 全局运行时管理
# 维护多个会话的 AgentRuntime 实例
# ============================================================
runtimes: dict[str, AgentRuntime] = {}
session_manager = SessionManager()


def get_or_create_runtime(session_id: Optional[str] = None) -> AgentRuntime:
    """
    获取或创建 AgentRuntime 实例

    Args:
        session_id: 会话 ID，为 None 时创建新会话

    Returns:
        AgentRuntime 实例
    """
    if session_id and session_id in runtimes:
        return runtimes[session_id]

    # 创建新的 Runtime
    trace = TraceLogger(session_id=session_id or "web")
    runtime = AgentRuntime(
        max_steps=10,
        session_id=session_id,
        trace=trace,
    )
    runtimes[runtime.session_id] = runtime
    return runtime


# ============================================================
# 挂载静态文件目录（为 index.html 提供 CSS/JS 支持）
# ============================================================
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================
# Pydantic 请求/响应模型
# ============================================================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int


class ToolInfo(BaseModel):
    name: str
    description: str


# ============================================================
# 路由：静态页面
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端 HTML 页面"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>前端页面未找到</h1><p>请确保 static/index.html 存在</p>")


# ============================================================
# 路由：对话接口（流式输出 - SSE）
# ============================================================
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口 - Server-Sent Events (SSE)

    实时逐 token 输出 Agent 回复，提供更好的用户体验。

    SSE 事件格式：
    - event: tool_call - 工具调用信息
    - event: tool_result - 工具执行结果
    - event: text - LLM 文本输出（逐 token）
    - event: done - 对话完成
    - event: error - 错误信息
    """
    try:
        # 获取或创建 Runtime
        runtime = get_or_create_runtime(req.session_id)

        # 定义生成器函数（用于 SSE）
        async def event_generator():
            """
            SSE 事件流生成器

            工作流程：
            1. 记录用户输入
            2. 执行 ReAct 循环
            3. 每次工具调用/结果时发送事件
            4. LLM 回复时逐 token 发送
            5. 完成后发送 done 事件
            """
            try:
                # 记录用户输入
                yield f"event: user_input\ndata: {json.dumps(req.message)}\n\n"

                # 添加到对话历史
                runtime.session.add_message("user", req.message)

                # 检查 API Key 是否配置
                api_key = os.getenv("OPENAI_API_KEY", "")
                if not api_key or api_key == "sk-your-api-key-here":
                    yield f"event: error\ndata: 未配置 OPENAI_API_KEY，请在 .env 文件中填写正确的 API Key\n\n"
                    yield f"event: done\ndata: {{\"session_id\": \"{runtime.session_id}\"}}\n\n"
                    return

                # 对话记忆压缩检查
                if runtime.memory.should_compress(runtime.session.messages):
                    runtime.trace.log_system("对话历史过长，正在压缩...")
                    compressed = runtime.memory.compress(runtime.session.messages, runtime.llm)
                    runtime.session.messages = compressed
                    yield f"event: system\ndata: 对话历史已压缩\n\n"

                # ReAct 主循环
                step = 0
                while step < runtime.max_steps:
                    step += 1
                    yield f"event: step\ndata: Step {step}/{runtime.max_steps}\n\n"

                    # 调用 LLM
                    response = runtime.llm.chat(
                        messages=runtime.session.get_messages(),
                        tools=runtime.tool_schemas,
                    )

                    tool_calls = response.get("tool_calls")

                    if tool_calls:
                        # 有工具调用
                        runtime.session.add_message(
                            "assistant",
                            response.get("content") or "",
                            tool_calls=response["tool_calls"],
                        )

                        for tc in tool_calls:
                            func_name = tc["function"]["name"]
                            func_args = tc["function"]["arguments"]

                            # 发送工具调用事件（使用 json.dumps 确保序列化正确）
                            try:
                                args_parsed = json.loads(func_args) if isinstance(func_args, str) else func_args
                            except json.JSONDecodeError:
                                args_parsed = {"raw": func_args}
                            tool_call_data = json.dumps({"tool": func_name, "args": args_parsed})
                            yield f"event: tool_call\ndata: {tool_call_data}\n\n"

                            # 执行工具
                            tool = get_tool_by_name(func_name)
                            if tool:
                                result = tool.safe_execute(func_args)
                                result_text = result.to_string()
                                # 发送工具结果事件
                                tool_result_data = json.dumps({
                                    "tool": func_name,
                                    "success": result.success,
                                    "result": result_text[:500],  # 截断过长结果
                                })
                                yield f"event: tool_result\ndata: {tool_result_data}\n\n"
                            else:
                                result_text = f"[ERROR] 未知工具: {func_name}"

                            # 添加结果到历史
                            runtime.session.add_message(
                                "tool",
                                result_text,
                                tool_call_id=tc["id"],
                            )

                        # 继续循环
                        continue
                    else:
                        # 没有工具调用 → 使用流式输出逐 token 返回
                        full_content = ""
                        # 使用 while 循环而非 for 循环，以便在 yield 后让出事件循环
                        # 让 uvicorn 的 I/O 缓冲区有机会冲刷到 TCP socket，实现真正的流式输出
                        token_iter = iter(runtime.llm.chat_stream(
                            messages=runtime.session.get_messages(),
                            tools=None,  # 确定不需要工具了
                        ))
                        while True:
                            try:
                                token = next(token_iter)
                            except StopIteration:
                                break
                            full_content += token
                            yield f"event: text\ndata: {json.dumps(token)}\n\n"
                            # 让出事件循环：允许 uvicorn 将 SSE 数据从写缓冲区冲刷到 TCP socket
                            # 否则同步 next() 会阻塞事件循环，数据被缓冲区累积后一次性发送
                            await asyncio.sleep(0)

                        # 添加到对话历史并保存
                        runtime.session.add_message("assistant", full_content)
                        runtime.session_manager.save_session(runtime.session)
                        break
                else:
                    # 超过最大步数
                    yield f"event: error\ndata: {json.dumps('达到最大步数限制')}\n\n"
                    yield f"event: text\ndata: {json.dumps('抱歉，执行步骤过多，无法继续处理。')}\n\n"

                # 发送完成事件
                done_data = json.dumps({"session_id": runtime.session_id})
                yield f"event: done\ndata: {done_data}\n\n"

            except Exception as e:
                error_msg = str(e).replace("\n", " ").replace("\r", "")
                yield f"event: error\ndata: {error_msg}\n\n"
                yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 路由：对话接口（非流式）
# ============================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    非流式对话接口

    等待完整回复后返回，适合不支持 SSE 的客户端。
    """
    try:
        runtime = get_or_create_runtime(req.session_id)
        response = runtime.run(req.message)
        return ChatResponse(
            response=response,
            session_id=runtime.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 路由：会话管理
# ============================================================
@app.get("/api/sessions")
async def list_sessions():
    """列出所有已保存的会话"""
    sessions = session_manager.list_sessions()
    return [
        SessionInfo(
            session_id=s["session_id"],
            created_at="",  # SessionManager 未返回，简化处理
            updated_at=s["updated_at"],
            message_count=s["message_count"],
        )
        for s in sessions
    ]


@app.post("/api/sessions/new")
async def create_session():
    """创建新会话"""
    runtime = get_or_create_runtime()
    return {"session_id": runtime.session_id, "message": "新会话已创建"}


@app.post("/api/sessions/switch")
async def switch_session(session_id: str):
    """切换到指定会话"""
    loaded = session_manager.load_session(session_id)
    if not loaded:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    runtime = get_or_create_runtime(session_id)
    runtime.session = loaded
    return {"session_id": session_id, "message": "已切换到会话"}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    success = session_manager.delete_session(session_id)
    if success:
        runtimes.pop(session_id, None)
        return {"message": "会话已删除"}
    raise HTTPException(status_code=404, detail="会话不存在")


# ============================================================
# 路由：工具列表
# ============================================================
@app.get("/api/tools")
async def get_tools():
    """获取所有可用工具的信息"""
    schemas = get_all_tool_schemas()
    return [
        ToolInfo(
            name=s["function"]["name"],
            description=s["function"]["description"],
        )
        for s in schemas
    ]


# ============================================================
# 路由：执行追踪
# ============================================================
@app.get("/api/trace")
async def get_trace(session_id: Optional[str] = None, limit: int = 50):
    """
    获取执行追踪日志

    Args:
        session_id: 会话 ID（可选）
        limit: 返回最新的 N 条日志
    """
    # 查找对应的 Runtime
    if session_id and session_id in runtimes:
        runtime = runtimes[session_id]
        entries = runtime.trace.get_all_entries()
    else:
        # 返回最近一个 Runtime 的日志
        if runtimes:
            runtime = list(runtimes.values())[-1]
            entries = runtime.trace.get_all_entries()
        else:
            entries = []

    # 限制返回数量
    return entries[-limit:]


# ============================================================
# 路由：系统信息
# ============================================================
@app.get("/api/info")
async def get_info():
    """获取系统信息"""
    return {
        "version": "0.2.0",
        "tools_count": len(get_all_tool_schemas()),
        "active_sessions": len(runtimes),
        "model": os.getenv("OPENAI_MODEL", "unknown"),
    }


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)

