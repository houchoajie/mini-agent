"""
Mini Agent - 主入口

交互式 CLI 界面，支持：
- 多轮对话
- 会话管理（新建/切换/列出）
- 执行追踪日志查看

运行方式：
    python main.py                  # 新建会话
    python main.py --session ID     # 恢复指定会话
    python main.py --list           # 列出所有会话

环境变量配置：
    在项目根目录创建 .env 文件：
    OPENAI_API_KEY=sk-your-key
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_MODEL=gpt-4o-mini
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).parent / ".env")

from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.trace import TraceLogger


def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("  🤖 Mini Agent - 最小可用 Agent 系统")
    print("=" * 60)
    print("  命令：")
    print("    /new          - 新建会话")
    print("    /switch ID    - 切换到指定会话")
    print("    /sessions     - 列出所有会话")
    print("    /trace        - 查看执行追踪摘要")
    print("    /clear        - 清空当前对话历史")
    print("    /quit         - 退出")
    print("=" * 60)


def main():
    """
    主函数 - 交互式 CLI 循环

    流程：
    1. 解析命令行参数
    2. 初始化 Agent Runtime
    3. 进入交互循环：读取用户输入 → Agent 处理 → 显示结果
    """
    # ============================================================
    # 解析命令行参数
    # ============================================================
    session_id = None
    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        if idx + 1 < len(sys.argv):
            session_id = sys.argv[idx + 1]

    if "--list" in sys.argv:
        # 列出所有会话
        manager = SessionManager()
        sessions = manager.list_sessions()
        if not sessions:
            print("暂无保存的会话。")
        else:
            print(f"共 {len(sessions)} 个会话：")
            for s in sessions:
                print(f"  {s['session_id']} | 更新: {s['updated_at']} | 消息数: {s['message_count']}")
        return

    # ============================================================
    # 检查 API Key
    # ============================================================
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-your-api-key-here":
        print("❌ 错误：未配置 OPENAI_API_KEY")
        print("请在项目根目录创建 .env 文件并配置：")
        print("  OPENAI_API_KEY=sk-your-actual-key")
        print("  OPENAI_BASE_URL=https://api.openai.com/v1")
        print("  OPENAI_MODEL=gpt-4o-mini")
        sys.exit(1)

    # ============================================================
    # 初始化 Agent Runtime
    # ============================================================
    print_banner()

    trace = TraceLogger(session_id=session_id or "default")
    runtime = AgentRuntime(
        max_steps=10,
        session_id=session_id,
        trace=trace,
    )
    print(f"📌 当前会话: {runtime.session_id}")
    print(f"📌 最大步数: {runtime.max_steps}")
    print()

    # ============================================================
    # 交互式对话循环
    # ============================================================
    session_manager = SessionManager()

    while True:
        try:
            # 读取用户输入
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # ============================================================
            # 处理内置命令
            # ============================================================
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == "/quit" or cmd == "/exit":
                    print("👋 再见！")
                    break

                elif cmd == "/new":
                    # 新建会话
                    runtime = AgentRuntime(max_steps=10, trace=TraceLogger())
                    print(f"✅ 新会话已创建: {runtime.session_id}")
                    continue

                elif cmd == "/switch":
                    if not arg:
                        print("用法: /switch <session_id>")
                        continue
                    result = runtime.switch_session(arg)
                    print(f"{'✅' if '已切换' in result else '❌'} {result}")
                    continue

                elif cmd == "/sessions":
                    sessions = session_manager.list_sessions()
                    if not sessions:
                        print("暂无保存的会话。")
                    else:
                        print(f"共 {len(sessions)} 个会话：")
                        for s in sessions:
                            marker = " ← 当前" if s["session_id"] == runtime.session_id else ""
                            print(f"  {s['session_id']} | 更新: {s['updated_at']} | 消息数: {s['message_count']}{marker}")
                    continue

                elif cmd == "/trace":
                    print(f"\n{runtime.get_trace_summary()}\n")
                    continue

                elif cmd == "/clear":
                    runtime.session.clear_history()
                    print("✅ 对话历史已清空")
                    continue

                else:
                    print(f"未知命令: {cmd}")
                    continue

            # ============================================================
            # 正常对话 - 流式调用 Agent Runtime
            # ============================================================
            print()
            print("Agent: ", end="", flush=True)
            for event in runtime.run_stream(user_input):
                if event["type"] == "text_chunk":
                    print(event["data"], end="", flush=True)
            print()
            print()

        except KeyboardInterrupt:
            print("\n\n👋 收到中断信号，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")
            continue


if __name__ == "__main__":
    main()
