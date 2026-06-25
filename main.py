"""
============================================================
Mini Agent — CLI 主入口
============================================================

交互式命令行界面，完整的用户交互流程：

1. 登录/注册（密码加盐哈希存储，多用户数据隔离）
2. 命令行参数解析（--session 恢复会话 / --list 列出会话）
3. API Key 检查
4. Agent Runtime 初始化（含用户上下文、会话创建）
5. 交互式对话循环（支持内置命令和自然语言对话）

内置命令：
    /new          — 新建会话（自动保存当前会话）
    /switch ID   — 切换到指定会话
    /sessions    — 列出当前用户的所有会话
    /trace       — 查看执行追踪摘要（含 token 统计）
    /clear       — 清空当前对话历史（保留 system prompt）
    /quit        — 退出程序（自动保存当前会话）

运行方式：
    python main.py                  # 新建会话（需先登录）
    python main.py --session ID     # 恢复指定会话
    python main.py --list           # 列出当前用户的所有会话

数据隔离：
    每个用户的数据存储在自己的目录下：
    .agent_data/<username>/{session, log, task}/
    不同用户之间完全隔离。

环境变量（在项目根目录 .env 文件中配置）：
    OPENAI_API_KEY      API 密钥（必填）
    OPENAI_BASE_URL     API 端点（支持第三方兼容服务）
    OPENAI_MODEL        模型名称
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（必须先于其他 import 执行）
load_dotenv(Path(__file__).parent / ".env")

from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.user_manager import UserManager


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


def login_loop(user_manager: UserManager) -> str:
    """
    登录/注册交互循环，返回登录成功的用户名。

    首次使用时提示注册，已有用户时可直接登录。
    密码使用加盐 SHA-256 哈希存储（见 user_manager.py），
    不会以明文形式保存在磁盘上。

    交互流程：
        1. 选择登录/注册/退出
        2. 输入用户名和密码
        3. 验证通过后进入主程序

    Returns:
        登录成功的用户名
    """
    while True:
        print("\n=== Mini Agent 登录 ===")
        if not user_manager.has_users():
            print("首次使用请先注册。\n")

        print("1. 登录")
        print("2. 注册")
        print("3. 退出")
        choice = input("请选择: ").strip()

        if choice == "1":
            username = input("用户名: ").strip()
            password = input("密码: ").strip()
            ok, msg = user_manager.login(username, password)
            if ok:
                print(f"✅ 登录成功！欢迎 {username}")
                return username
            print(f"❌ {msg}")

        elif choice == "2":
            username = input("用户名: ").strip()
            password = input("密码: ").strip()
            ok, msg = user_manager.register(username, password)
            print(f"{'✅' if ok else '❌'} {msg}")

        elif choice == "3":
            print("👋 再见！")
            sys.exit(0)

        else:
            print("❌ 无效选择，请输入 1、2 或 3")


def main():
    """
    主函数 — 交互式 CLI 循环。

    完整执行流程：
    1. 用户登录/注册（UserManager 验证凭证）
    2. 解析命令行参数（--session, --list）
    3. API Key 检查（未配置则退出）
    4. 打印启动横幅
    5. 初始化 AgentRuntime（自动创建/恢复会话，TraceLogger 使用正确 session_id）
    6. 进入交互式 while 循环：
       a. 读取用户输入
       b. 检查是否为内置命令（/开头）
          └── 命令 → 执行对应操作（/new, /switch, /quit 等）
          └── 对话 → 调用 runtime.run_stream() 流式输出
       c. 异常捕获 → 友好错误提示，继续运行（不崩溃）

    运行时数据文件：
        .agent_data/<username>/session/  — 会话 JSON 文件
        .agent_data/<username>/log/      — Trace 日志文件
        .agent_data/<username>/task/     — 任务数据文件

    注意：
        - TraceLogger 由 Runtime 内部创建（不再由 main.py 提前创建）
        - 保证日志文件的 session_id 与实际会话 ID 一致
        - 每次用户输入和工具结果都会立即持久化（_save()）
    """
    # === 用户管理 ===
    user_manager = UserManager()
    username = login_loop(user_manager)

    # ============================================================
    # 解析命令行参数
    # ============================================================
    session_id = None
    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        if idx + 1 < len(sys.argv):
            session_id = sys.argv[idx + 1]

    if "--list" in sys.argv:
        # 列出当前用户的所有会话
        user_dir = user_manager.get_user_dir(username)
        manager = SessionManager(user_dir=user_dir)
        sessions = manager.list_sessions()
        if not sessions:
            print(f"用户 '{username}' 暂无保存的会话。")
        else:
            print(f"用户 '{username}' 共有 {len(sessions)} 个会话：")
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
    # 初始化 Agent Runtime（带用户上下文）
    # ============================================================
    print_banner()

    runtime = AgentRuntime(
        max_steps=10,
        session_id=session_id,
        username=username,
    )
    print(f"📌 当前用户: {username}")
    print(f"📌 当前会话: {runtime.session_id}")
    print(f"📌 最大步数: {runtime.max_steps}")
    print()

    # ============================================================
    # 交互式对话循环
    # ============================================================
    # 使用当前用户的 session_manager
    session_manager = SessionManager(user_dir=user_manager.get_user_dir(username))

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
                    runtime.save_session()
                    print("👋 再见！")
                    break

                elif cmd == "/new":
                    # 保存当前会话再新建（TraceLogger 由 Runtime 自动创建）
                    runtime.save_session()
                    runtime = AgentRuntime(max_steps=10, username=username)
                    print(f"✅ 新会话已创建: {runtime.session_id}")
                    continue

                elif cmd == "/switch":
                    if not arg:
                        print("用法: /switch <session_id>")
                        continue
                    runtime.save_session()
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
                    runtime.save_session()
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
            ask_question = None  # 多轮交互：工具向用户提问
            for event in runtime.run_stream(user_input):
                if event["type"] == "text_chunk":
                    print(event["data"], end="", flush=True)
                elif event["type"] == "ask_user":
                    ask_question = event["data"]["question"]
            print()
            # 多轮交互：工具向用户提问，等待用户输入
            if ask_question:
                print("\n" + "=" * 60)
                print(f"🤖 {ask_question}")
                print("=" * 60)
                # 流已结束，下轮用户输入会通过正常循环处理
            print()

        except KeyboardInterrupt:
            runtime.save_session()
            print("\n\n👋 收到中断信号，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")
            continue


if __name__ == "__main__":
    main()
