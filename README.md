# Mini Agent - 从零实现的最小可用 Agent

## 项目简介

从零实现的最小可用 AI Agent 系统，核心 runtime 完全自主实现，不依赖 LangChain / OpenHands 等现成 Agent 框架。

**核心能力**：
- ✅ 多轮对话 + Session 持久化（原子写入，防文件损坏）
- ✅ ReAct 循环（推理 → 行动 → 观察 → 继续），支持流式输出
- ✅ 6 个内置工具（calculator / search / todo_manager / file_reader / file_writer / datetime_tool）
- ✅ 跨轮次状态持久化（任务管理 + 文件读写）
- ✅ 对话记忆自动压缩（扁平摘要 + 滑动窗口，业界主流方案）
- ✅ LLM 调用自动重试（指数退避）
- ✅ 多用户隔离（注册/登录，用户数据完全隔离）
- ✅ 工具生命周期钩子（before_execute / after_execute / on_error）
- ✅ 工具结果缓存（纯函数工具自动缓存）
- ✅ 危险操作多轮确认（ask_user 机制）
- ✅ Per-Tool Token 配额（精细控制每个工具的 token 消耗）
- ✅ 会话 Token 限额（防止 token 费用失控）
- ✅ 集中配置管理（`.env` + 环境变量统一配置）
- ✅ 完整执行 Trace 日志（控制台 + JSONL 文件）
- ✅ 完整单元测试（`tests/` 目录）
- ✅ 异步 Runtime（asyncio 原生非阻塞流式 + 工具并行执行）

---

## 快速开始

### 1. 环境要求

- Python >= 3.13
- OpenAI API Key（或兼容的 API 服务，如 DeepSeek、智谱等）

### 2. 安装依赖

```bash
pip install openai python-dotenv fastapi uvicorn PyMuPDF python-docx fpdf2
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-actual-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

> **提示**：支持任何兼容 OpenAI API 的服务（如 DeepSeek、智谱等），只需修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

### 4. 运行

```bash
# 首次运行会提示注册，之后可登录
python main.py

# 恢复指定会话
python main.py --session session_abc123

# 列出当前用户的所有会话
python main.py --list
```

### 5. 交互命令

| 命令 | 说明 |
|------|------|
| `/new` | 新建会话 |
| `/switch <id>` | 切换到指定会话 |
| `/sessions` | 列出所有会话 |
| `/trace` | 查看执行追踪摘要（含 token 统计） |
| `/clear` | 清空当前对话历史 |
| `/quit` | 退出 |

---

## 系统设计

### 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                     main.py (CLI 入口)                    │
│       用户输入 → 登录 → 命令解析 → Agent Runtime → 结果展示  │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│                   Agent Runtime (核心引擎)                 │
│                                                          │
│  ┌───────────────────┐  ┌──────────────────────────┐     │
│  │   LLM Client      │  │     Tool Executor        │     │
│  │  (OpenAI API 封装) │  │  (超时+重试+异步+上下文)  │     │
│  └────────┬──────────┘  └───────────┬──────────────┘     │
│           │                         │                    │
│  ┌────────▼─────────────────────────▼──────────────┐     │
│  │              ReAct 主循环                        │     │
│  │  输入 → LLM推理 → 工具调用? → 执行 → 反馈 → 继续   │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────────────┐     │
│  │  Conversation    │  │     Trace Logger         │     │
│  │  Memory          │  │  (控制台+JSONL文件持久化)  │     │
│  │  (自动摘要压缩)   │  │                          │     │
│  └──────────────────┘  └──────────────────────────┘     │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────────────┐     │
│  │  Session Manager │  │     User Manager         │     │
│  │  (会话持久化)     │  │  (注册/登录/数据隔离)     │     │
│  └──────────────────┘  └──────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### 目录结构

```
├── main.py                  # 入口文件 - 交互式 CLI（异步）
├── pyproject.toml           # 项目配置
├── .env.example             # 环境变量模板
├── .env                     # 环境变量配置（用户创建，已 gitignore）
│
├── agent/
│   ├── __init__.py
│   ├── runtime.py           # 核心 - Agent ReAct 循环引擎（异步）
│   ├── llm.py               # LLM 客户端 - OpenAI API 封装 + 重试 + 流式
│   ├── session.py           # 会话管理 - 多轮对话 + 状态持久化
│   ├── memory.py            # 对话记忆 - 扁平摘要 + 滑动窗口压缩
│   ├── trace.py             # 执行追踪日志系统（控制台 + JSONL）
│   ├── config.py            # 集中配置管理（.env + 环境变量）
│   ├── user_manager.py      # 用户管理 - 注册/登录/密码哈希/数据隔离
│   ├── tool_executor.py     # 工具执行器 - 超时/重试/异步分派
│   │
│   └── tools/
│       ├── __init__.py      # 工具注册表（ALL_TOOLS）
│       ├── base.py          # 工具基类 + ToolResult + ToolContext + ErrorCode
│       ├── calculator.py    # 数学计算器（AST 安全求值 + 自动缓存）
│       ├── search.py        # Mock 搜索引擎
│       ├── todo_manager.py  # 任务管理器（文件锁 + 原子写入 + 用户隔离）
│       ├── file_reader.py   # 文件读取工具（白名单 + 安全限制 + 范围读取）
│       ├── file_writer.py   # 文件写入工具（原子写入 + 覆盖确认 + PDF/Word）
│       └── datetime_tool.py # 日期时间工具（1 秒缓存）
│
├── tests/
│   ├── __init__.py
│   └── test_tool_base.py    # 工具基类核心功能测试
│
├── .agent_data/             # 运行时数据（自动生成，已 gitignore）
│   ├── userInfo.json        # 全局用户凭证（密码加盐哈希存储）
│   ├── <用户名>/             # 用户数据目录
│   │   ├── session/         # 会话 JSON 文件
│   │   ├── log/             # Trace 日志文件
│   │   └── task/            # 任务数据（todos.json + 锁文件）
│
├── ARCHITECTURE.md          # 详细架构设计文档
└── tools.md                 # 工具执行架构分析（同步/异步分派模式详解）
```

### ReAct 循环流程

```
用户输入
   │
   ▼
┌──────────────────────────────┐
│ 对话记忆压缩检查              │
│ (消息>50条时自动压缩)         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Token 限额检查               │
│  (超限时清空历史保留 system)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  流式调用 LLM                 │◄────────────────────┐
│  (异步非阻塞，边输出边检测      │                     │
│   tool_calls)                │                     │
└──────────┬───────────────────┘                     │
           │                                          │
           ▼                                          │
   LLM 返回 tool_calls？                              │
    ├── 是 ──► 工具执行器                              │
    │          ├─ 超时控制 + 自动重试                    │
    │          ├─ 生命周期钩子 + 缓存 + 配额检查         │
    │          ├─ 需要用户确认? → ask_user 暂停         │
    │          └─ 结果加入历史 ────────────────────────┘
    │          (step++, 检查 max_steps + 连续失败)
    │
    └── 否 ──► 输出文本回复 ──► 保存会话 ──► 结束本轮
                          │
                   连续失败≥3次 或 step≥max_steps
                          │
                          ▼
                   强制总结终止
```

### 核心组件说明

#### 1. Agent Runtime (`agent/runtime.py`)

核心 ReAct 循环引擎，**全程异步**（`asyncio`）。每次 `run_stream(user_input)` 调用返回一个异步生成器，逐事件推送：

1. 将用户输入追加到对话历史
2. **检查对话记忆是否需要压缩**（超过 50 条消息时自动压缩）
3. **检查 Token 限额**（超出时强制清空历史）
4. 调用 LLM（流式，传递历史 + 工具 schema）
5. 若 LLM 请求工具调用 → 执行工具 → 结果回传 → 重复（最多 `max_steps` 次）
6. 若 LLM 直接回复 → 返回结果
7. 超过 `max_steps`（默认 10）→ 强制终止
8. 工具连续失败 ≥ 3 次 → 放弃工具，强制 LLM 直接总结

**事件类型**（`run_stream()` yield 的事件字典）：

| 事件类型 | 说明 |
|---------|------|
| `text_chunk` | LLM 流式输出的文本片段 |
| `tool_call` | LLM 请求工具调用 |
| `tool_result` | 工具执行结果 |
| `ask_user` | 需要用户确认（多轮交互） |
| `step` | ReAct 步数信息 |
| `done` | 本轮结束 |
| `error` | 执行错误 |

#### 2. LLM Client (`agent/llm.py`)

封装 OpenAI Chat Completions API，支持：

- **Function Calling**：工具调用能力
- **自动重试**：失败时指数退避重试（1s → 2s → 4s），默认 3 次
- **流式输出**：`chat_stream_detect_tools_async()` 使用 `AsyncOpenAI`，逐 token 响应 + 并行检测 tool_calls
- **已移除 max_tokens 限制**：避免兼容 API 静默截断回复

配置项（通过集中配置模块 `config.py` 读取 `.env`）：

| 配置项 | 环境变量 | 默认值 |
|-------|---------|--------|
| API 密钥 | `OPENAI_API_KEY` | `""`（必填） |
| API 端点 | `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| 模型名称 | `OPENAI_MODEL` | `gpt-4o-mini` |
| 请求超时 | `LLM_TIMEOUT` | `60` 秒 |
| LLM max_tokens | `DEFAULT_LLM_MAX_TOKENS` | `4096`（0 转为 None） |

#### 3. Conversation Memory (`agent/memory.py`)

对话记忆管理器，采用 **扁平摘要 + 滑动窗口** 混合方案（业界主流，ChatGPT / Claude / Gemini 均采用）：

- **触发条件**：当非 system 消息数超过 `max_messages`（默认 50）
- **压缩策略**：保留最近 `keep_recent` 条消息（默认 20），早期消息压缩为**一段**扁平摘要
- **摘要生成**：优先使用 LLM 生成高质量摘要，失败时降级为文本截断
- **增量更新**：多次压缩时，将旧摘要和新掉出窗口的消息合并成一段新摘要替换旧摘要（始终只有 1 段）

**为什么是扁平而非多轮追加**：
- 多轮追加：压缩 5 次后摘要占 5×500=2500 字，内容大量重复
- 扁平摘要：始终固定 800 字，每次用新摘要替换旧摘要，信息密度最高

#### 4. Session Manager (`agent/session.py`)

- 每个会话包含：session_id、消息历史、时间戳、元数据
- 以 JSON 文件持久化到 `.agent_data/<username>/session/`
- 原子写入（先写临时文件再 rename），防止崩溃导致文件损坏
- 支持创建、加载、保存、列出、删除会话

#### 5. Trace Logger (`agent/trace.py`)

- 实时控制台输出（带颜色 emoji 标记，输出到 **stderr** 避免污染 LLM 回复）
- JSONL 文件持久化到 `.agent_data/<username>/log/`
- 记录级别：system / user / llm_request / llm_response / tool_call / tool_result / error / step / token_usage
- 包含 token 使用统计

#### 6. User Manager (`agent/user_manager.py`)

- **注册/登录**：密码使用 `salt$sha256_hex` 格式存储，不保存明文
- **数据隔离**：每个用户的数据存储在自己的目录下（`.agent_data/<username>/{session, log, task}/`）
- **用户信息**：持久化到 `.agent_data/userInfo.json`

#### 7. 配置管理 (`agent/config.py`)

集中管理从 `.env` 和系统环境变量加载的所有应用配置：

| 配置函数 | 环境变量 | 默认值 | 说明 |
|---------|---------|--------|------|
| `get_api_key()` | `OPENAI_API_KEY` | `""` | API 密钥（必填） |
| `get_base_url()` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 端点 |
| `get_model()` | `OPENAI_MODEL` | `gpt-4o-mini` | 模型名称 |
| `get_total_token_limit()` | `TOTAL_TOKEN_LIMIT` | `0` | 会话累计 token 上限，0=不限制 |
| `get_default_llm_max_tokens()` | `DEFAULT_LLM_MAX_TOKENS` | `4096` | 每次 LLM 调用的 max_tokens |
| `is_token_limit_enabled()` | `ENABLE_TOKEN_LIMIT` | `true` | 是否启用 token 限额检查 |
| `get_max_tool_retries()` | `MAX_TOOL_RETRIES` | `2` | 工具失败时的最大重试次数 |
| `get_tool_retry_delay()` | `TOOL_RETRY_DELAY` | `1.0` | 工具重试间隔秒数 |

#### 8. Tool Executor (`agent/tool_executor.py`)

工具执行的**调度器层**，负责决定工具走哪条执行路径：

```
tool_executor.execute_async(tool, arguments, context)
  │
  ├─ 工具重写了 execute_async()? ─── True
  │     └─ asyncio.wait_for(safe_execute_async(), timeout)  ← 原生异步
  │
  └─ False
        └─ run_in_executor(线程池, safe_execute, ...)        ← 线程池执行同步
```

| 特性 | 实现 |
|------|------|
| 超时控制 | `asyncio.wait_for` + 线程池 |
| 自动重试 | 按 `MAX_TOOL_RETRIES` 配置重试，仅失败时重试 |
| 异步检测 | `_is_async_override()` 用 `is` 比较函数对象 |
| 上下文注入 | 每次执行前 `tool.set_context(context)` |

---

## 工具说明

### 工具系统架构

工具系统分**三层**：

| 层 | 职责 | 实现 |
|----|------|------|
| 核心逻辑层 | 纯业务逻辑 | 工具子类实现 `execute()` |
| 安全流水线层 | 参数校验 + 钩子 + 缓存 + 配额 | `safe_execute()` / `safe_execute_async()` |
| 调度器层 | 超时 + 重试 + 异步分派 | `tool_executor.execute_async()` |

### 安全执行流水线（`safe_execute`）

```
参数JSON
   │
   ▼
json.loads() 解析
   │
   ▼
参数校验 (类型/必须/枚举)
   │
   ▼
before_execute 钩子 (工具级 → 全局注册表)
   │
   ▼
配额检查 (per-tool quota)
   │  ├─ quota_limit>0 且已超限 → 返回配额耗尽错误
   │  └─ 未超限或不限额 → 继续执行
   │
   ▼
缓存检查 (纯函数工具，命中直接返回)
   │
   ▼
★ execute() 或 execute_async()
   │
   ▼
缓存写入
   │
   ▼
after_execute 钩子 (工具级 → 全局注册表)
   │
   ▼
结果不截断 (max_result_tokens=0, max_result_chars=0)
   │
   ▼
配额使用量统计
   │
   ▼
ToolResult
```

### 1. Calculator (`calculator`)

安全的数学表达式计算器，使用 Python AST 解析而非 `eval()`。

**特性**：
- AST 白名单模式（非黑名单），只允许白名单运算符和函数
- 幂运算限制指数 ≤ 1000，防止 DDOS
- **自动缓存**：相同表达式命中直接返回历史结果（1 秒有效期）
- 带错误码的友好错误提示

**支持**：`+`, `-`, `*`, `/`, `//`, `%`, `**`, `sqrt`, `sin`, `cos`, `tan`, `log`, `abs`, `round`, `min`, `max`, `pi`, `e`

**配额**：500 tokens（计算结果通常很短）

### 2. Search (`search`)

Mock 搜索引擎，返回预定义的搜索结果。内置知识库覆盖：Python、Agent、Weather、大模型、Function Calling。

**配额**：2000 tokens

### 3. Todo Manager (`todo_manager`)

持久化任务管理器，支持 CRUD 操作。数据按**用户隔离**存储。

**核心特性**：
- **用户数据隔离**：通过 `set_user_dir()` 注入用户目录，不同用户的任务互不干扰
- **文件锁**：`os.open(O_CREAT|O_EXCL)` 原子创建锁文件，阻止并发写入
- **过期锁清理**：超过 10 秒的锁视为"孤儿锁"自动清理
- **原子写入**：先写 `.tmp` 文件，再 `replace()` 原子替换

**操作**：`create`, `list`, `update`, `delete`, `get`
**状态**：`pending`, `in_progress`, `done`

**安全确认**：`delete` 操作需要用户确认（`ask_user` 多轮交互机制），防止误删除。

**跨轮次场景示例**：
```
第一轮：
  用户: "帮我创建两个任务：实现登录功能、编写单元测试"
  Agent: 调用 todo_manager create × 2

第二轮（新会话）：
  用户: "看看现在的任务"
  Agent: 调用 todo_manager list，返回所有任务
```

### 4. File Reader (`read_file`)

读取本地文本文件内容，支持多种文件格式。

**特性**：
- **扩展名白名单**：`.txt`, `.py`, `.json`, `.csv` 等 30+ 种
- **文件大小保护**：超过 10MB 拒绝读取
- **范围读取**：`start_line` / `end_line` 参数实现精确范围读取
- **文件结构概览**：Python 文件自动提取顶层函数/类定义，Markdown 提取标题，JSON 提取顶层键名（仅超过 50 行的文件）
- **分页提示**：当内容未读完时自动提示 LLM 继续读取
- **UTF-8 编码强制**

### 5. File Writer (`write_file`)

将内容写入本地文件，支持覆盖和追加模式。

**特性**：
- **扩展名白名单** + **系统路径黑名单**（禁止写入 `C:/Windows` 等）
- **写入大小限制**：1MB
- **自动创建父目录**
- **原子写入**：覆盖模式下先写 `.tmp` 再原子替换
- **覆盖确认**：覆盖已存在文件时需用户确认（`ask_user` 多轮交互），可通过 `force=true` 跳过
- **支持多种格式**：纯文本、PDF（需 PyMuPDF）、Word（需 python-docx）

### 6. DateTime Tool (`datetime_tool`)

获取当前日期和时间信息，支持自定义格式。

**特性**：纯函数，1 秒缓存防重复调用。

---

## 高级特性

### 对话记忆压缩

当对话轮次较多时，系统会自动压缩早期消息以控制 token 消耗：

```
触发条件: 消息数 > 50 条（可配置）
压缩策略: 保留最近 20 条 + 扁平摘要（始终 1 段）
摘要生成: LLM 生成（失败时降级为文本截断）
```

**压缩流程**：
1. 分离 system、已有摘要（1 段）、非 system 消息
2. 非 system 消息分为：早期消息（待压缩）和最近消息（保留原文）
3. 将已有摘要 + 早期消息合并发给 LLM，生成一段新摘要，替换旧摘要
4. 返回 `[原始system] + [新摘要] + [最近消息]`

### LLM 自动重试

API 调用失败时采用指数退避策略自动重试：

```
第 1 次失败 → 等待 1 秒 → 重试
第 2 次失败 → 等待 2 秒 → 重试
第 3 次失败 → 等待 4 秒 → 重试
全部失败 → 抛出异常
```

### 工具自动重试

工具执行失败时按 `MAX_TOOL_RETRIES`（默认 2 次）配置重试：

```
第 1 次失败 → 等待 TOOL_RETRY_DELAY(1s) → 重试
第 2 次失败 → 等待 TOOL_RETRY_DELAY(1s) → 重试
全部失败 → 返回失败 ToolResult
```

### 多轮交互确认（ask_user）

当工具需要用户确认时才执行（如覆盖文件、删除任务）：

```
LLM 调用工具 → 工具检测到需要确认
  → 返回 ToolResult(ask_user="是否确认覆盖？")
  → Runtime 检测到 ask_user 非空
    → yield ask_user 事件 → 暂停 ReAct 循环
  → 用户输入确认文字
  → 新 ReAct 循环 → LLM 看到确认信息
    → 再次调用工具（带 confirm=true）
    → 跳过确认检查 → 实际执行操作
```

**已适配的操作**：
| 工具 | 触发条件 |
|------|---------|
| `write_file` | 覆盖已存在的文件 |
| `todo_manager` | delete 操作 |

### 工具结果缓存

纯函数工具（如 Calculator、DateTimeTool）自动缓存执行结果：

- 缓存 Key：工具名 + 参数的 JSON 序列化
- 有效期：1 秒（避免重复高频调用）
- 跳过条件：非纯函数工具（如 FileReader 每次结果可能不同）

### Per-Tool Token 配额

每个工具可以独立配置单会话 token 消耗上限：

| 工具 | 配额(token) | 理由 |
|------|------------|------|
| `calculator` | 500 | 计算结果短 |
| `search` | 2000 | 搜索结果适中 |
| `read_file` | 5000 | 文件内容可能很多 |
| 其他工具 | 0(不限额) | 仅统计不限制 |

### 会话 Token 限额

`TOTAL_TOKEN_LIMIT` 控制单次会话的总 token 上限（配置文件 `.env` 中的参数）：

- 超限时先尝试记忆压缩
- 仍超限 → 清空所有非 system 消息
- `ENABLE_TOKEN_LIMIT=true` 启用检查

### 流式输出

`LLMClient.chat_stream_detect_tools_async()` 支持逐 token 流式输出 + 并行检测 tool_calls：

```python
async for event in runtime.run_stream(user_input):
    if event["type"] == "text_chunk":
        print(event["data"], end="", flush=True)
```

### 错误码系统

所有工具异常携带标准错误码（`ErrorCode` 常量类），供 Runtime 和 LLM 理解错误类型：

| 错误码 | 含义 |
|--------|------|
| `INVALID_PARAMS` | 参数校验失败 |
| `FILE_NOT_FOUND` | 文件不存在 |
| `FILE_TOO_LARGE` | 文件大小超限 |
| `PERMISSION_DENIED` | 权限不足 |
| `TIMEOUT` | 操作超时 |
| `RATE_LIMITED` | 工具配额耗尽 |
| `DEPENDENCY_MISSING` | 缺少依赖库 |
| `ASK_USER` | 需要用户确认 |

---

## 用户管理

### 登录/注册流程

```
首次运行：
  → 提示注册（用户名 + 密码）
  → 密码加盐哈希存储
  → 创建用户数据目录

再次运行：
  → 提示登录
  → 验证密码
  → 进入主程序
```

### 数据隔离

```
.agent_data/
├── userInfo.json               ← 全局用户凭证（salt$sha256_hex）
│
├── zhangsan/                   ← 用户 A 的数据
│   ├── session/session_xxx.json
│   ├── log/trace_xxx.jsonl
│   └── task/todos.json
│
└── lisi/                       ← 用户 B 的数据（完全隔离）
    ├── session/session_xxx.json
    ├── log/trace_xxx.jsonl
    └── task/todos.json
```

---

## 健壮性与兜底机制

系统采用**五层兜底机制**，确保任何环节出问题时系统都能优雅降级：

```
Layer 1: 输入层
  ├─ 空输入跳过
  └─ 异常命令提示

Layer 2: Runtime 层
  ├─ 最大步数限制 → 强制总结
  ├─ 连续失败检测 → 放弃工具
  ├─ Token 限额 → 压缩/清空
  ├─ 记忆压缩 → 摘要生成/降级截断
  └─ LLM 调用异常 → 返回友好错误

Layer 3: LLM 层
  ├─ 自动重试（指数退避）
  ├─ 流式回退到非流式
  └─ tools=None 时强制文本回复

Layer 4: 工具层
  ├─ 参数校验失败 → 友好错误信息
  ├─ 配额检查 → RATE_LIMITED 错误
  ├─ 执行超时 → TIMEOUT 错误码
  ├─ 依赖缺失 → 提示安装命令
  └─ 缓存 → 防重复计算

Layer 5: 持久化层
  ├─ 原子写入 → 防文件损坏
  ├─ 文件锁 → 防并发写入
  ├─ os.fsync → 确保落盘
  └─ JSON 解析异常 → 返回空数据
```

---

## 扩展工具

新增工具只需 2 步：

### 步骤 1：创建工具文件

在 `agent/tools/` 下创建新文件，继承 `BaseTool`：

```python
# agent/tools/my_tool.py
from agent.tools.base import BaseTool, ToolResult, ErrorCode

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "工具功能描述（给 LLM 看的）"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "参数说明"},
            },
            "required": ["param1"],
        }

    @property
    def quota_limit(self) -> int:
        """可选：设置此工具的 token 配额，0 为不限额"""
        return 0

    @property
    def is_pure_function(self) -> bool:
        """可选：纯函数工具启用缓存"""
        return False

    def execute(self, param1: str) -> ToolResult:
        # 实现工具逻辑
        return ToolResult(success=True, result=f"结果: {param1}")
```

**可选增强**（按需实现）：

| 功能 | 实现方式 |
|------|---------|
| 异步执行 | 重写 `execute_async()` 方法 |
| 生命周期钩子 | 重写 `before_execute()` / `after_execute()` |
| 参数动态调整 | 重写 `adjust_parameters()` |
| 使用示例 | 重写 `examples` 属性 |
| 依赖检查 | 使用 `@require_import('package_name')` 装饰器 |

### 步骤 2：注册工具

在 `agent/tools/__init__.py` 中导入并加入 `ALL_TOOLS` 列表：

```python
from agent.tools.my_tool import MyTool

ALL_TOOLS: list[BaseTool] = [
    # ... 现有工具 ...
    MyTool(),
]
```

---

## 更多文档

- [ARCHITECTURE.md](ARCHITECTURE.md) — 详细架构设计文档（完整调用流程、组件详解、设计权衡）
- [tools.md](tools.md) — 工具执行架构设计（同步/异步分派模式分析）

---

## 设计理念

### 为什么从零实现而非使用 LangChain

| 因素 | 自实现 | 使用 LangChain |
|------|--------|---------------|
| 学习成本 | 高（需理解全部细节） | 低（抽象封装好） |
| 定制自由度 | 极高 | 受框架约束 |
| 调试难度 | 低（代码完全可控） | 中（需理解框架内部） |
| 项目大小 | 2000+ 行 | 框架本身数十万行 |

本项目目标是"从零实现"，理解 Agent 的工作原理。自实现可以完全控制每一行代码的行为。

### 为什么用异步 Runtime

当前架构采用 **异步 Runtime + 同步/异步工具混合执行** 的模式：

| 组件 | 模式 | 选择原因 |
|------|------|---------|
| Runtime 主循环 | 异步 (async for) | 流式 LLM 输出不阻塞，工具并发执行 |
| LLM 客户端 | 异步 (AsyncOpenAI) | 网络 I/O 密集，协程高效 |
| 无 I/O 工具 | 同步 execute() 线程池 | 无需重写，零成本迁移 |
| I/O 工具 | 可选 execute_async() 原生协程 | 可逐步收益 |

详见 [tools.md](tools.md) 第 4 节的优缺点分析。
