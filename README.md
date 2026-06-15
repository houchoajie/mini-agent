# Mini Agent - 从零实现的最小可用 Agent

## 项目简介

从零实现的最小可用 AI Agent 系统，核心 runtime 完全自主实现，不依赖 LangChain / OpenHands 等现成 Agent 框架。

**核心能力**：
- 多轮对话 + Session 持久化
- ReAct 循环（推理 → 行动 → 观察 → 继续）
- 6 个内置工具（calculator / search / todo_manager / read_file / write_file / datetime_tool）
- 跨轮次状态持久化（任务管理 + 文件读写）
- 对话记忆自动压缩（长对话优化）
- LLM 调用自动重试（指数退避）
- 流式输出支持
- 完整执行 Trace 日志

---

## 快速开始

### 1. 环境要求

- Python >= 3.13
- OpenAI API Key（或兼容的 API 服务）

### 2. 安装依赖

```bash
pip install openai python-dotenv
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
# 新建会话
python main.py

# 恢复指定会话
python main.py --session session_abc123

# 列出所有会话
python main.py --list
```

### 5. 交互命令

| 命令 | 说明 |
|------|------|
| `/new` | 新建会话 |
| `/switch <id>` | 切换到指定会话 |
| `/sessions` | 列出所有会话 |
| `/trace` | 查看执行追踪摘要 |
| `/clear` | 清空当前对话历史 |
| `/quit` | 退出 |

---

## 系统设计

### 整体架构

```
┌──────────────────────────────────────────────────┐
│                   main.py (CLI)                   │
│          用户输入 → 命令解析 → 结果显示              │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│              Agent Runtime (核心)                  │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐   │
│  │  LLM    │  │  Tool    │  │   Session      │   │
│  │ Client  │  │ Executor │  │   Manager      │   │
│  │+重试    │  │          │  │                │   │
│  │+流式    │  │          │  │                │   │
│  └────┬────┘  └────┬─────┘  └───────┬────────┘   │
│       │            │                │             │
│  ┌────▼────────────▼────────────────▼─────────┐   │
│  │           ReAct 主循环                       │   │
│  │  输入 → LLM推理 → 工具调用? → 执行 → 反馈    │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
│  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  Memory 记忆管理  │  │  Trace Logger 日志   │   │
│  │  (自动摘要压缩)   │  │  (控制台+文件)        │   │
│  └──────────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### 目录结构

```
├── main.py                  # 入口文件 - 交互式 CLI
├── agent/
│   ├── __init__.py
│   ├── runtime.py           # 核心 - Agent ReAct 循环引擎
│   ├── llm.py               # LLM 客户端 - OpenAI API 封装 + 重试 + 流式
│   ├── session.py           # 会话管理 - 多轮对话 + 状态持久化
│   ├── memory.py            # 对话记忆 - 历史摘要压缩
│   ├── trace.py             # 执行追踪日志系统
│   └── tools/
│       ├── __init__.py      # 工具注册表
│       ├── base.py          # 工具基类 + ToolResult
│       ├── calculator.py    # 数学计算器（AST 安全求值）
│       ├── search.py        # Mock 搜索引擎
│       ├── todo_manager.py  # 任务管理器（持久化存储）
│       ├── file_reader.py   # 文件读取工具
│       ├── file_writer.py   # 文件写入工具
│       └── datetime_tool.py # 日期时间工具
├── .env.example             # 环境变量模板
├── pyproject.toml           # 项目配置
└── .agent_data/             # 运行时数据（自动生成）
    ├── sessions/            # 会话持久化文件
    ├── logs/                # Trace 日志文件
    └── todos.json           # 任务数据
```

### ReAct 循环流程

```
用户输入
   │
   ▼
┌──────────────────────┐
│  对话记忆压缩检查      │
│  (消息>50条时自动压缩) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  发送消息历史到 LLM   │◄────────────────────┐
│  (附带工具定义)       │                     │
└──────────┬───────────┘                     │
           │                                  │
           ▼                                  │
   LLM 返回 tool_calls？                      │
    ├── 是 ──► 执行工具                        │
    │          │                              │
    │          ▼                              │
    │    工具结果加入历史 ─────────────────────┘
    │    (step++, 检查 max_steps)
    │
    └── 否 ──► 输出文本回复 ──► 保存会话 ──► 结束本轮
```

### 核心组件说明

#### 1. Agent Runtime (`agent/runtime.py`)

核心 ReAct 循环引擎。每次 `run(user_input)` 调用：

1. 将用户输入追加到对话历史
2. **检查对话记忆是否需要压缩**（超过 50 条消息时自动压缩）
3. 调用 LLM（传递历史 + 工具 schema）
4. 若 LLM 请求工具调用 → 执行工具 → 结果回传 → 重复
5. 若 LLM 直接回复 → 返回结果
6. 超过 `max_steps` → 强制终止

#### 2. LLM Client (`agent/llm.py`)

封装 OpenAI Chat Completions API，支持：
- **Function Calling**：工具调用能力
- **自动重试**：失败时指数退避重试（1s → 2s → 4s），默认 3 次
- **流式输出**：`chat_stream()` 方法支持逐 token 输出

配置项：
- `OPENAI_API_KEY`：API 密钥
- `OPENAI_BASE_URL`：API 端点（支持第三方兼容服务）
- `OPENAI_MODEL`：模型名称

#### 3. Conversation Memory (`agent/memory.py`)

对话记忆管理器，解决长对话的 token 消耗问题：

- **触发条件**：当非 system 消息数超过 `max_messages`（默认 50）
- **压缩策略**：保留最近 `keep_recent` 条消息（默认 20），早期消息压缩为摘要
- **摘要生成**：优先使用 LLM 生成高质量摘要，失败时降级为简单截断
- **增量更新**：多次压缩时会合并旧摘要和新内容

#### 4. Session Manager (`agent/session.py`)

- 每个会话包含：session_id、消息历史、时间戳、元数据
- 以 JSON 文件持久化到 `.agent_data/sessions/`
- 支持创建、加载、保存、列出、删除会话

#### 5. Trace Logger (`agent/trace.py`)

- 实时控制台输出 + JSONL 文件持久化
- 记录级别：system / user / llm_request / llm_response / tool_call / tool_result / error / step
- 包含 token 使用统计

---

## 工具说明

### 1. Calculator (`calculator`)

安全的数学表达式计算器，使用 Python AST 解析而非 `eval()`。

**支持**：`+`, `-`, `*`, `/`, `//`, `%`, `**`, `sqrt`, `sin`, `cos`, `tan`, `log`, `abs`, `round`, `min`, `max`, `pi`, `e`

**示例**：
```
输入: expression="2 ** 10 + sqrt(144)"
输出: "计算结果: 2 ** 10 + sqrt(144) = 1036"
```

### 2. Search (`search`)

Mock 搜索引擎，返回预定义的搜索结果。内置知识库覆盖：Python、Agent、Weather、大模型、Function Calling。

**示例**：
```
输入: query="Python 教程"
输出: 包含 Python 官方文档和教程链接的搜索结果
```

### 3. Todo Manager (`todo_manager`)

持久化任务管理器，支持 CRUD 操作。数据存储到 `.agent_data/todos.json`。

**操作**：`create`, `list`, `update`, `delete`, `get`
**状态**：`pending`, `in_progress`, `done`

**跨轮次场景示例**：
```
第一轮：
  用户: "帮我创建两个任务：实现登录功能、编写单元测试"
  Agent: 调用 todo_manager create × 2，创建 task_001, task_002

第二轮（新对话或同一对话）：
  用户: "看看现在的任务列表"
  Agent: 调用 todo_manager list，返回已创建的任务

  用户: "把第一个任务标记为进行中"
  Agent: 调用 todo_manager update，更新 task_001 状态为 in_progress
```

### 4. File Reader (`read_file`)

读取本地文本文件内容，支持多种文件格式。

**安全机制**：
- 扩展名白名单过滤（.txt, .py, .json, .csv 等 30+ 种）
- 最大行数限制（默认 200 行，可配置）
- 文件大小保护（超过 10MB 拒绝读取）
- UTF-8 编码强制

**示例**：
```
输入: file_path="C:/Users/xxx/main.py", max_lines=100
输出: "文件: C:/Users/xxx/main.py\n大小: 1,234 字节 | 行数: 50\n..."
```

### 5. File Writer (`write_file`)

将内容写入本地文件，支持覆盖和追加模式。

**安全机制**：
- 扩展名白名单过滤
- 系统路径黑名单保护（禁止写入 C:/Windows 等）
- 写入大小限制（1MB）
- 自动创建父目录

**示例**：
```
输入: file_path="notes.txt", content="今日待办：...", append=false
输出: "文件写入成功！路径: notes.txt, 大小: 256 字节"
```

### 6. DateTime Tool (`datetime_tool`)

获取当前日期和时间信息，支持自定义格式。

**示例**：
```
输入: format="%Y-%m-%d %H:%M:%S"
输出: "当前时间: 2024-01-15 14:30:00\nUnix 时间戳: 1705305000\n星期: 一"
```

---

## 高级特性

### 对话记忆压缩

当对话轮次较多时，系统会自动压缩早期消息以控制 token 消耗：

```
触发条件: 消息数 > 50 条（可配置）
压缩策略: 保留最近 20 条 + 早期摘要
摘要生成: LLM 生成（失败时降级为截断）
```

**压缩流程**：
1. 每次 `run()` 前检查是否需要压缩
2. 将早期消息格式化为文本
3. 调用 LLM 生成摘要（或简单截断）
4. 替换消息历史为: [system] + [摘要] + [最近 20 条]

### LLM 自动重试

API 调用失败时采用指数退避策略自动重试：

```
第 1 次失败 → 等待 1 秒 → 重试
第 2 次失败 → 等待 2 秒 → 重试
第 3 次失败 → 等待 4 秒 → 重试
全部失败 → 抛出异常
```

### 流式输出

`LLMClient.chat_stream()` 支持逐 token 流式输出，提升用户体验：

```python
for chunk in llm.chat_stream(messages):
    print(chunk, end="", flush=True)
```

---

## Memory（记忆）的召回时机与放置方式

本系统的 "Memory" 体现为三个层面：

### 1. 对话历史 Memory（短期记忆）

**放置方式**：
- 所有对话消息（user / assistant / tool）按顺序存储在 `Session.messages` 列表中
- 每轮 `run()` 调用时，完整的消息历史传递给 LLM

**召回时机**：
- 每次 LLM 调用前，完整的对话历史作为 `messages` 参数传入
- LLM 可以看到之前所有的用户输入、自己的回复、工具调用结果

**持久化**：
- 每轮对话结束后自动保存到 `.agent_data/sessions/<id>.json`
- 支持通过 `/switch` 或 `--session` 恢复历史会话

**压缩机制**：
- 当消息数超过 50 条时，自动触发压缩
- 早期消息被压缩为摘要，保留最近 20 条
- 摘要由 LLM 生成，保留关键上下文

### 2. 工具状态 Memory（长期记忆）

**放置方式**：
- `todo_manager` 工具将任务数据持久化到 `.agent_data/todos.json`
- 数据独立于对话历史，跨会话可用

**召回时机**：
- 当用户提到任务相关的问题时，Agent 会调用 `todo_manager list/get` 查询已有状态
- LLM 通过系统提示词知道"任务是持久化的"，会主动调用工具获取状态

### 3. System Prompt 注入（隐式记忆）

**放置方式**：
- 系统提示词在会话创建时注入为第一条 `system` 消息
- 定义了 Agent 的角色、能力和行为规范

**召回时机**：
- 每次 LLM 调用时都会看到 system prompt
- 确保 Agent 在所有轮次中保持一致的行为

---

## 扩展工具

新增工具只需 2 步：

### 步骤 1：创建工具文件

在 `agent/tools/` 下创建新文件，继承 `BaseTool`：

```python
# agent/tools/my_tool.py
from agent.tools.base import BaseTool, ToolResult

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

    def execute(self, param1: str) -> ToolResult:
        # 实现工具逻辑
        return ToolResult(success=True, result=f"结果: {param1}")
```

### 步骤 2：注册工具

在 `agent/tools/__init__.py` 中导入并加入 `ALL_TOOLS` 列表：

```python
from agent.tools.my_tool import MyTool

ALL_TOOLS: list[BaseTool] = [
    # ... 现有工具 ...
    MyTool(),
]
```

### 步骤 3：更新 System Prompt

在 `agent/session.py` 的 `_default_system_prompt()` 中添加新工具的使用说明。

---

## AI Prompt 与问题解决记录

### 开发过程中使用的 AI Prompt

1. **架构设计阶段**
   - "设计一个最小可用 Agent 的 Python 项目结构，包含 ReAct 循环、工具系统、会话管理"
   - "如何在不使用 LangChain 的情况下实现 OpenAI Function Calling 的工具调用循环"

2. **工具实现阶段**
   - "如何用 Python AST 实现安全的数学表达式计算器，避免 eval 的安全风险"
   - "设计一个 Mock 搜索引擎的数据结构和匹配逻辑"
   - "实现持久化任务管理器，支持 CRUD 和 JSON 文件存储"
   - "设计文件读写工具的安全机制（白名单、黑名单、大小限制）"

3. **增强功能阶段**
   - "如何实现对话历史的自动摘要压缩，控制 token 消耗"
   - "OpenAI API 调用的指数退避重试策略实现"
   - "流式输出 chat_stream 的实现方式"

4. **调试与优化阶段**
   - "OpenAI tool_calls 响应中 content 为 None 的处理方式"
   - "pyproject.toml build-backend 配置 setuptools.build_meta 的正确写法"
   - "PowerShell 中 && 语法不支持的解决方案"

### 关键问题解决记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `pip install -e .` 报错 | pyproject.toml build-backend 配置错误 | 改为 `setuptools.build_meta` |
| PowerShell 不支持 `&&` | Windows PowerShell 5.x 语法限制 | 改用 `;` 分隔或单独执行命令 |
| tool_calls 中 content 为 None | OpenAI API 在有工具调用时 content 可能为空 | `response.get("content") or ""` 空值兜底 |
| LLM 不主动查询任务状态 | 系统提示词未强调跨轮次查询行为 | 在 system prompt 中明确说明持久化特性 |
| 长对话 token 消耗过大 | 消息历史无限增长 | 实现 ConversationMemory 自动压缩 |
| API 调用偶发失败 | 网络不稳定或限流 | 实现指数退避重试机制 |
