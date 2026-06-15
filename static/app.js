/**
 * Mini Agent - 前端交互逻辑
 * 
 * 功能：
 * 1. SSE 流式对话
 * 2. 会话管理
 * 3. 工具列表展示
 * 4. Trace 日志查看
 * 5. 系统信息展示
 */

// ============================================================
// 全局状态
// ============================================================
let currentSessionId = null;
let isLoading = false;

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    loadSystemInfo();
    loadTools();
    loadSessions();
    setupEventListeners();
});

/**
 * 初始化 UI 组件
 */
function initializeUI() {
    // 自动调整 textarea 高度
    const textarea = document.getElementById('user-input');
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

/**
 * 设置事件监听器
 */
function setupEventListeners() {
    // 表单提交
    document.getElementById('chat-form').addEventListener('submit', handleChatSubmit);

    // 新建会话
    document.getElementById('btn-new-session').addEventListener('click', createNewSession);

    // 切换 Trace 面板
    document.getElementById('btn-toggle-trace').addEventListener('click', toggleTracePanel);
    document.getElementById('btn-close-trace').addEventListener('click', toggleTracePanel);

    // 清空对话
    document.getElementById('btn-clear-chat').addEventListener('click', clearChat);
}

// ============================================================
// 对话功能
// ============================================================

/**
 * 处理表单提交
 */
async function handleChatSubmit(e) {
    e.preventDefault();

    if (isLoading) return;

    const textarea = document.getElementById('user-input');
    const message = textarea.value.trim();
    if (!message) return;

    // 清空输入
    textarea.value = '';
    textarea.style.height = 'auto';

    // 显示用户消息
    addMessage('user', message);

    // 发送请求（流式）
    await sendChatStream(message);
}

/**
 * 添加消息到聊天区
 */
function addMessage(role, content, isHTML = false) {
    const chatMessages = document.getElementById('chat-messages');

    // 移除欢迎消息
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (isHTML) {
        contentDiv.innerHTML = content;
    } else {
        contentDiv.textContent = content;
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return contentDiv;
}

/**
 * 添加工具调用信息
 */
function addToolCallInfo(toolName, args, result = null) {
    const chatMessages = document.getElementById('chat-messages');

    const infoDiv = document.createElement('div');
    infoDiv.className = 'tool-call-info';
    infoDiv.innerHTML = `
        <div class="tool-name">🔨 调用工具: ${toolName}</div>
        <div class="tool-args">参数: <code>${JSON.stringify(args)}</code></div>
        ${result ? `<div class="tool-result">结果: ${result}</div>` : ''}
    `;

    chatMessages.appendChild(infoDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * 显示加载动画
 */
function showLoading() {
    const chatMessages = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message agent';
    loadingDiv.id = 'loading-indicator';
    loadingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * 隐藏加载动画
 */
function hideLoading() {
    const loading = document.getElementById('loading-indicator');
    if (loading) {
        loading.remove();
    }
}

/**
 * 流式发送聊天请求（SSE）
 */
async function sendChatStream(message) {
    isLoading = true;
    const sendBtn = document.getElementById('btn-send');
    sendBtn.disabled = true;

    showLoading();

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        hideLoading();

        // 创建 Agent 消息容器
        const agentContent = addMessage('agent', '', true);
        let agentText = '';

        // 读取 SSE 事件流
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 解析 SSE 事件
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                const eventMatch = line.match(/^event: (.+)$/m);
                const dataMatch = line.match(/^data: (.+)$/m);

                if (eventMatch && dataMatch) {
                    const event = eventMatch[1];
                    const data = dataMatch[1];

                    handleSSEEvent(event, data, agentContent);
                }
            }
        }

    } catch (error) {
        hideLoading();
        addMessage('agent', `❌ 错误: ${error.message}`);
        console.error('Chat error:', error);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
    }
}

/**
 * 处理 SSE 事件
 */
function handleSSEEvent(event, data, agentContent) {
    try {
        switch (event) {
            case 'user_input':
                // 用户输入事件（已在发送前显示）
                break;

            case 'tool_call':
                const toolData = JSON.parse(data);
                addToolCallInfo(toolData.tool, toolData.args);
                break;

            case 'tool_result':
                const resultData = JSON.parse(data);
                // 可以在工具调用信息中添加结果
                break;

            case 'text':
                // 文本输出 - 逐 token 追加（后端流式输出）
                agentContent.textContent += JSON.parse(data);
                // 滚动到底部
                document.getElementById('chat-messages').scrollTop = 
                    document.getElementById('chat-messages').scrollHeight;
                break;

            case 'done':
                const doneData = JSON.parse(data);
                currentSessionId = doneData.session_id;
                document.getElementById('current-session-id').textContent = currentSessionId;
                // 重新加载会话列表
                loadSessions();
                break;

            case 'error':
                agentContent.innerHTML += `<br><span style="color: red;">❌ ${data}</span>`;
                break;

            case 'step':
                console.log('Step:', data);
                break;

            case 'system':
                console.log('System:', data);
                break;
        }
    } catch (e) {
        console.error('Event handling error:', e);
    }
}

// ============================================================
// 会话管理
// ============================================================

/**
 * 创建新会话
 */
async function createNewSession() {
    try {
        const response = await fetch('/api/sessions/new', {
            method: 'POST',
        });
        const data = await response.json();
        currentSessionId = data.session_id;
        document.getElementById('current-session-id').textContent = currentSessionId;

        // 清空聊天
        clearChat();

        // 重新加载会话列表
        loadSessions();

        console.log('新会话已创建:', currentSessionId);
    } catch (error) {
        console.error('创建会话失败:', error);
    }
}

/**
 * 加载会话列表
 */
async function loadSessions() {
    try {
        const response = await fetch('/api/sessions');
        const sessions = await response.json();

        const sessionsList = document.getElementById('sessions-list');
        sessionsList.innerHTML = '';

        if (sessions.length === 0) {
            sessionsList.innerHTML = '<p style="color: #9CA3AF; font-size: 12px;">暂无会话</p>';
            return;
        }

        sessions.forEach(session => {
            const sessionDiv = document.createElement('div');
            sessionDiv.className = 'session-item';
            if (session.session_id === currentSessionId) {
                sessionDiv.classList.add('active');
            }

            sessionDiv.innerHTML = `
                <span class="session-id">${session.session_id}</span>
                <span class="session-meta">${session.message_count} 条消息</span>
            `;

            sessionDiv.addEventListener('click', () => switchSession(session.session_id));
            sessionsList.appendChild(sessionDiv);
        });
    } catch (error) {
        console.error('加载会话列表失败:', error);
    }
}

/**
 * 切换会话
 */
async function switchSession(sessionId) {
    try {
        const response = await fetch(`/api/sessions/switch?session_id=${sessionId}`, {
            method: 'POST',
        });
        const data = await response.json();
        currentSessionId = data.session_id;
        document.getElementById('current-session-id').textContent = currentSessionId;

        // 清空聊天并加载新会话历史
        clearChat();
        loadSessions();

        console.log('已切换到会话:', currentSessionId);
    } catch (error) {
        console.error('切换会话失败:', error);
    }
}

// ============================================================
// 工具列表
// ============================================================

/**
 * 加载工具列表
 */
async function loadTools() {
    try {
        const response = await fetch('/api/tools');
        const tools = await response.json();

        const toolsList = document.getElementById('tools-list');
        toolsList.innerHTML = '';

        tools.forEach(tool => {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'tool-item';
            toolDiv.innerHTML = `
                <div class="tool-name">${tool.name}</div>
                <div class="tool-desc">${tool.description}</div>
            `;
            toolsList.appendChild(toolDiv);
        });
    } catch (error) {
        console.error('加载工具列表失败:', error);
    }
}

// ============================================================
// Trace 日志
// ============================================================

/**
 * 切换 Trace 面板显示
 */
async function toggleTracePanel() {
    const panel = document.getElementById('trace-panel');
    panel.classList.toggle('open');

    if (panel.classList.contains('open')) {
        await loadTrace();
    }
}

/**
 * 加载 Trace 日志
 */
async function loadTrace() {
    try {
        const response = await fetch('/api/trace?limit=100');
        const entries = await response.json();

        const traceEntries = document.getElementById('trace-entries');
        traceEntries.innerHTML = '';

        if (entries.length === 0) {
            traceEntries.innerHTML = '<p style="color: #9CA3AF;">暂无日志</p>';
            return;
        }

        entries.forEach(entry => {
            const entryDiv = document.createElement('div');
            entryDiv.className = `trace-entry ${entry.level}`;
            entryDiv.innerHTML = `
                <div class="timestamp">${entry.timestamp}</div>
                <div class="message">${entry.message || JSON.stringify(entry)}</div>
            `;
            traceEntries.appendChild(entryDiv);
        });

        // 滚动到底部
        traceEntries.scrollTop = traceEntries.scrollHeight;
    } catch (error) {
        console.error('加载 Trace 失败:', error);
    }
}

// ============================================================
// 系统信息
// ============================================================

/**
 * 加载系统信息
 */
async function loadSystemInfo() {
    try {
        const response = await fetch('/api/info');
        const info = await response.json();

        const systemInfo = document.getElementById('system-info');
        systemInfo.innerHTML = `
            <p><strong>版本:</strong> ${info.version}</p>
            <p><strong>模型:</strong> ${info.model}</p>
            <p><strong>工具数量:</strong> ${info.tools_count}</p>
            <p><strong>活跃会话:</strong> ${info.active_sessions}</p>
        `;
    } catch (error) {
        console.error('加载系统信息失败:', error);
    }
}

// ============================================================
// 辅助功能
// ============================================================

/**
 * 清空聊天
 */
function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h2>👋 欢迎使用 Mini Agent</h2>
            <p>这是一个从零实现的 AI Agent 系统，支持：</p>
            <ul>
                <li>多轮对话 + 会话持久化</li>
                <li>6 个内置工具（计算器、搜索、任务管理、文件读写、时间查询）</li>
                <li>ReAct 循环自动推理</li>
                <li>流式输出实时响应</li>
            </ul>
            <p class="hint">💡 在下方输入框开始对话吧！</p>
        </div>
    `;
}
