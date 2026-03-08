/**
 * OpenClaw-Lang WebChat — Client Application
 *
 * WebSocket-based real-time chat with tool call visualization,
 * markdown rendering, and session management.
 */

// ===== State =====
let ws = null;
let isConnected = false;
let isStreaming = false;
let currentResponse = '';
let currentMessageEl = null;
let senderId = 'webchat-' + Math.random().toString(36).substring(2, 8);
let sessions = [];
let messageHistory = [];

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    fetchStatus();
    setupInput();
});

// ===== WebSocket =====
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        isConnected = true;
        updateStatus('Connected', true);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleEvent(data);
        } catch (e) {
            console.error('Failed to parse event:', e);
        }
    };

    ws.onclose = () => {
        isConnected = false;
        updateStatus('Disconnected', false);
        // Reconnect after 3s
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        updateStatus('Error', false);
    };
}

// ===== Event Handling =====
function handleEvent(event) {
    switch (event.type) {
        case 'lifecycle':
            handleLifecycle(event);
            break;
        case 'tool_start':
            handleToolStart(event);
            break;
        case 'tool_end':
            handleToolEnd(event);
            break;
        case 'assistant_delta':
            handleAssistantDelta(event);
            break;
        case 'assistant_message':
            handleAssistantMessage(event);
            break;
        case 'nudge':
            handleNudge(event);
            break;
        case 'pong':
            break;
        case 'error':
            showError(event.message || 'Unknown error');
            break;
    }
}

function handleLifecycle(event) {
    if (event.phase === 'start') {
        isStreaming = true;
        currentResponse = '';
        showTyping(true);
    } else if (event.phase === 'end') {
        isStreaming = false;
        showTyping(false);
        finalizeMessage();
        enableInput();
    } else if (event.phase === 'error') {
        isStreaming = false;
        showTyping(false);
        showError(event.error || 'Agent error');
        enableInput();
    }
}

function handleToolStart(event) {
    showTyping(false);
    const toolEl = createToolCall(event.tool_name, event.tool_args, 'running');
    appendToChat(toolEl);
}

function handleToolEnd(event) {
    // Update the last tool call with output
    const toolCalls = document.querySelectorAll('.tool-call');
    const lastTool = toolCalls[toolCalls.length - 1];
    if (lastTool) {
        const statusEl = lastTool.querySelector('.tool-call-status');
        if (statusEl) statusEl.textContent = '✅ done';
        const bodyEl = lastTool.querySelector('.tool-call-body');
        if (bodyEl) bodyEl.textContent = event.output || '(no output)';
    }
}

function handleAssistantDelta(event) {
    showTyping(false);

    if (!currentMessageEl) {
        currentMessageEl = createAssistantMessage('');
        appendToChat(currentMessageEl);
    }

    currentResponse += event.content;
    updateMessageContent(currentMessageEl, currentResponse);
    scrollToBottom();
}

function handleAssistantMessage(event) {
    // If we already have a streaming message, update it with final content
    if (currentMessageEl && event.content) {
        currentResponse = event.content;
        updateMessageContent(currentMessageEl, currentResponse);
    }
}

function finalizeMessage() {
    if (currentMessageEl && currentResponse) {
        messageHistory.push({ role: 'assistant', content: currentResponse });
    }
    currentMessageEl = null;
    currentResponse = '';
}

// ===== Message Creation =====
function createUserMessage(content) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    return div;
}

function createAssistantMessage(content) {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar">🦞</div>
        <div class="message-content">${renderMarkdown(content)}</div>
    `;
    return div;
}

function createToolCall(name, args, status) {
    const div = document.createElement('div');
    div.className = 'tool-call';
    const argsStr = typeof args === 'object' ? JSON.stringify(args, null, 2) : String(args);
    div.innerHTML = `
        <div class="tool-call-header" onclick="toggleToolCall(this.parentElement)">
            <span class="tool-call-chevron">▶</span>
            <span class="tool-call-icon">🔧</span>
            <span class="tool-call-name">${escapeHtml(name)}</span>
            <span class="tool-call-status">${status === 'running' ? '⏳ running...' : '✅ done'}</span>
        </div>
        <div class="tool-call-body">${escapeHtml(argsStr)}</div>
    `;
    return div;
}

function toggleToolCall(el) {
    el.classList.toggle('expanded');
}

function updateMessageContent(el, content) {
    const contentEl = el.querySelector('.message-content');
    if (contentEl) {
        contentEl.innerHTML = renderMarkdown(content);
    }
}

// ===== Send Message =====
function sendMessage() {
    const textarea = document.getElementById('inputTextarea');
    const content = textarea.value.trim();
    if (!content || !isConnected || isStreaming) return;

    // Hide welcome screen
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'none';

    // Add user message to chat
    const userMsg = createUserMessage(content);
    appendToChat(userMsg);
    messageHistory.push({ role: 'user', content });

    // Send via WebSocket
    ws.send(JSON.stringify({
        type: 'message',
        content: content,
        sender_id: senderId,
        channel: 'webchat',
    }));

    // Clear input
    textarea.value = '';
    textarea.style.height = 'auto';
    disableInput();
    scrollToBottom();
}

function quickSend(content) {
    const textarea = document.getElementById('inputTextarea');
    textarea.value = content;
    sendMessage();
}

// ===== Input Management =====
function setupInput() {
    const textarea = document.getElementById('inputTextarea');
    textarea.addEventListener('input', () => {
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = !textarea.value.trim() || isStreaming;
    });
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = !textarea.value.trim() || isStreaming;
}

function disableInput() {
    document.getElementById('sendBtn').disabled = true;
    document.getElementById('inputTextarea').disabled = true;
}

function enableInput() {
    const textarea = document.getElementById('inputTextarea');
    textarea.disabled = false;
    textarea.focus();
    document.getElementById('sendBtn').disabled = !textarea.value.trim();
}

// ===== UI Helpers =====
function appendToChat(element) {
    const chatArea = document.getElementById('chatArea');
    const typingIndicator = document.getElementById('typingIndicator');
    chatArea.insertBefore(element, typingIndicator);
    scrollToBottom();
}

function scrollToBottom() {
    const chatArea = document.getElementById('chatArea');
    requestAnimationFrame(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

function showTyping(visible) {
    const indicator = document.getElementById('typingIndicator');
    indicator.classList.toggle('visible', visible);
    if (visible) scrollToBottom();
}

function showError(message) {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, #ef4444, #dc2626);">⚠️</div>
        <div class="message-content" style="border-color: rgba(239,68,68,0.3);">
            <strong style="color: #ef4444;">Error:</strong> ${escapeHtml(message)}
        </div>
    `;
    appendToChat(div);
    enableInput();
}

function handleNudge(event) {
    // Proactive nudge from the cron scheduler — render as a special message
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'none';

    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, #eab308, #f59e0b);">🔔</div>
        <div class="message-content" style="border-color: rgba(234,179,8,0.3);">
            <div style="font-size:11px; color: var(--text-tertiary); margin-bottom:4px;">
                ⏰ Scheduled: <strong style="color: var(--accent-secondary);">${escapeHtml(event.job_name || 'Nudge')}</strong>
            </div>
            ${renderMarkdown(event.content || 'No content')}
        </div>
    `;
    appendToChat(div);
    scrollToBottom();

    // Play a subtle notification sound (browser notification API)
    if (Notification.permission === 'granted') {
        new Notification('🦞 OpenClaw Nudge', { body: event.job_name || 'Scheduled task completed' });
    } else if (Notification.permission !== 'denied') {
        Notification.requestPermission();
    }
}

function updateStatus(text, connected) {
    document.getElementById('statusText').textContent = text;
    const dot = document.querySelector('.status-dot');
    dot.style.background = connected ? 'var(--color-success)' : 'var(--color-error)';
    dot.style.boxShadow = connected
        ? '0 0 8px rgba(34, 197, 94, 0.5)'
        : '0 0 8px rgba(239, 68, 68, 0.5)';
}

function newChat() {
    // Reset state
    messageHistory = [];
    currentMessageEl = null;
    currentResponse = '';
    senderId = 'webchat-' + Math.random().toString(36).substring(2, 8);

    // Clear chat
    const chatArea = document.getElementById('chatArea');
    const children = Array.from(chatArea.children);
    children.forEach(child => {
        if (!child.id || (child.id !== 'welcomeScreen' && child.id !== 'typingIndicator')) {
            child.remove();
        }
    });

    // Show welcome
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'flex';

    enableInput();
}

// ===== API Calls =====
async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('headerModel').textContent = data.model || 'Unknown';
        updateStatus(`Connected · ${data.model}`, true);
    } catch (e) {
        console.error('Failed to fetch status:', e);
    }
}

// ===== Markdown Rendering =====
function renderMarkdown(text) {
    if (!text) return '';

    // Escape HTML first
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered lists
    html = html.replace(/^[\s]*[-*] (.+)$/gm, '<li>$1</li>');

    // Ordered lists
    html = html.replace(/^[\s]*\d+\. (.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> items
    html = html.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n+/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[1-3]>)/g, '$1');
    html = html.replace(/(<\/h[1-3]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
