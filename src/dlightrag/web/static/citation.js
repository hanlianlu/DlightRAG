/* DlightRAG Web Frontend — Citation & Streaming Logic
 *
 * Security: All user input uses textContent (safe). Server-rendered HTML
 * (SSE tokens + done event) is trusted content from our own server with
 * Jinja2 auto-escaping applied to all dynamic values.
 */

// === Conversation State ===

// Conversation history sent to backend for multi-turn context.
// Format: [{role: "user", content: "..."}, {role: "assistant", content: "..."}, ...]
const conversationHistory = [];

// Server-driven budget — updated from SSE `meta` event each response.
// Defaults are generous fallbacks used before the first server response.
let historyCharBudget = 300000;  // chars (~150K tokens mixed scripts)
let historyMaxMessages = 100;    // message count cap (50 turns × 2)

/**
 * Build history window to send to backend.
 * Applies server-driven limits: min(max messages, char budget).
 * The backend still does authoritative token-level truncation.
 */
function getHistoryWindow() {
    if (conversationHistory.length === 0) return [];

    // Step 1: cap by message count
    let window = conversationHistory.slice(-historyMaxMessages);

    // Step 2: cap by char budget (walk backwards)
    let totalChars = 0;
    let cutoff = window.length;
    for (let i = window.length - 1; i >= 0; i--) {
        totalChars += (window[i].content || '').length;
        if (totalChars > historyCharBudget) {
            cutoff = i + 1;
            break;
        }
    }
    if (cutoff > 0 && cutoff < window.length) {
        window = window.slice(cutoff);
    }

    return window;
}

// === Panel Management ===

function openPanel(title) {
    const panel = document.getElementById('panel');
    const panelTitle = document.getElementById('panel-title');
    panel.classList.add('open');
    if (title) panelTitle.textContent = title;
}

function closePanel() {
    document.getElementById('panel').classList.remove('open');
}

// === Citation Filter-Focus ===

function filterSource(badge) {
    const ref = badge.dataset.ref;
    const chunk = badge.dataset.chunk;
    const panelContent = document.getElementById('panel-content');

    // Find the source data for the current answer
    const answerEl = badge.closest('.ai-message');
    if (!answerEl) return;

    const sourceData = answerEl.querySelector('.source-data');
    if (!sourceData) return;

    // Clone source data into panel (trusted server-rendered HTML)
    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    clone.style.display = '';
    while (clone.firstChild) {
        panelContent.appendChild(clone.firstChild);
    }

    // Filter: show only the clicked chunk
    const chunks = panelContent.querySelectorAll('.source-chunk');
    let found = false;
    chunks.forEach(function(c) {
        if (c.dataset.ref === ref && c.dataset.chunk === chunk) {
            c.style.display = '';
            c.classList.add('active');
            found = true;
        } else {
            c.style.display = 'none';
            c.classList.remove('active');
        }
    });

    // Show "Show all" button
    const showAllBtn = panelContent.querySelector('.show-all-btn');
    if (showAllBtn && found) {
        showAllBtn.style.display = '';
    }

    openPanel('SOURCES');
}

function showAllSources() {
    const panelContent = document.getElementById('panel-content');
    const chunks = panelContent.querySelectorAll('.source-chunk');
    chunks.forEach(function(c) {
        c.style.display = '';
        c.classList.remove('active');
    });

    const showAllBtn = panelContent.querySelector('.show-all-btn');
    if (showAllBtn) showAllBtn.style.display = 'none';
}

// === Layout State ===

function activateChatMode() {
    const app = document.querySelector('.app');
    if (!app.classList.contains('has-messages')) {
        app.classList.add('has-messages');
    }
}

// === Streaming Answer via SSE ===

async function submitQuery(query) {
    const chatMessages = document.getElementById('chat-messages');
    const chatArea = document.getElementById('chat-area');

    // Switch to chat mode (bottom input bar)
    activateChatMode();

    // Add user message (textContent = safe from XSS)
    const userDiv = document.createElement('div');
    userDiv.className = 'user-message';
    userDiv.textContent = query;
    chatMessages.appendChild(userDiv);

    // Add AI message container
    const aiDiv = document.createElement('div');
    aiDiv.className = 'ai-message';

    const headerDiv = document.createElement('div');
    headerDiv.className = 'ai-message-header';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.textContent = '\u25CF';
    headerDiv.appendChild(dot);
    headerDiv.appendChild(document.createTextNode(' DlightRAG'));
    aiDiv.appendChild(headerDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'ai-message-content';
    const streamingDot = document.createElement('span');
    streamingDot.className = 'streaming-dot';
    contentDiv.appendChild(streamingDot);
    aiDiv.appendChild(contentDiv);

    chatMessages.appendChild(aiDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    // Build history window using server-driven budget
    const historyWindow = getHistoryWindow();

    let fullAnswer = '';
    let success = false;

    try {
        const response = await fetch('/web/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                conversation_history: historyWindow,
            }),
        });

        if (!response.ok) {
            contentDiv.textContent = 'Service error. Please try again.';
            contentDiv.style.color = '#c44';
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let firstToken = true;

        while (true) {
            const result = await reader.read();
            if (result.done) break;

            buffer += decoder.decode(result.value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';  // Keep incomplete line

            let eventType = '';
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (eventType === 'token') {
                        fullAnswer += data;
                    }
                    handleSSEData(eventType, data, contentDiv, aiDiv, chatArea, firstToken);
                    if (eventType === 'token' && firstToken) firstToken = false;
                    eventType = '';
                }
            }
        }

        success = true;
    } catch (err) {
        contentDiv.textContent = 'Connection error. Please try again.';
        contentDiv.style.color = '#c44';
    }

    // Only record in history on success (keeps history consistent)
    if (success && fullAnswer) {
        conversationHistory.push({role: 'user', content: query});
        conversationHistory.push({role: 'assistant', content: fullAnswer});
    }
}

function handleSSEData(eventType, data, contentDiv, aiDiv, chatArea, firstToken) {
    if (eventType === 'token') {
        if (firstToken) {
            contentDiv.textContent = '';
        }
        // Server-escaped HTML token — append as trusted content
        const span = document.createElement('span');
        span.textContent = data;
        contentDiv.appendChild(span);
        chatArea.scrollTop = chatArea.scrollHeight;
    } else if (eventType === 'done') {
        // Trusted server-rendered enriched HTML with citation badges + source data
        const tmp = document.createElement('div');
        tmp.innerHTML = data;  // eslint-disable-line -- trusted server content

        const answerContent = tmp.querySelector('#answer-content');
        const sourceData = tmp.querySelector('#source-data');

        if (answerContent) {
            contentDiv.innerHTML = answerContent.innerHTML;  // eslint-disable-line -- trusted server content
        }

        if (sourceData) {
            // Attach source data to this answer (hidden)
            sourceData.className = 'source-data';
            sourceData.style.display = 'none';
            sourceData.removeAttribute('id');
            aiDiv.appendChild(sourceData);
        }
    } else if (eventType === 'meta') {
        // Server-driven budget update
        try {
            const meta = JSON.parse(data);
            if (meta.history_char_budget) historyCharBudget = meta.history_char_budget;
            if (meta.history_max_messages) historyMaxMessages = meta.history_max_messages;
        } catch (e) {
            // Ignore parse errors; keep existing budget
        }
    } else if (eventType === 'error') {
        contentDiv.textContent = data;
        contentDiv.style.color = '#c44';
    }
}

// === Query Form ===

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('query-form');
    if (!form) return;

    const textarea = form.querySelector('.composer-input');

    // Auto-resize textarea as user types
    function autoResize() {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
    }

    if (textarea) {
        textarea.addEventListener('input', autoResize);

        // Submit on Enter (Shift+Enter for newline)
        textarea.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    }

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const query = textarea.value.trim();
        if (!query) return;
        textarea.value = '';
        textarea.style.height = 'auto';
        submitQuery(query);
    });

    // Files button opens panel
    const filesBtn = document.getElementById('files-btn');
    if (filesBtn) {
        filesBtn.addEventListener('htmx:afterRequest', function () {
            openPanel('FILES');
        });
    }

    // Workspace button opens panel (htmx loads content, JS opens panel)
    const wsBtn = document.getElementById('workspace-btn');
    if (wsBtn) {
        wsBtn.addEventListener('htmx:afterRequest', function () {
            openPanel('WORKSPACES');
        });
    }
});
