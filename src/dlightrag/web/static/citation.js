/* DlightRAG Web Frontend — Citation & Streaming Logic
 *
 * Security: All user input uses textContent (safe). Server-rendered HTML
 * (SSE tokens + done event) is trusted content from our own server with
 * _escape_html applied to all dynamic values in routes.py.
 */

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

// === Streaming Answer via SSE ===

async function submitQuery(query) {
    const chatMessages = document.getElementById('chat-messages');
    const emptyState = document.getElementById('empty-state');
    if (emptyState) emptyState.remove();

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
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const formData = new FormData();
        formData.append('query', query);

        const response = await fetch('/web/answer', {
            method: 'POST',
            body: formData,
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
                    handleSSEData(eventType, data, contentDiv, aiDiv, chatMessages, firstToken);
                    if (eventType === 'token' && firstToken) firstToken = false;
                    eventType = '';
                }
            }
        }
    } catch (err) {
        contentDiv.textContent = 'Connection error. Please try again.';
        contentDiv.style.color = '#c44';
    }
}

function handleSSEData(eventType, data, contentDiv, aiDiv, chatMessages, firstToken) {
    if (eventType === 'token') {
        if (firstToken) {
            contentDiv.textContent = '';
        }
        // Server-escaped HTML token — append as trusted content
        const span = document.createElement('span');
        span.textContent = data;
        contentDiv.appendChild(span);
        chatMessages.scrollTop = chatMessages.scrollHeight;
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
    } else if (eventType === 'error') {
        contentDiv.textContent = data;
        contentDiv.style.color = '#c44';
    }
}

// === Query Form ===

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('query-form');
    if (!form) return;

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const input = form.querySelector('.query-input');
        const query = input.value.trim();
        if (!query) return;
        input.value = '';
        submitQuery(query);
    });

    // Files button opens panel
    const filesBtn = document.getElementById('files-btn');
    if (filesBtn) {
        filesBtn.addEventListener('htmx:afterRequest', function () {
            openPanel('FILES');
        });
    }

    // Workspace button
    const wsBtn = document.getElementById('workspace-btn');
    if (wsBtn) {
        wsBtn.addEventListener('click', function () {
            const panelContent = document.getElementById('panel-content');
            panelContent.setAttribute('hx-get', '/web/workspaces');
            panelContent.setAttribute('hx-trigger', 'load');
            panelContent.setAttribute('hx-swap', 'innerHTML');
            htmx.process(panelContent);
            htmx.trigger(panelContent, 'load');
            openPanel('WORKSPACES');
        });
    }
});
