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

// === Image Upload State ===

const MAX_IMAGES = 3;
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const pendingImages = []; // [{file, base64, objectUrl}]

function addImage(file) {
    if (pendingImages.length >= MAX_IMAGES) return;
    if (!file.type.startsWith('image/')) return;
    if (file.size > MAX_IMAGE_SIZE) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const base64 = e.target.result.split(',')[1]; // strip data:...;base64,
        const objectUrl = URL.createObjectURL(file);
        pendingImages.push({file, base64, objectUrl});
        renderThumbnails();
    };
    reader.readAsDataURL(file);
}

function removeImage(index) {
    const removed = pendingImages.splice(index, 1);
    if (removed.length > 0) URL.revokeObjectURL(removed[0].objectUrl);
    renderThumbnails();
}

function clearImages() {
    pendingImages.forEach(function(img) { URL.revokeObjectURL(img.objectUrl); });
    pendingImages.length = 0;
    renderThumbnails();
}

function renderThumbnails() {
    const strip = document.getElementById('thumbnail-strip');
    if (!strip) return;
    strip.replaceChildren();
    pendingImages.forEach(function(img, idx) {
        var item = document.createElement('div');
        item.className = 'thumbnail-item';

        var imgEl = document.createElement('img');
        imgEl.className = 'thumbnail-img';
        imgEl.src = img.objectUrl;
        imgEl.alt = 'Attached image';
        item.appendChild(imgEl);

        var btn = document.createElement('button');
        btn.className = 'thumbnail-remove';
        btn.textContent = 'x';
        btn.setAttribute('data-idx', idx);
        btn.addEventListener('click', function() { removeImage(idx); });
        item.appendChild(btn);

        strip.appendChild(item);
    });
}

// === Workspace State ===

let activeWorkspaces = [];

function initWorkspaces() {
    const chips = document.querySelectorAll('#workspace-chips .ws-chip');
    activeWorkspaces = [];
    chips.forEach(function(chip) {
        var ws = chip.getAttribute('data-ws');
        if (ws) activeWorkspaces.push(ws);
    });
    if (activeWorkspaces.length === 0) activeWorkspaces = ['default'];
}

function toggleWorkspace(ws) {
    var idx = activeWorkspaces.indexOf(ws);
    if (idx >= 0) {
        if (activeWorkspaces.length <= 1) return;
        activeWorkspaces.splice(idx, 1);
    } else {
        activeWorkspaces.push(ws);
    }
    renderWorkspaceChips();
    updateWorkspacePanelCheckboxes();
}

function renderWorkspaceChips() {
    var container = document.getElementById('workspace-chips');
    if (!container) return;

    var addBtn = document.getElementById('ws-add-btn');

    container.replaceChildren();
    activeWorkspaces.forEach(function(ws) {
        var chip = document.createElement('span');
        chip.className = 'ws-chip';
        chip.setAttribute('data-ws', ws);
        chip.textContent = ws;

        if (activeWorkspaces.length > 1) {
            var removeBtn = document.createElement('button');
            removeBtn.className = 'ws-chip-remove';
            removeBtn.textContent = 'x';
            removeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                toggleWorkspace(ws);
            });
            chip.appendChild(removeBtn);
        }
        container.appendChild(chip);
    });

    if (addBtn) container.appendChild(addBtn);
}

function updateWorkspacePanelCheckboxes() {
    var items = document.querySelectorAll('#panel-content .workspace-check-item');
    items.forEach(function(item) {
        var ws = item.getAttribute('data-ws');
        var isActive = activeWorkspaces.indexOf(ws) >= 0;
        item.classList.toggle('selected', isActive);
        var checkbox = item.querySelector('.ws-checkbox');
        if (checkbox) checkbox.classList.toggle('checked', isActive);
    });
}

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

    activateChatMode();

    // Add user message with optional image thumbnails
    var userWrapper = document.createElement('div');
    userWrapper.style.marginLeft = 'auto';
    userWrapper.style.maxWidth = '85%';

    if (pendingImages.length > 0) {
        var msgImages = document.createElement('div');
        msgImages.className = 'message-images';
        pendingImages.forEach(function(img) {
            var imgEl = document.createElement('img');
            imgEl.className = 'message-img';
            imgEl.src = img.objectUrl;
            imgEl.alt = 'Attached image';
            msgImages.appendChild(imgEl);
        });
        userWrapper.appendChild(msgImages);
    }

    var userDiv = document.createElement('div');
    userDiv.className = 'user-message';
    userDiv.textContent = query;
    userWrapper.appendChild(userDiv);
    chatMessages.appendChild(userWrapper);

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

    const historyWindow = getHistoryWindow();

    // Capture image data before clearing
    var imageData = pendingImages.map(function(img) { return img.base64; });
    clearImages();

    let fullAnswer = '';
    let success = false;

    try {
        const response = await fetch('/web/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                images: imageData,
                workspaces: activeWorkspaces,
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

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

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
        closePanel();
        submitQuery(query);
    });

    // Files button opens panel
    const filesBtn = document.getElementById('files-btn');
    if (filesBtn) {
        filesBtn.addEventListener('htmx:afterRequest', function () {
            openPanel('FILES');
        });
    }

    // Image upload: + button
    var plusBtn = document.getElementById('composer-plus');
    var imageInput = document.getElementById('image-input');
    if (plusBtn && imageInput) {
        plusBtn.addEventListener('click', function() {
            imageInput.click();
        });
        imageInput.addEventListener('change', function() {
            var files = Array.from(imageInput.files || []);
            files.forEach(function(f) { addImage(f); });
            imageInput.value = '';
        });
    }

    // Drag and drop
    var dropOverlay = document.getElementById('drop-overlay');
    var dragCounter = 0;

    document.addEventListener('dragenter', function(e) {
        e.preventDefault();
        if (e.dataTransfer && e.dataTransfer.types.indexOf('Files') >= 0) {
            dragCounter++;
            if (dropOverlay) dropOverlay.classList.add('active');
        }
    });

    document.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            if (dropOverlay) dropOverlay.classList.remove('active');
        }
    });

    document.addEventListener('dragover', function(e) {
        e.preventDefault();
    });

    document.addEventListener('drop', function(e) {
        e.preventDefault();
        dragCounter = 0;
        if (dropOverlay) dropOverlay.classList.remove('active');
        var files = Array.from(e.dataTransfer.files || []);
        files.forEach(function(f) {
            if (f.type.startsWith('image/')) addImage(f);
        });
    });

    // Clipboard paste
    document.addEventListener('paste', function(e) {
        var items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.startsWith('image/')) {
                var file = items[i].getAsFile();
                if (file) addImage(file);
            }
        }
    });

    // Workspace: + workspace button opens panel
    var wsAddBtn = document.getElementById('ws-add-btn');
    if (wsAddBtn) {
        wsAddBtn.addEventListener('htmx:afterRequest', function() {
            openPanel('WORKSPACES');
            updateWorkspacePanelCheckboxes();
        });
    }

    // Panel auto-close on chat area click
    var chatArea = document.getElementById('chat-area');
    if (chatArea) {
        chatArea.addEventListener('click', function() {
            closePanel();
        });
    }

    // Panel auto-close on Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closePanel();
    });

    // Initialize workspace state
    initWorkspaces();
});
