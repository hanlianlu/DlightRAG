// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {activeWorkspaces, appendExchange, getHistoryWindow, getSessionId, conversationHistory} from './state.js';
import {clearImages, getPendingImageData, renderMessageImages} from './images.js';
import {closePanel} from './panel.js';

let queryInFlight = false;

function activateChatMode() {
    const app = document.querySelector('.app');
    if (app && !app.classList.contains('has-messages')) app.classList.add('has-messages');
}

function markOutOfContext(historyKept) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    const outOfContextCount = conversationHistory.length - historyKept;
    if (outOfContextCount <= 0) {
        chatMessages.querySelectorAll('.out-of-context').forEach(function(el) {
            el.classList.remove('out-of-context');
        });
        const old = chatMessages.querySelector('.context-divider');
        if (old) old.remove();
        return;
    }
    let msgIndex = 0;
    let dividerInsertBefore = null;
    Array.from(chatMessages.children).forEach(function(node) {
        if (node.classList.contains('context-divider')) return;
        if (msgIndex < outOfContextCount) {
            node.classList.add('out-of-context');
        } else {
            node.classList.remove('out-of-context');
            if (dividerInsertBefore === null) dividerInsertBefore = node;
        }
        msgIndex++;
    });
    const existing = chatMessages.querySelector('.context-divider');
    if (existing) existing.remove();
    if (dividerInsertBefore) {
        const divider = document.createElement('div');
        divider.className = 'context-divider';
        divider.textContent = 'Above messages are outside AI memory';
        chatMessages.insertBefore(divider, dividerInsertBefore);
    }
}

export async function submitQuery(query) {
    if (queryInFlight) return;
    queryInFlight = true;
    const chatMessages = document.getElementById('chat-messages');
    const chatArea = document.getElementById('chat-area');

    activateChatMode();

    const userWrapper = document.createElement('div');
    userWrapper.className = 'user-message-wrapper';
    renderMessageImages(userWrapper);
    const userDiv = document.createElement('div');
    userDiv.className = 'user-message';
    userDiv.textContent = query;
    userWrapper.appendChild(userDiv);
    chatMessages.appendChild(userWrapper);

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

    const imageData = getPendingImageData();
    clearImages();

    let fullAnswer = '';
    let imageIds = [];
    let success = false;

    try {
        const response = await fetch('/web/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                images: imageData,
                workspaces: activeWorkspaces,
                conversation_history: getHistoryWindow(),
                session_id: getSessionId(),
            }),
        });

        if (!response.ok) {
            contentDiv.textContent = 'Service error. Please try again.';
            contentDiv.classList.add('text-error');
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let firstToken = true;
        let sseEvent = '';
        let sseDataParts = [];

        while (true) {
            const result = await reader.read();
            if (result.done) break;
            buffer += decoder.decode(result.value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (line === '') {
                    if (sseDataParts.length > 0) {
                        const data = sseDataParts.join('\n');
                        if (sseEvent === 'token') {
                            try { fullAnswer += JSON.parse(data); } catch (_) { fullAnswer += data; }
                        }
                        const handled = handleSSEData(sseEvent, data, contentDiv, aiDiv, chatArea, firstToken);
                        if (handled && handled.finalAnswer) fullAnswer = handled.finalAnswer;
                        if (handled && handled.imageIds) imageIds = handled.imageIds;
                        if (sseEvent === 'token' && firstToken) firstToken = false;
                    }
                    sseEvent = '';
                    sseDataParts = [];
                } else if (line.startsWith('event: ')) {
                    sseEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    sseDataParts.push(line.slice(6));
                } else if (line.startsWith('data:')) {
                    sseDataParts.push(line.slice(5));
                }
            }
        }
        success = true;
    } catch (_) {
        contentDiv.textContent = 'Connection error. Please try again.';
        contentDiv.classList.add('text-error');
    } finally {
        queryInFlight = false;
    }

    if (success && fullAnswer) appendExchange(query, fullAnswer, imageIds);
}

function handleSSEData(eventType, data, contentDiv, aiDiv, chatArea, firstToken) {
    if (eventType === 'token') {
        if (firstToken) contentDiv.textContent = '';
        let text;
        try { text = JSON.parse(data); } catch (_) { text = data; }
        const span = document.createElement('span');
        span.textContent = text;
        contentDiv.appendChild(span);
        chatArea.scrollTop = chatArea.scrollHeight;
    } else if (eventType === 'preview') {
        let previewHtml;
        try { previewHtml = JSON.parse(data); } catch (_) { previewHtml = data; }
        contentDiv.innerHTML = previewHtml;
        chatArea.scrollTop = chatArea.scrollHeight;
    } else if (eventType === 'done') {
        let html;
        let finalAnswer = null;
        let imageIds = [];
        try {
            const payload = JSON.parse(data);
            html = typeof payload === 'string' ? payload : payload.html;
            finalAnswer = typeof payload === 'string' ? null : payload.answer || null;
            imageIds = typeof payload === 'string' ? [] : payload.current_image_ids || [];
        } catch (_) {
            html = data;
        }
        const tmp = document.createElement('div');
        tmp.innerHTML = html;
        const answerContent = tmp.querySelector('#answer-content');
        const sourceData = tmp.querySelector('#source-data');
        if (answerContent) contentDiv.innerHTML = answerContent.innerHTML;
        if (sourceData) {
            sourceData.className = 'source-data visually-hidden';
            sourceData.removeAttribute('id');
            aiDiv.appendChild(sourceData);
        }
        if (window.renderMathInElement) {
            renderMathInElement(contentDiv, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                ],
                throwOnError: false,
            });
        }
        return {finalAnswer: finalAnswer, imageIds: imageIds};
    } else if (eventType === 'highlights') {
        let highlightsHtml;
        try { highlightsHtml = JSON.parse(data); } catch (_) { highlightsHtml = data; }
        const sourceData = aiDiv.querySelector('.source-data');
        if (sourceData) sourceData.innerHTML = highlightsHtml;
    } else if (eventType === 'meta') {
        try {
            const meta = JSON.parse(data);
            if (typeof meta.history_kept === 'number') markOutOfContext(meta.history_kept);
        } catch (_) { /* ignore */ }
    } else if (eventType === 'progress') {
        try {
            const info = JSON.parse(data);
            const labels = {
                planning: 'Analyzing query…',
                searching: 'Searching knowledge base…',
                generating: 'Generating answer…',
            };
            const label = labels[info.phase] || info.phase;
            const dot = contentDiv.querySelector('.streaming-dot');
            if (dot) {
                dot.textContent = '';
                dot.className = 'streaming-dot progress-phase';
                dot.setAttribute('data-phase', label);
            }
        } catch (_) { /* ignore */ }
    } else if (eventType === 'error') {
        contentDiv.textContent = data;
        contentDiv.classList.add('text-error');
    }
    return null;
}

export function setupQueryForm() {
    const form = document.getElementById('query-form');
    if (!form) return;
    const textarea = form.querySelector('.composer-input');
    function autoResize() {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
    }
    if (textarea) {
        textarea.addEventListener('input', autoResize);
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    }
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const query = textarea.value.trim();
        if (!query) return;
        textarea.value = '';
        textarea.style.height = 'auto';
        closePanel();
        submitQuery(query);
    });
}
