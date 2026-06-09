// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {conversationStore} from '../stores/conversationStore.ts';
import {renderMessageImages} from '../ui/images.ts';
import {renderMath} from '../ui/mathjax.ts';
import {parseData} from './sse.ts';
import chatStyles from '../styles/chat.module.css';

// Security: all SSE HTML payloads are sanitized server-side by nh3
// (src/dlightrag/web/safe_html.py → sanitize_html_fragment()) before they
// reach any innerHTML below.  The sanitizer strips <script>, <style>,
// <iframe>, on* attributes, and restricts URLs to http/https/data:image/.
// If you add a new innerHTML call, ensure the data flows through
// safe_answer_preview / safe_answer_done / safe_source_panel.

function activateChatMode() {
    const app = document.querySelector('.app');
    if (app && !app.classList.contains('has-messages')) app.classList.add('has-messages');
}

function fixExternalLinks(container) {
    container.querySelectorAll('a[href]').forEach(function(a) {
        if (!a.getAttribute('target')) {
            a.setAttribute('target', '_blank');
            a.setAttribute('rel', 'noopener noreferrer');
        }
    });
}

function markOutOfContext(historyKept) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    const outOfContextCount = conversationStore.historyWindow.length - historyKept;
    if (outOfContextCount <= 0) {
        chatMessages.querySelectorAll('.' + chatStyles.outOfContext).forEach(function(el) {
            el.classList.remove(chatStyles.outOfContext);
        });
        const old = chatMessages.querySelector('.' + chatStyles.contextDivider);
        if (old) old.remove();
        return;
    }

    let msgIndex = 0;
    let dividerInsertBefore = null;
    Array.from(chatMessages.children).forEach(function(node) {
        if (node.classList.contains(chatStyles.contextDivider)) return;
        if (msgIndex < outOfContextCount) {
            node.classList.add(chatStyles.outOfContext);
        } else {
            node.classList.remove(chatStyles.outOfContext);
            if (dividerInsertBefore === null) dividerInsertBefore = node;
        }
        msgIndex++;
    });

    const existing = chatMessages.querySelector('.' + chatStyles.contextDivider);
    if (existing) existing.remove();
    if (dividerInsertBefore) {
        const divider = document.createElement('div');
        divider.className = chatStyles.contextDivider;
        divider.textContent = 'Above messages are outside AI memory';
        chatMessages.insertBefore(divider, dividerInsertBefore);
    }
}

function scrollToBottom(turn) {
    if (turn.chatArea) turn.chatArea.scrollTop = turn.chatArea.scrollHeight;
}

export function createChatTurn(query) {
    const chatMessages = document.getElementById('chat-messages');
    const chatArea = document.getElementById('chat-area');

    activateChatMode();

    const userWrapper = document.createElement('div');
    userWrapper.className = chatStyles.userMessageWrapper;
    renderMessageImages(userWrapper);

    const userDiv = document.createElement('div');
    userDiv.className = chatStyles.userMessage;
    userDiv.textContent = query;
    userWrapper.appendChild(userDiv);
    chatMessages.appendChild(userWrapper);

    const aiDiv = document.createElement('div');
    aiDiv.className = chatStyles.aiMessage;

    const headerDiv = document.createElement('div');
    headerDiv.className = chatStyles.aiMessageHeader;
    const dot = document.createElement('span');
    dot.className = chatStyles.dot;
    dot.textContent = '●';
    headerDiv.appendChild(dot);
    headerDiv.appendChild(document.createTextNode(' DlightRAG'));
    aiDiv.appendChild(headerDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = chatStyles.aiMessageContent;
    const streamingDot = document.createElement('span');
    streamingDot.className = chatStyles.streamingDot;
    contentDiv.appendChild(streamingDot);
    aiDiv.appendChild(contentDiv);

    chatMessages.appendChild(aiDiv);

    const turn = {chatArea, aiDiv, contentDiv};
    scrollToBottom(turn);
    return turn;
}

export function setAnswerError(turn, message) {
    turn.contentDiv.textContent = typeof message === 'string' ? message : 'Service error. Please try again.';
    turn.contentDiv.classList.add(chatStyles.textError);
}

export function createAnswerRenderer(turn) {
    let firstToken = true;
    let fullAnswer = '';
    let imageIds = [];
    let failed = false;

    function handleToken(data) {
        const text = parseData(data);
        const token = typeof text === 'string' ? text : String(text);
        if (firstToken) {
            turn.contentDiv.textContent = '';
            firstToken = false;
        }
        fullAnswer += token;
        const span = document.createElement('span');
        span.textContent = token;
        turn.contentDiv.appendChild(span);
        scrollToBottom(turn);
    }

    function handlePreview(data) {
        const previewHtml = parseData(data);
        turn.contentDiv.innerHTML = typeof previewHtml === 'string' ? previewHtml : '';
        renderMath(turn.contentDiv);
        fixExternalLinks(turn.contentDiv);
        scrollToBottom(turn);
    }

    function handleDone(data) {
        const payload = parseData(data);
        const html = typeof payload === 'string' ? payload : payload.html;
        if (typeof payload !== 'string') {
            fullAnswer = payload.answer || fullAnswer;
            imageIds = payload.current_image_ids || [];
        }

        const tmp = document.createElement('div');
        tmp.innerHTML = html || '';
        const answerContent = tmp.querySelector('#answer-content');
        const sourceData = tmp.querySelector('#source-data');
        const imageStrip = tmp.querySelector('.answer-image-strip');
        const refList = tmp.querySelector('.answer-references');

        if (answerContent) turn.contentDiv.innerHTML = answerContent.innerHTML;
        if (imageStrip) {
            turn.contentDiv.appendChild(imageStrip.cloneNode(true));
        }
        if (refList) {
            turn.contentDiv.appendChild(refList.cloneNode(true));
        }
        if (sourceData) {
            sourceData.className = 'source-data visually-hidden';
            sourceData.removeAttribute('id');
            turn.aiDiv.appendChild(sourceData);
        }

        renderMath(turn.contentDiv);
        fixExternalLinks(turn.contentDiv);
    }

    function handleHighlights(data) {
        const highlightsHtml = parseData(data);
        const sourceData = turn.aiDiv.querySelector('.source-data');
        if (sourceData) {
            sourceData.innerHTML = typeof highlightsHtml === 'string' ? highlightsHtml : '';
        }
    }

    function handleMeta(data) {
        const meta = parseData(data);
        if (typeof meta.history_kept === 'number') markOutOfContext(meta.history_kept);
    }

    function handleProgress(data) {
        const info = parseData(data);
        const labels = {
            planning: 'Analyzing query...',
            searching: 'Searching knowledge base...',
            generating: 'Generating answer...',
        };
        const label = labels[info.phase] || info.phase;
        const dot = turn.contentDiv.querySelector('.' + chatStyles.streamingDot);
        if (dot) {
            dot.textContent = '';
            dot.className = chatStyles.streamingDot + ' ' + chatStyles.progressPhase;
            dot.setAttribute('data-phase', label || '');
        }
    }

    return {
        handle(eventType, data) {
            if (eventType === 'token') handleToken(data);
            else if (eventType === 'preview') handlePreview(data);
            else if (eventType === 'done') handleDone(data);
            else if (eventType === 'highlights') handleHighlights(data);
            else if (eventType === 'meta') handleMeta(data);
            else if (eventType === 'progress') handleProgress(data);
            else if (eventType === 'error') {
                failed = true;
                setAnswerError(turn, parseData(data));
            }
        },
        get answer() {
            return fullAnswer;
        },
        get imageIds() {
            return imageIds;
        },
        get failed() {
            return failed;
        },
    };
}
