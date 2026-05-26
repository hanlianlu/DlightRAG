// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {activeWorkspaces, appendExchange, getHistoryWindow, getSessionId} from './state.js';
import {clearImages, getPendingImageData} from './images.js';
import {closePanel} from './panel.js';
import {streamSSE} from './sse.js';
import {createAnswerRenderer, createChatTurn, setAnswerError} from './chat_renderer.js';

let queryInFlight = false;

export async function submitQuery(query) {
    if (queryInFlight) return;
    queryInFlight = true;

    const turn = createChatTurn(query);
    const imageData = getPendingImageData();
    clearImages();

    let renderer = null;
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
            setAnswerError(turn, 'Service error. Please try again.');
            return;
        }

        renderer = createAnswerRenderer(turn);
        await streamSSE(response, function(eventType, data) {
            renderer.handle(eventType, data);
        });
        success = !renderer.failed;
    } catch (_) {
        setAnswerError(turn, 'Connection error. Please try again.');
    } finally {
        queryInFlight = false;
    }

    if (success && renderer && renderer.answer) {
        appendExchange(query, renderer.answer, renderer.imageIds);
    }
}

export function setupQueryForm() {
    const form = document.getElementById('query-form');
    if (!form) return;
    const textarea = form.querySelector('.composer-input');

    function autoResize() {
        var lineHeight = parseInt(getComputedStyle(textarea).lineHeight, 10) || 24;
        var rows = Math.floor(textarea.scrollHeight / lineHeight);
        form.classList.toggle('multiline', rows > 1);

        textarea.style.height = 'auto';
        var maxHeight = 160;
        textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + 'px';
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
        textarea.style.height = '';
        form.classList.remove('multiline');
        closePanel();
        submitQuery(query);
    });
}
