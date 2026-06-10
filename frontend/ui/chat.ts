// @ts-nocheck — full types deferred per spec
// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {workspaceStore} from '../stores/workspaceStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {sessionStore} from '../stores/sessionStore.ts';
import {clearImages, getPendingImageData} from './images.ts';
import {closePanel} from './panel.ts';
import {streamSSE} from '../lib/sse.ts';
import {createAnswerRenderer, createChatTurn, setAnswerError} from '../lib/chat_renderer.ts';

let queryInFlight = false;

function submitComposerForm(form) {
    form.requestSubmit();
}

function isLineBreakInput(e) {
    return e.inputType === 'insertLineBreak';
}

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
                workspaces: workspaceStore.active,
                conversation_history: conversationStore.historyWindow,
                session_id: sessionStore.sessionId,
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
        conversationStore.append(
            query,
            renderer.answer,
            renderer.imageIds,
            imageData.length > 0 ? imageData : undefined,
        );
    }
}

export function setupQueryForm() {
    const form = document.getElementById('query-form');
    if (!form) return;
    const textarea = form.querySelector('.composer-input');
    const sendBtn = form.querySelector('.composer-send') as HTMLButtonElement | null;

    function toggleSendButton() {
        if (sendBtn && textarea) {
            sendBtn.disabled = !textarea.value.trim();
        }
    }

    function autoResize() {
        toggleSendButton();
        const computed = getComputedStyle(textarea);
        const lineHeight = parseFloat(computed.lineHeight) || 24;
        const maxHeight = parseFloat(computed.maxHeight) || 160;
        textarea.style.height = 'auto';
        const contentHeight = textarea.scrollHeight;
        const isMultiline = textarea.value.includes('\n') || contentHeight > lineHeight * 1.5;
        form.classList.toggle('multiline', isMultiline);
        textarea.style.height = Math.min(contentHeight, maxHeight) + 'px';
        textarea.style.overflowY = contentHeight > maxHeight ? 'auto' : 'hidden';
    }

    if (textarea) {
        let allowNextLineBreak = false;

        textarea.addEventListener('keydown', function(e) {
            if (e.key !== 'Enter') return;
            allowNextLineBreak = e.shiftKey === true;
        });
        textarea.addEventListener('beforeinput', function(e) {
            if (!isLineBreakInput(e)) return;
            if (e.isComposing === true || allowNextLineBreak) {
                allowNextLineBreak = false;
                return;
            }
            e.preventDefault();
            allowNextLineBreak = false;
            submitComposerForm(form);
        });
        textarea.addEventListener('keyup', function(e) {
            if (e.key === 'Enter') allowNextLineBreak = false;
        });
        textarea.addEventListener('input', autoResize);

        // Initial state: send button disabled until user types
        toggleSendButton();
    }
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const query = textarea.value.trim();
        if (!query) return;
        textarea.value = '';
        textarea.style.height = '';
        textarea.style.overflowY = '';
        form.classList.remove('multiline');
        toggleSendButton();
        closePanel();
        submitQuery(query);
    });
}
