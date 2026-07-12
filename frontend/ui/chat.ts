// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {workspaceStore} from '../stores/workspaceStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {clearImages, getPendingImageData} from './images.ts';
import {streamSSE} from '../lib/sse.ts';
import {createAnswerRenderer, createChatTurn, setAnswerError} from '../lib/chat_renderer.ts';

let queryInFlight = false;
let currentQueryController: AbortController | null = null;
const STREAM_IDLE_TIMEOUT_MS = 120_000;

/** Abort the in-flight answer request (user stop / navigation), if any. */
export function cancelQuery(): void {
    currentQueryController?.abort(new DOMException('Query cancelled', 'AbortError'));
}

function submitComposerForm(form: HTMLFormElement): void {
    form.requestSubmit();
}

function isLineBreakInput(e: InputEvent): boolean {
    return e.inputType === 'insertLineBreak';
}

export async function submitQuery(query: string): Promise<void> {
    if (queryInFlight) return;
    queryInFlight = true;

    const controller = new AbortController();
    currentQueryController = controller;
    let idleTimer = 0;
    const armIdleTimeout = (): void => {
        clearTimeout(idleTimer);
        idleTimer = window.setTimeout(
            () => controller.abort(new DOMException('Stream idle timeout', 'TimeoutError')),
            STREAM_IDLE_TIMEOUT_MS,
        );
    };

    const turn = createChatTurn(query);
    const imageData = getPendingImageData();
    clearImages();

    try {
        armIdleTimeout();
        const conversationId = await conversationStore.ensureActive();
        const response = await fetch('/web/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            signal: controller.signal,
            body: JSON.stringify({
                query: query,
                images: imageData,
                workspaces: workspaceStore.active,
                conversation_id: conversationId,
            }),
        });

        if (!response.ok) {
            setAnswerError(turn, 'Service error. Please try again.');
            return;
        }

        const activeRenderer = createAnswerRenderer(turn);
        await streamSSE(response, function(eventType, data) {
            armIdleTimeout();
            activeRenderer.handle(eventType, data);
        });
    } catch (_) {
        setAnswerError(
            turn,
            controller.signal.aborted ? 'Answer stopped.' : 'Connection error. Please try again.',
        );
    } finally {
        clearTimeout(idleTimer);
        if (currentQueryController === controller) currentQueryController = null;
        queryInFlight = false;
    }
}

export function setupQueryForm(): void {
    const form = document.getElementById('query-form') as HTMLFormElement | null;
    if (!form) return;
    const textarea = form.querySelector<HTMLTextAreaElement>('.composer-input');
    if (!textarea) return;
    const queryForm = form;
    const queryInput = textarea;
    const sendBtn = form.querySelector('.composer-send') as HTMLButtonElement | null;

    function toggleSendButton() {
        if (sendBtn) {
            sendBtn.disabled = !queryInput.value.trim();
        }
    }

    function autoResize() {
        toggleSendButton();
        const computed = getComputedStyle(queryInput);
        const lineHeight = parseFloat(computed.lineHeight) || 24;
        const maxHeight = parseFloat(computed.maxHeight) || 160;
        queryInput.style.height = 'auto';
        const contentHeight = queryInput.scrollHeight;
        const isMultiline = queryInput.value.includes('\n') || contentHeight > lineHeight * 1.5;
        queryForm.classList.toggle('multiline', isMultiline);
        queryInput.style.height = Math.min(contentHeight, maxHeight) + 'px';
        queryInput.style.overflowY = contentHeight > maxHeight ? 'auto' : 'hidden';
    }

    let allowNextLineBreak = false;

    queryInput.addEventListener('keydown', function(e: KeyboardEvent) {
        if (e.key === 'Escape' && queryInFlight) {
            cancelQuery();
            return;
        }
        if (e.key !== 'Enter') return;
        allowNextLineBreak = e.shiftKey === true;
    });
    queryInput.addEventListener('beforeinput', function(e: InputEvent) {
        if (!isLineBreakInput(e)) return;
        if (e.isComposing === true || allowNextLineBreak) {
            allowNextLineBreak = false;
            return;
        }
        e.preventDefault();
        allowNextLineBreak = false;
        submitComposerForm(queryForm);
    });
    queryInput.addEventListener('keyup', function(e: KeyboardEvent) {
        if (e.key === 'Enter') allowNextLineBreak = false;
    });
    queryInput.addEventListener('input', autoResize);

    toggleSendButton();

    queryForm.addEventListener('submit', function(e: SubmitEvent) {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;
        queryInput.value = '';
        queryInput.style.height = '';
        queryInput.style.overflowY = '';
        queryForm.classList.remove('multiline');
        toggleSendButton();
        submitQuery(query);
    });
}
