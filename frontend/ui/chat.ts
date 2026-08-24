// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from '../api/csrf';
import {undoMemoryChange} from '../api/memory.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {clearAttachments, getPendingAttachments} from './attachments.ts';
import {createSSEParser} from '../lib/sse.ts';
import {buildAnswerRequest} from '../lib/answer_request.ts';
import {readStoredAnswerMode} from './answer_mode.ts';
import {
    cancelAnswerRun,
    continueAnswerRun,
    getAnswerRun,
    getAnswerRunChildren,
    steerAnswerRun,
} from '../api/conversations.ts';
import type {
    AnswerRunDescriptor,
    ConversationAttachmentReference,
    ConversationTurn,
} from '../api/conversations.ts';
import {
    createAnswerRenderer,
    createChatTurn,
    markAnswerPending,
    renderStoredTurn,
    setAnswerError,
    setAnswerRetryable,
} from '../lib/chat_renderer.ts';
import chatStyles from '../styles/chat.module.css';
import type {ChatTurn} from '../lib/chat_renderer.ts';
import {answerRunStore, payloadFingerprint} from '../stores/answerRunStore.ts';
import {chatSessionStore} from '../stores/chatSessionStore.ts';
import {conversationRoute} from '../lib/router.ts';
import {webRouter} from './router.ts';
import {refreshMemorySettingsPanel} from './memory.ts';
import {showActionToast, showToast} from './toast.ts';

// A dropped connection is a transport fault, never a decision about the run, so
// the browser reattaches from its last durable sequence. The budget bounds
// consecutive attempts that consume nothing, not reconnects that make progress.
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 500;
const NEW_CHAT_RUN_KEY = '__new_chat__';
const seenMemoryOperations = new Set<string>();

interface MemoryOperationEvent {
    body?: string;
    change_id?: string | null;
    intent_id?: string;
    kind?: string | null;
    live?: boolean;
    operation?: 'remember' | 'forget' | 'undo';
    outcome?: 'changed' | 'unchanged' | 'rejected' | 'conflict';
}

function memorySummary(event: MemoryOperationEvent): string {
    const body = String(event.body || '').replace(/\s+/g, ' ').trim();
    const concise = body.length > 120 ? body.slice(0, 117) + '…' : body;
    if (event.outcome === 'unchanged') return 'Already remembered.';
    if (event.outcome === 'conflict') return 'Profile Memory changed; recall it before retrying.';
    if (event.outcome === 'rejected') return 'Profile Memory operation was rejected.';
    if (event.operation === 'forget') return concise ? `Forgot: ${concise}` : 'Profile Memory forgotten.';
    if (event.operation === 'undo') return concise ? `Restored: ${concise}` : 'Profile Memory restored.';
    return concise ? `Remembered: ${concise}` : 'Saved to Profile Memory.';
}

function handleMemoryOperation(data: string): void {
    let event: MemoryOperationEvent;
    try {
        event = JSON.parse(data) as MemoryOperationEvent;
    } catch {
        return;
    }
    if (!event.live) return;
    const identity = event.change_id || `${event.intent_id || ''}:${event.operation}:${event.outcome}`;
    if (!identity || seenMemoryOperations.has(identity)) return;
    seenMemoryOperations.add(identity);
    const message = memorySummary(event);
    if (event.outcome !== 'changed' || !event.change_id) {
        showToast(message, 5000);
        return;
    }
    const changeId = event.change_id;
    showActionToast(message, {
        actionLabel: 'Undo',
        duration: 12_000,
        onAction: async () => {
            const receipt = await undoMemoryChange(changeId);
            if (receipt.outcome !== 'changed') throw new Error('Memory undo conflicted');
            void refreshMemorySettingsPanel().catch(() => {});
            return 'Profile Memory change undone.';
        },
    });
    void refreshMemorySettingsPanel().catch(() => {});
}

// The POST that turns one submission into a durable run. Only this window may
// block conversation navigation; following the accepted run never does.
let submissionPending = false;
let queryInFlight = false;
let queryStopping = false;
// Detaches this tab's event reader. It never cancels the run: only the explicit
// Stop action does that, through the durable cancel endpoint.
let currentFollowController: AbortController | null = null;
let currentRunId: string | null = null;

/** Ask the server to stop the run this tab is following. */
export function cancelQuery(): void {
    const runId = currentRunId;
    if (!runId || queryStopping) return;
    queryStopping = true;
    void cancelAnswerRun(runId).catch(function() {
        queryStopping = false;
    });
}

/** True only until the server has accepted the submission as a durable run. */
export function isSubmissionPending(): boolean {
    return submissionPending;
}

/** Stop reading this tab's run events. The run itself keeps producing. */
export function detachAnswerRun(): void {
    const controller = currentFollowController;
    if (!controller) return;
    // Release the composer now rather than when the reader finally unwinds, so
    // the conversation being opened can immediately reattach to its own run.
    currentFollowController = null;
    currentRunId = null;
    endSubmission();
    controller.abort();
}

function submitComposerForm(form: HTMLFormElement): void {
    form.requestSubmit();
}

function isLineBreakInput(e: InputEvent): boolean {
    return e.inputType === 'insertLineBreak';
}

/** Show a steering instruction as a plain user message in the live thread. */
function appendUserMessage(text: string): void {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    const wrapper = document.createElement('div');
    wrapper.className = chatStyles.userMessageWrapper;
    wrapper.dataset.steer = 'true';
    const bubble = document.createElement('div');
    bubble.className = chatStyles.userMessage;
    bubble.textContent = text;
    wrapper.appendChild(bubble);
    chatMessages.appendChild(wrapper);
    const chatArea = document.getElementById('chat-area');
    if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
}

function sleep(ms: number): Promise<void> {
    return new Promise(function(resolve) { setTimeout(resolve, ms); });
}

/**
 * Follow one durable run until it reaches a terminal event.
 *
 * Reconnects resume after the last durable sequence this tab consumed, so no
 * event is replayed twice and none is skipped. Aborting only detaches, and the
 * returned flag reports whether the run actually finished.
 */
async function followAnswerRun(
    turn: ChatTurn,
    conversationId: string,
    runId: string,
    controller: AbortController,
): Promise<boolean> {
    const renderer = createAnswerRenderer(turn);
    let barren = 0;
    // Only a connection that consumed a new durable sequence counts as progress:
    // a long answer may legitimately outlive any number of dropped connections.
    function mayRetry(before: number): boolean {
        barren = answerRunStore.lastSequence(conversationId, runId) > before ? 0 : barren + 1;
        return barren <= MAX_RECONNECT_ATTEMPTS;
    }

    while (!controller.signal.aborted) {
        const after = answerRunStore.lastSequence(conversationId, runId);
        let response: Response;
        try {
            response = await fetch(`/web/api/answer/${encodeURIComponent(runId)}/events`, {
                signal: controller.signal,
                headers: after > 0 ? {'Last-Event-ID': String(after)} : undefined,
            });
        } catch (_) {
            if (controller.signal.aborted) return false;
            if (!mayRetry(after)) break;
            await sleep(RECONNECT_DELAY_MS);
            continue;
        }
        if (response.status === 410 || response.status === 404) break;
        if (!response.ok) {
            setAnswerError(turn, 'Service error. Please try again.');
            return true;
        }
        try {
            await readAnswerEvents(response, conversationId, runId, renderer);
        } catch (_) {
            if (controller.signal.aborted) return false;
            if (!mayRetry(after)) break;
            await sleep(RECONNECT_DELAY_MS);
            continue;
        }
        if (renderer.terminal) return true;
        if (!mayRetry(after)) break;
        await sleep(RECONNECT_DELAY_MS);
    }
    if (controller.signal.aborted) return false;
    return await settleFromStoredRun(turn, conversationId, runId);
}

/**
 * The event log is gone or unreachable, so the run row stays authoritative.
 *
 * A run that is still producing is a connection failure this tab can recover
 * from, so it offers an explicit reattach rather than a spinner nothing feeds.
 */
async function settleFromStoredRun(
    turn: ChatTurn,
    conversationId: string,
    runId: string,
): Promise<boolean> {
    let stored: ConversationTurn;
    try {
        stored = await getAnswerRun(runId);
    } catch (_) {
        setAnswerError(turn, 'Service error. Please try again.');
        return true;
    }
    if (stored.status === 'queued' || stored.status === 'running') {
        setAnswerRetryable(turn, 'Connection lost. This answer is still running.', function() {
            void resumePendingTurn(turn, conversationId, stored);
        });
        return false;
    }
    renderStoredTurn(turn, stored);
    return true;
}

async function readAnswerEvents(
    response: Response,
    conversationId: string,
    runId: string,
    renderer: ReturnType<typeof createAnswerRenderer>,
): Promise<void> {
    if (!response.body) throw new Error('Response body is not streamable');
    const parser = createSSEParser(function(eventType, data, id) {
        const sequence = Number(id);
        if (Number.isFinite(sequence) && sequence > 0) {
            if (sequence <= answerRunStore.lastSequence(conversationId, runId)) return;
            answerRunStore.recordSequence(conversationId, runId, sequence);
        }
        if (conversationStore.activeConversationId !== conversationId) return;
        if (eventType === 'memory_operation_settled') handleMemoryOperation(data);
        renderer.handle(eventType, data);
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    try {
        while (true) {
            const result = await reader.read();
            if (result.done) break;
            parser.push(decoder.decode(result.value, {stream: true}));
        }
        parser.push(decoder.decode());
        parser.flush();
    } finally {
        // Release the underlying connection on any exit (completion, error, abort).
        reader.cancel().catch(() => {});
    }
}

/** Reattach to a pending turn discovered from conversation history. */
export async function resumePendingTurn(
    turn: ChatTurn,
    conversationId: string,
    stored: ConversationTurn,
): Promise<void> {
    if (queryInFlight) return;
    turn.aiDiv.dataset.runId = stored.answer_run_id;
    answerRunStore.trackRun(conversationId, stored.answer_run_id);
    const controller = new AbortController();
    beginFollowing(controller, stored.answer_run_id, stored.cancel_requested);
    let finished = false;
    try {
        finished = await followAnswerRun(turn, conversationId, stored.answer_run_id, controller);
    } catch (_) {
        setAnswerError(turn, 'Connection error. Please try again.');
        finished = true;
    } finally {
        endFollowing(controller, conversationId, finished);
    }
}

function beginFollowing(
    controller: AbortController,
    runId: string,
    cancelRequested: boolean,
): void {
    submissionPending = false;
    queryInFlight = true;
    queryStopping = cancelRequested;
    currentFollowController = controller;
    currentRunId = runId;
    chatSessionStore.setActive(true);
}

/** Release the composer and its Stop affordance; this tab follows nothing now. */
function endSubmission(): void {
    submissionPending = false;
    queryInFlight = false;
    queryStopping = false;
    chatSessionStore.setActive(false);
}

function endFollowing(
    controller: AbortController,
    conversationId: string,
    finished: boolean,
): void {
    if (currentFollowController !== controller) return;
    currentFollowController = null;
    currentRunId = null;
    endSubmission();
    if (!finished) return;
    // A finished run retires its submission key, so asking the same question
    // again is a new run rather than a replay of this one.
    answerRunStore.clear(conversationId);
    void conversationStore.refreshActive();
}

export async function submitQuery(query: string): Promise<void> {
    if (queryInFlight) return;
    queryInFlight = true;
    submissionPending = true;
    chatSessionStore.setActive(true);

    let conversationId = conversationStore.answerConversationId;

    const controller = new AbortController();
    let following = false;
    let finished = false;

    const pendingAttachments = getPendingAttachments();
    // Render the just-submitted attachments in the user bubble from fresh object
    // URLs so clearing the composer strip (which revokes its own URLs) cannot
    // blank the sent turn. History reload later re-renders from server URLs.
    const liveAttachmentRefs: ConversationAttachmentReference[] = pendingAttachments.map(
        function(item, index) {
            const previewUrl = URL.createObjectURL(item.file);
            return {
                attachment_id: item.id,
                ordinal: index + 1,
                kind: item.kind,
                filename: item.file.name,
                mime_type: item.file.type,
                byte_size: item.file.size,
                url: previewUrl,
                thumbnail_url: previewUrl,
                label: item.file.name,
            };
        },
    );
    const turn = createChatTurn(query, liveAttachmentRefs);
    const attachmentFiles = pendingAttachments.map(function(item) { return item.file; });

    try {
        if (!conversationStore.canAnswer) {
            setAnswerError(
                turn,
                'Conversation service is unavailable. Please retry loading the conversation.',
            );
            return;
        }
        const activeWorkspaces = [...workspaceStore.active];
        const fingerprint = await payloadFingerprint({
            query,
            attachments: pendingAttachments.map(function(item) {
                return {name: item.file.name, size: item.file.size, type: item.file.type};
            }),
            workspaces: activeWorkspaces,
        });
        if (conversationStore.answerConversationId !== conversationId) {
            setAnswerError(turn, 'The active conversation changed before this answer started.');
            return;
        }
        const runKey = conversationId ?? NEW_CHAT_RUN_KEY;
        const submissionId = answerRunStore.getOrCreateSubmissionId(runKey, fingerprint);
        const {body: requestBody, headers: requestHeaders} = buildAnswerRequest(
            {
                query,
                workspaces: activeWorkspaces,
                conversationId,
                submissionId,
                ...(readStoredAnswerMode() ? {mode: readStoredAnswerMode()!} : {}),
            },
            attachmentFiles,
        );
        clearAttachments();
        const response = await fetch('/web/api/answer', {
            method: 'POST',
            headers: {...csrfHeaders(), ...(requestHeaders ?? {})},
            body: requestBody,
        });

        if (!response.ok) {
            if (response.status < 500) answerRunStore.clear(runKey);
            setAnswerError(turn, 'Service error. Please try again.');
            return;
        }
        const descriptor = await response.json() as AnswerRunDescriptor;
        const acceptedConversationId = descriptor.conversation.conversation_id;
        if (conversationId && acceptedConversationId !== conversationId) {
            answerRunStore.clear(runKey);
            setAnswerError(turn, 'The answer was accepted for an unexpected conversation.');
            return;
        }
        answerRunStore.attachRun(runKey, descriptor.run_id);
        if (!conversationId) {
            answerRunStore.transfer(runKey, acceptedConversationId);
            conversationStore.adoptCreatedConversation(descriptor.conversation);
            conversationId = acceptedConversationId;
            await webRouter.navigate(conversationRoute(conversationId), {
                replace: true,
                notify: false,
                bypassGuard: true,
            });
        } else {
            conversationStore.upsertSummary(descriptor.conversation);
        }
        turn.aiDiv.dataset.runId = descriptor.run_id;
        markAnswerPending(turn, descriptor.cancel_requested);
        beginFollowing(controller, descriptor.run_id, descriptor.cancel_requested);
        following = true;
        finished = await followAnswerRun(turn, conversationId, descriptor.run_id, controller);
    } catch (_) {
        if (!controller.signal.aborted) {
            setAnswerError(turn, 'Connection error. Please try again.');
            finished = true;
        }
    } finally {
        if (following && conversationId) endFollowing(controller, conversationId, finished);
        else endSubmission();
    }
}

async function handleRunAction(action: string, runId: string): Promise<void> {
    if (!runId) return;
    if (action === 'children') {
        const roster = document.querySelector('dl-children-roster');
        roster?.open(async () => {
            try {
                return await getAnswerRunChildren(runId);
            } catch (_) {
                return [];
            }
        });
        return;
    }
    if (action !== 'follow-up' && action !== 'fork') return;
    const dialog = document.querySelector('dl-continuation-dialog');
    if (!dialog) return;
    await new Promise<void>((resolve) => {
        dialog.addEventListener('dl-continuation-result', function handler(event: Event) {
            dialog.removeEventListener('dl-continuation-result', handler);
            const result = (event as CustomEvent).detail;
            if (result.query) void startContinuation(action, runId, String(result.query));
            resolve();
        });
        dialog.open(action === 'fork' ? 'fork' : 'follow-up');
    });
}

async function startContinuation(action: string, runId: string, query: string): Promise<void> {
    try {
        const descriptor = await continueAnswerRun(
            runId,
            action === 'fork' ? 'fork' : 'follow-up',
            query,
            crypto.randomUUID(),
        );
        conversationStore.upsertSummary(descriptor.conversation);
        if (action === 'fork') {
            await webRouter.navigate(conversationRoute(descriptor.conversation.conversation_id));
        } else {
            await conversationStore.open(descriptor.conversation.conversation_id, {
                showLoading: false,
                preserveOnError: true,
            });
        }
    } catch (_) {
        window.alert('The continuation could not be started.');
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

    document.addEventListener('answer-run-action', function(event: Event) {
        const detail = (event as CustomEvent<{action?: string; runId?: string}>).detail;
        void handleRunAction(String(detail?.action || ''), String(detail?.runId || ''));
    });

    function toggleSendButton() {
        if (!sendBtn) return;
        // Three states: Send (idle), Steer (running + text), Stop (running + empty).
        const hasText = Boolean(queryInput.value.trim());
        sendBtn.disabled = !hasText && !queryInFlight;
        sendBtn.classList.toggle('is-stop', queryInFlight && !hasText);
        sendBtn.classList.toggle('is-steer', queryInFlight && hasText);
        sendBtn.setAttribute('aria-label', queryInFlight ? (hasText ? 'Steer' : 'Stop') : 'Send');
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

    async function submitSteer(): Promise<void> {
        const text = queryInput.value.trim();
        const runId = currentRunId;
        if (!text || !runId) return;
        try {
            await steerAnswerRun(runId, text);
        } catch (_) {
            window.alert('This run can no longer be steered.');
            return;
        }
        appendUserMessage(text);
        queryInput.value = '';
        autoResize();
        queryInput.focus();
    }

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
        // While the agent is working, a submitted message steers the live run
        // instead of starting a second answer (Pi-style safe-point injection).
        if (queryInFlight) {
            void submitSteer();
            return;
        }
        submitComposerForm(queryForm);
    });
    queryInput.addEventListener('keyup', function(e: KeyboardEvent) {
        if (e.key === 'Enter') allowNextLineBreak = false;
    });
    queryInput.addEventListener('input', autoResize);

    toggleSendButton();

    // Send ⇄ Steer ⇄ Stop: with text while running it steers; empty while
    // running it cancels; otherwise it submits a new answer.
    sendBtn?.addEventListener('click', function(e) {
        if (!queryInFlight) return;
        e.preventDefault();
        if (queryInput.value.trim()) {
            void submitSteer();
        } else {
            cancelQuery();
        }
    });
    chatSessionStore.subscribe(function() {
        toggleSendButton();
    });

    queryForm.addEventListener('submit', function(e: SubmitEvent) {
        e.preventDefault();
        if (queryInFlight) {
            void submitSteer();  // Enter during a live run steers it
            return;
        }
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
