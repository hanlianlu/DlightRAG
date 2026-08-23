// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from '../api/csrf';
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
import type {ChatTurn} from '../lib/chat_renderer.ts';
import {answerRunStore, payloadFingerprint} from '../stores/answerRunStore.ts';
import {chatSessionStore} from '../stores/chatSessionStore.ts';
import {conversationRoute} from '../lib/router.ts';
import {webRouter} from './router.ts';

// A dropped connection is a transport fault, never a decision about the run, so
// the browser reattaches from its last durable sequence. The budget bounds
// consecutive attempts that consume nothing, not reconnects that make progress.
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 500;
const NEW_CHAT_RUN_KEY = '__new_chat__';

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
        try {
            const children = await getAnswerRunChildren(runId);
            const summary = children.length
                ? children.map((child) => `${child.status}: ${child.objective || child.child_session_id}`).join('\n')
                : 'No child agents were started.';
            window.alert(summary);
        } catch (_) {
            window.alert('Child agent status is unavailable.');
        }
        return;
    }
    if (action === 'steer') {
        const instruction = window.prompt('Steering instruction for the active Research run:');
        if (!instruction?.trim()) return;
        try {
            await steerAnswerRun(runId, instruction.trim());
        } catch (_) {
            window.alert('This run can no longer be steered.');
        }
        return;
    }
    if (action !== 'follow-up' && action !== 'fork') return;
    const query = window.prompt(action === 'fork' ? 'Start a branch with:' : 'Follow up with:');
    if (!query?.trim()) return;
    try {
        const descriptor = await continueAnswerRun(
            runId,
            action,
            query.trim(),
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
        // While an answer streams the button acts as Stop and stays actionable;
        // otherwise it is Send, enabled only when there is text to send.
        sendBtn.disabled = queryInFlight ? false : !queryInput.value.trim();
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
        if (queryInFlight) return;  // one answer at a time — keep the draft
        submitComposerForm(queryForm);
    });
    queryInput.addEventListener('keyup', function(e: KeyboardEvent) {
        if (e.key === 'Enter') allowNextLineBreak = false;
    });
    queryInput.addEventListener('input', autoResize);

    toggleSendButton();

    // Send ⇄ Stop: while an answer streams the button stops it; otherwise it
    // submits. Enter is gated separately so drafting a follow-up never stops
    // the current answer by accident.
    sendBtn?.addEventListener('click', function(e) {
        if (!queryInFlight) return;
        e.preventDefault();
        cancelQuery();
    });
    chatSessionStore.subscribe(function() {
        if (!sendBtn) return;
        const active = chatSessionStore.active;
        sendBtn.classList.toggle('is-stop', active);
        sendBtn.setAttribute('aria-label', active ? 'Stop' : 'Send');
        toggleSendButton();
    });

    queryForm.addEventListener('submit', function(e: SubmitEvent) {
        e.preventDefault();
        if (queryInFlight) return;  // never clear or submit while an answer streams
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
