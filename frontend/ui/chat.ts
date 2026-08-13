// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {workspaceStore} from '../stores/workspaceStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {clearAttachments, getPendingAttachments} from './attachments.ts';
import {createSSEParser} from '../lib/sse.ts';
import {buildAnswerRequest} from '../lib/answer_request.ts';
import {cancelAnswerRun, getAnswerRun} from '../api/conversations.ts';
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
} from '../lib/chat_renderer.ts';
import type {ChatTurn} from '../lib/chat_renderer.ts';
import {bus} from '../events/bus.ts';
import {answerRunStore, payloadFingerprint} from '../stores/answerRunStore.ts';

// A dropped connection is a transport fault, never a decision about the run, so
// the browser reattaches from its last durable sequence a bounded number of
// times before falling back to the run's stored state.
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 500;

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

export function isQueryInFlight(): boolean {
    return queryInFlight;
}

/** User pressed Stop — cancellation was requested, waiting for the run to end. */
export function isQueryStopping(): boolean {
    return queryStopping;
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
    for (let attempt = 0; attempt <= MAX_RECONNECT_ATTEMPTS; attempt += 1) {
        if (controller.signal.aborted) return false;
        const after = answerRunStore.lastSequence(conversationId, runId);
        let response: Response;
        try {
            response = await fetch(`/web/answer/${encodeURIComponent(runId)}/events`, {
                signal: controller.signal,
                headers: after > 0 ? {'Last-Event-ID': String(after)} : undefined,
            });
        } catch (error) {
            if (controller.signal.aborted) return false;
            if (attempt === MAX_RECONNECT_ATTEMPTS) throw error;
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
        } catch (error) {
            if (controller.signal.aborted) return false;
            if (attempt === MAX_RECONNECT_ATTEMPTS) throw error;
            await sleep(RECONNECT_DELAY_MS);
            continue;
        }
        if (renderer.terminal) return true;
        if (attempt === MAX_RECONNECT_ATTEMPTS) break;
        await sleep(RECONNECT_DELAY_MS);
    }
    // The event log was trimmed or the stream ended without a terminal event;
    // the run row stays authoritative.
    if (controller.signal.aborted) return false;
    try {
        const stored = await getAnswerRun(runId);
        renderStoredTurn(turn, stored);
        return stored.status !== 'queued' && stored.status !== 'running';
    } catch (_) {
        setAnswerError(turn, 'Service error. Please try again.');
        return true;
    }
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
    queryInFlight = true;
    queryStopping = cancelRequested;
    currentFollowController = controller;
    currentRunId = runId;
    bus.emit('conversationStreamChanged', {active: true});
}

function endFollowing(
    controller: AbortController,
    conversationId: string,
    finished: boolean,
): void {
    if (currentFollowController !== controller) return;
    currentFollowController = null;
    currentRunId = null;
    queryInFlight = false;
    queryStopping = false;
    bus.emit('conversationStreamChanged', {active: false});
    if (!finished) return;
    // A finished run retires its submission key, so asking the same question
    // again is a new run rather than a replay of this one.
    answerRunStore.clear(conversationId);
    bus.emit('conversationAnswerSaved', {conversationId});
}

export async function submitQuery(query: string): Promise<void> {
    if (queryInFlight) return;
    queryInFlight = true;
    bus.emit('conversationStreamChanged', {active: true});

    const conversationId = conversationStore.answerConversationId;
    if (conversationId) conversationStore.beginLiveAnswer(conversationId);
    let ownsLiveViewport = conversationId !== null;
    const releaseLiveViewport = (discardPendingSelection = false): void => {
        if (!ownsLiveViewport || !conversationId) return;
        const deferredSelection = conversationStore.finishLiveAnswer(
            conversationId,
            discardPendingSelection,
        );
        ownsLiveViewport = false;
        if (deferredSelection) {
            bus.emit('conversationDeferredSelectionReady', {conversationId: deferredSelection});
        }
    };

    const controller = new AbortController();
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
        if (!conversationId) {
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
        if (conversationStore.activeConversationId !== conversationId) {
            setAnswerError(turn, 'The active conversation changed before this answer started.');
            return;
        }
        const submissionId = answerRunStore.getOrCreateSubmissionId(conversationId, fingerprint);
        const {body: requestBody, headers: requestHeaders} = buildAnswerRequest(
            {
                query,
                workspaces: activeWorkspaces,
                conversationId: conversationStore.activeConversationId,
                submissionId,
            },
            attachmentFiles,
        );
        clearAttachments();
        const response = await fetch('/web/answer', {
            method: 'POST',
            ...(requestHeaders ? {headers: requestHeaders} : {}),
            body: requestBody,
        });

        if (!response.ok) {
            if (response.status < 500) answerRunStore.clear(conversationId);
            setAnswerError(turn, 'Service error. Please try again.');
            return;
        }
        const descriptor = await response.json() as AnswerRunDescriptor;
        answerRunStore.attachRun(conversationId, descriptor.run_id);
        conversationStore.upsertSummary(descriptor.conversation);
        markAnswerPending(turn, descriptor.cancel_requested);
        // The run is durable history now, so the composer is released and the
        // rest of this call only follows an object the server already owns.
        releaseLiveViewport(true);
        beginFollowing(controller, descriptor.run_id, descriptor.cancel_requested);
        finished = await followAnswerRun(turn, conversationId, descriptor.run_id, controller);
    } catch (_) {
        if (!controller.signal.aborted) {
            setAnswerError(turn, 'Connection error. Please try again.');
            finished = true;
        }
    } finally {
        releaseLiveViewport();
        if (conversationId) endFollowing(controller, conversationId, finished);
        if (currentFollowController === null) {
            queryInFlight = false;
            queryStopping = false;
            bus.emit('conversationStreamChanged', {active: false});
        }
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
    bus.on('conversationStreamChanged', function({active}) {
        if (!sendBtn) return;
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
