// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {workspaceStore} from '../stores/workspaceStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {clearAttachments, getPendingAttachments} from './attachments.ts';
import {streamSSE} from '../lib/sse.ts';
import {buildAnswerRequest} from '../lib/answer_request.ts';
import type {ConversationAttachmentReference} from '../api/conversations.ts';
import {
    createAnswerRenderer,
    createChatTurn,
    markAnswerStopped,
    renderAnswerSaveOutcome,
    setAnswerError,
} from '../lib/chat_renderer.ts';
import {bus} from '../events/bus.ts';
import {
    isDefinitiveSaveOutcome,
    describeConversationSaveOutcome,
    payloadFingerprint,
    pendingSubmissionStore,
    shouldKeepLiveConversation,
} from '../stores/pendingSubmissionStore.ts';

let queryInFlight = false;
let queryStopping = false;
let currentQueryController: AbortController | null = null;
const STREAM_IDLE_TIMEOUT_MS = 120_000;

/** Abort the in-flight answer request (user stop / navigation), if any. */
export function cancelQuery(): void {
    if (currentQueryController && !currentQueryController.signal.aborted) {
        queryStopping = true;
        currentQueryController.abort(new DOMException('Query cancelled', 'AbortError'));
    }
}

export function isQueryInFlight(): boolean {
    return queryInFlight;
}

/** User pressed Stop — abort was sent, waiting for stream cleanup. */
export function isQueryStopping(): boolean {
    return queryStopping;
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
    currentQueryController = controller;
    let idleTimer = 0;
    const armIdleTimeout = (): void => {
        clearTimeout(idleTimer);
        idleTimer = window.setTimeout(
            () => controller.abort(new DOMException('Stream idle timeout', 'TimeoutError')),
            STREAM_IDLE_TIMEOUT_MS,
        );
    };

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
        armIdleTimeout();
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
        const submissionId = pendingSubmissionStore.getOrCreate(conversationId, fingerprint);
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
        for (let attempt = 0; attempt < 2; attempt += 1) {
            try {
                const response = await fetch('/web/answer', {
                    method: 'POST',
                    ...(requestHeaders ? {headers: requestHeaders} : {}),
                    signal: controller.signal,
                    body: requestBody,
                });

                if (!response.ok) {
                    if (response.status < 500) pendingSubmissionStore.clear(conversationId);
                    setAnswerError(turn, 'Service error. Please try again.');
                    return;
                }

                const activeRenderer = createAnswerRenderer(turn);
                await streamSSE(response, function(eventType, data) {
                    armIdleTimeout();
                    if (conversationStore.activeConversationId !== conversationId) return;
                    activeRenderer.handle(eventType, data);
                });
                if (activeRenderer.failed || isDefinitiveSaveOutcome(activeRenderer.saveOutcome)) {
                    pendingSubmissionStore.clear(conversationId);
                }
                const saveOutcome = activeRenderer.saveOutcome;
                if (shouldKeepLiveConversation(saveOutcome)) {
                    releaseLiveViewport(true);
                    if (saveOutcome.conversation) {
                        conversationStore.upsertSummary(saveOutcome.conversation);
                    }
                    bus.emit('conversationAnswerSaved', {conversationId});
                } else if (saveOutcome?.conversation_saved === false) {
                    renderAnswerSaveOutcome(
                        turn,
                        describeConversationSaveOutcome(saveOutcome),
                        function() {
                            bus.emit('conversationSaveCheckRequested', {conversationId});
                        },
                    );
                }
                releaseLiveViewport();
                return;
            } catch (error) {
                if (controller.signal.aborted || attempt === 1) throw error;
            }
        }
    } catch (_) {
        if (controller.signal.aborted) {
            // User stopped: keep the partial answer and drop the now-stale
            // pending submission instead of wiping the turn.
            if (conversationId) pendingSubmissionStore.clear(conversationId);
            markAnswerStopped(turn);
        } else {
            setAnswerError(turn, 'Connection error. Please try again.');
        }
    } finally {
        releaseLiveViewport();
        clearTimeout(idleTimer);
        if (currentQueryController === controller) currentQueryController = null;
        queryInFlight = false;
        queryStopping = false;
        bus.emit('conversationStreamChanged', {active: false});
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
