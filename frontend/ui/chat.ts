// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {workspaceStore} from '../stores/workspaceStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {getPendingImageData} from './images.ts';
import {clearAttachments, getPendingDocumentFiles} from './attachments.ts';
import {streamSSE} from '../lib/sse.ts';
import {
    createAnswerRenderer,
    createChatTurn,
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

/**
 * Build the `/web/answer` request body. Keeps the legacy JSON path byte-for-byte
 * when no documents are attached; switches to multipart/form-data (matching the
 * server's `parse_web_answer_request` contract) only when documents are present.
 */
function buildAnswerRequestBody(input: {
    query: string;
    images: string[];
    workspaces: string[];
    conversationId: string;
    submissionId: string;
    documents: File[];
}): {body: BodyInit; headers?: Record<string, string>} {
    if (input.documents.length === 0) {
        return {
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: input.query,
                images: input.images,
                workspaces: input.workspaces,
                conversation_id: input.conversationId,
                submission_id: input.submissionId,
            }),
        };
    }
    const form = new FormData();
    form.append('query', input.query);
    form.append('images', JSON.stringify(input.images));
    form.append('workspaces', JSON.stringify(input.workspaces));
    form.append('conversation_id', input.conversationId);
    form.append('submission_id', input.submissionId);
    input.documents.forEach(function(file) { form.append('documents', file, file.name); });
    return {body: form};
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

    const turn = createChatTurn(query);
    const imageData = getPendingImageData();
    const documentFiles = getPendingDocumentFiles();

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
            images: imageData,
            documents: documentFiles.map(function(file) {
                return {name: file.name, size: file.size, type: file.type};
            }),
            workspaces: activeWorkspaces,
        });
        if (conversationStore.activeConversationId !== conversationId) {
            setAnswerError(turn, 'The active conversation changed before this answer started.');
            return;
        }
        const submissionId = pendingSubmissionStore.getOrCreate(conversationId, fingerprint);
        const {body: requestBody, headers: requestHeaders} = buildAnswerRequestBody({
            query,
            images: imageData,
            workspaces: activeWorkspaces,
            conversationId: conversationStore.activeConversationId,
            submissionId,
            documents: documentFiles,
        });
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
        setAnswerError(
            turn,
            controller.signal.aborted ? 'Answer stopped.' : 'Connection error. Please try again.',
        );
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
