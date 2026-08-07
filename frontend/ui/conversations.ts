// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import conversationStyles from '../styles/conversations.module.css';
import {
    ConversationApiError,
    createConversation,
    deleteAllConversations,
    deleteConversation,
    getConversationHistory,
    listConversations,
    renameConversation,
} from '../api/conversations.ts';
import {bus} from '../events/bus.ts';
import {
    renderConversationHistory,
    renderConversationHistoryError,
    renderConversationHistoryLoading,
} from '../lib/chat_renderer.ts';
import {isAbortError} from '../lib/errors.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import './conversation_list.ts';
import type {
    ConversationIntentDetail,
    ConversationListState,
    ConversationRenameDetail,
    ConversationRetryDetail,
} from './conversation_list.ts';
import {isQueryInFlight, isQueryStopping} from './chat.ts';
import {hasActiveFileMutation} from './files-panel.ts';
import {clearAttachments, getPendingDocumentFiles} from './attachments.ts';
import {getPendingImageData} from './images.ts';
import {closePanel} from './panel.ts';
import {syncPanelEffectiveWidth} from './resize.ts';
import {showToast} from './toast.ts';
import {syncShellInert, wrapTabFocus} from '../lib/dom.ts';

const COLLAPSED_KEY = 'dlightrag.conversation_sidebar_collapsed';
const DESKTOP_MEDIA = '(min-width: 1200px)';

type FocusResolver = () => HTMLElement | null;

let historyController: AbortController | null = null;
let bootstrapController: AbortController | null = null;
let pendingLifecycleAction = false;
let listState: ConversationListState = 'loading';
let drawerOpen = false;
let desktopCollapsed = false;
let drawerReturnFocus: HTMLElement | null = null;

try {
    desktopCollapsed = window.localStorage.getItem(COLLAPSED_KEY) === 'true';
} catch (_error) {
    desktopCollapsed = false;
}

function setCollapsedPreference(value: boolean): void {
    try {
        window.localStorage.setItem(COLLAPSED_KEY, value ? 'true' : 'false');
    } catch (_error) {
        // Ignore unavailable or blocked storage.
    }
}

function isDesktop(): boolean {
    return window.matchMedia(DESKTOP_MEDIA).matches;
}

function composerInput(): HTMLTextAreaElement | null {
    return document.querySelector<HTMLTextAreaElement>('#query-form .composer-input');
}

function focusComposer(): void {
    window.requestAnimationFrame(function() { composerInput()?.focus(); });
}

function hasUnsavedDraft(): boolean {
    return (
        Boolean(composerInput()?.value)
        || getPendingImageData().length > 0
        || getPendingDocumentFiles().length > 0
    );
}

function clearDraft(): void {
    const input = composerInput();
    if (input) {
        input.value = '';
        input.dispatchEvent(new Event('input', {bubbles: true}));
    }
    clearAttachments();
}

function clearConversationSources(): void {
    if (document.getElementById('panel')?.dataset.panelKind !== 'sources') return;
    document.getElementById('panel-content')?.replaceChildren();
    closePanel();
}

function resolveConversationActions(conversationId: string): HTMLButtonElement | null {
    return document.querySelector<HTMLButtonElement>(
        `[data-conversation-id="${CSS.escape(conversationId)}"] [aria-label="Conversation actions"]`,
    );
}

function resolveConversationSelect(conversationId: string): HTMLButtonElement | null {
    return document.querySelector<HTMLButtonElement>(
        `[data-conversation-id="${CSS.escape(conversationId)}"] .conversation-select`,
    );
}

function resolveActiveConversationSelect(): HTMLButtonElement | null {
    const conversationId = conversationStore.activeConversationId;
    return conversationId ? resolveConversationSelect(conversationId) : null;
}

function resolveNewConversationButton(): HTMLButtonElement | null {
    return document.getElementById('new-conversation-btn') as HTMLButtonElement | null;
}

/** Where focus belongs once a delete has settled and its row is gone. */
function resolveSurvivingConversation(): HTMLElement | null {
    return resolveActiveConversationSelect() || resolveNewConversationButton();
}

function resolveDeleteAllButton(): HTMLButtonElement | null {
    return document.getElementById('delete-all-conversations-btn') as HTMLButtonElement | null;
}

function conversationList() {
    return document.querySelector('conversation-list');
}

/** Waits for the list to re-render so focus lands on a node that still exists. */
async function restoreStableFocus(resolveTarget: FocusResolver): Promise<void> {
    await conversationList()?.updateComplete;
    resolveTarget()?.focus();
}

function dialogResult(
    dialog: HTMLDialogElement,
    resolveReturnTarget: FocusResolver,
): Promise<string> {
    dialog.returnValue = '';
    dialog.showModal();
    return new Promise(function(resolve) {
        dialog.addEventListener('close', function() {
            const result = dialog.returnValue;
            void restoreStableFocus(resolveReturnTarget);
            resolve(result);
        }, {once: true});
    });
}

async function confirmDiscardDraft(resolveReturnTarget: FocusResolver): Promise<boolean> {
    if (!hasUnsavedDraft()) return true;
    const dialog = document.getElementById('discard-draft-dialog') as HTMLDialogElement | null;
    if (!dialog) return false;
    return await dialogResult(dialog, resolveReturnTarget) === 'discard';
}

function lifecycleBlocked(): boolean {
    if (!isQueryInFlight()) return false;
    if (!isQueryStopping()) {
        showToast('Stop the current response before changing conversations.', 4000);
    }
    return true;
}

function setLifecyclePending(pending: boolean): void {
    pendingLifecycleAction = pending;
    syncList();
}

/** Pushes the shell state the list cannot derive on its own. */
function syncList(): void {
    const busy = isQueryInFlight() || pendingLifecycleAction;
    resolveNewConversationButton()?.toggleAttribute('disabled', busy);
    resolveDeleteAllButton()?.toggleAttribute('disabled', busy);
    const list = conversationList();
    if (!list) return;
    list.busy = busy;
    list.listState = listState;
}

async function commitRename(conversationId: string, title: string): Promise<void> {
    try {
        conversationStore.upsertSummary(await renameConversation(conversationId, title));
    } catch (error) {
        if (error instanceof ConversationApiError && error.status === 404) {
            const wasActive = conversationStore.activeConversationId === conversationId;
            conversationStore.remove(conversationId);
            if (wasActive) await selectFallbackConversation();
        } else if (error instanceof ConversationApiError && error.status === 422) {
            showToast('Conversation titles must be 1 to 120 characters.', 5000);
        } else {
            showToast('Could not rename the conversation.', 5000);
        }
    }
}


function closeCompactDrawer(restoreFocus = false): void {
    if (isDesktop() || !drawerOpen) return;
    drawerOpen = false;
    applySidebarState();
    if (restoreFocus) drawerReturnFocus?.focus();
    drawerReturnFocus = null;
}

function applySidebarState(): void {
    const desktop = isDesktop();
    const sidebar = document.getElementById('chat-sidebar');
    const toggle = document.getElementById('conversation-sidebar-toggle');
    const open = document.getElementById('conversation-sidebar-open');
    const backdrop = document.getElementById('conversation-sidebar-backdrop');
    const expanded = desktop ? !desktopCollapsed : drawerOpen;

    document.body.classList.toggle('conversation-sidebar-open', expanded);
    document.body.classList.toggle('conversation-drawer-open', !desktop && drawerOpen);
    syncPanelEffectiveWidth();
    open?.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    if (toggle) {
        toggle.setAttribute('aria-label', desktop ? 'Collapse conversations' : 'Close conversations');
    }
    if (sidebar) {
        sidebar.inert = !expanded;
        if (expanded) sidebar.removeAttribute('aria-hidden');
        else sidebar.setAttribute('aria-hidden', 'true');
        if (!desktop && drawerOpen) {
            sidebar.setAttribute('role', 'dialog');
            sidebar.setAttribute('aria-modal', 'true');
        } else {
            sidebar.removeAttribute('role');
            sidebar.removeAttribute('aria-modal');
        }
    }
    if (backdrop) backdrop.hidden = desktop || !drawerOpen;
    syncShellInert();
}

function openSidebar(trigger: HTMLElement | null): void {
    if (isDesktop()) {
        desktopCollapsed = false;
        setCollapsedPreference(false);
        applySidebarState();
        document.getElementById('new-conversation-btn')?.focus();
        return;
    }

    const panel = document.getElementById('panel');
    if (panel?.classList.contains('open')) {
        if (hasActiveFileMutation()) {
            showToast('Wait for the file change to finish before opening conversations.', 5000);
            return;
        }
        closePanel();
    }
    drawerReturnFocus = trigger;
    drawerOpen = true;
    applySidebarState();
    document.getElementById('new-conversation-btn')?.focus();
}

function toggleSidebar(): void {
    if (isDesktop()) {
        desktopCollapsed = true;
        setCollapsedPreference(true);
        applySidebarState();
        document.getElementById('conversation-sidebar-open')?.focus();
        return;
    }
    closeCompactDrawer(true);
}

function focusTrap(event: KeyboardEvent): void {
    if (isDesktop() || !drawerOpen || event.key !== 'Tab') return;
    const sidebar = document.getElementById('chat-sidebar');
    const focusable = Array.from(sidebar?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ) || []).filter((element) => !element.hidden);
    wrapTabFocus(focusable, event);
}

async function loadConversation(
    conversationId: string,
    showLoading: boolean,
    clearSources: boolean,
    requestGeneration = conversationStore.beginRequest(),
): Promise<boolean> {
    historyController?.abort();
    if (!conversationStore.select(conversationId)) return false;
    const controller = new AbortController();
    historyController = controller;
    syncList();
    if (clearSources) clearConversationSources();
    if (showLoading && conversationStore.canRenderHistory(requestGeneration)) {
        renderConversationHistoryLoading();
    }

    try {
        const history = await getConversationHistory(conversationId, controller.signal);
        if (!conversationStore.isCurrentRequest(requestGeneration)) return false;
        if (conversationStore.setHistory(history, requestGeneration)) renderConversationHistory(history);
        return true;
    } catch (error) {
        if (isAbortError(error) || !conversationStore.isCurrentRequest(requestGeneration)) return false;
        if (error instanceof ConversationApiError && error.status === 404) {
            conversationStore.remove(conversationId);
            await selectFallbackConversation();
            return false;
        }
        if (showLoading && conversationStore.canRenderHistory(requestGeneration)) {
            renderConversationHistoryError(function() {
                void selectConversation(conversationId);
            });
        }
        return true;
    } finally {
        if (historyController === controller) historyController = null;
    }
}

export async function selectConversation(
    conversationId: string,
    showLoading = true,
    clearSources = true,
): Promise<void> {
    await loadConversation(conversationId, showLoading, clearSources);
}

async function requestSelectConversation(conversationId: string): Promise<void> {
    if (conversationId === conversationStore.activeConversationId) {
        closeCompactDrawer(true);
        focusComposer();
        return;
    }
    if (
        pendingLifecycleAction
        || lifecycleBlocked()
        || !await confirmDiscardDraft(function() { return resolveConversationSelect(conversationId); })
    ) return;
    setLifecyclePending(true);
    const accepted = await loadConversation(conversationId, true, true);
    if (accepted) {
        clearDraft();
        closeCompactDrawer(true);
        focusComposer();
    }
    setLifecyclePending(false);
}

async function requestNewConversation(): Promise<void> {
    if (
        pendingLifecycleAction
        || lifecycleBlocked()
        || !await confirmDiscardDraft(resolveNewConversationButton)
    ) return;
    setLifecyclePending(true);
    const requestGeneration = conversationStore.beginRequest();
    try {
        const summary = await createConversation();
        if (!conversationStore.isCurrentRequest(requestGeneration)) return;
        conversationStore.upsertSummary(summary);
        listState = 'ready';
        const accepted = await loadConversation(
            summary.conversation_id,
            true,
            true,
            requestGeneration,
        );
        if (accepted) {
            clearDraft();
            closeCompactDrawer(true);
            focusComposer();
        }
    } catch {
        showToast('Could not create a new conversation.', 5000);
        if (conversationStore.conversations.length === 0) listState = 'empty-error';
    } finally {
        setLifecyclePending(false);
    }
}

async function selectFallbackConversation(focusAfter = true): Promise<void> {
    const fallback = conversationStore.conversations[0];
    if (fallback) {
        await loadConversation(fallback.conversation_id, true, true);
        if (focusAfter) focusComposer();
        return;
    }
    await createFallbackConversation(focusAfter);
}

async function createFallbackConversation(focusAfter = true): Promise<void> {
    const requestGeneration = conversationStore.beginRequest();
    try {
        const summary = await createConversation();
        if (!conversationStore.isCurrentRequest(requestGeneration)) return;
        conversationStore.upsertSummary(summary);
        listState = 'ready';
        await loadConversation(summary.conversation_id, true, true, requestGeneration);
        if (focusAfter) focusComposer();
    } catch {
        if (!conversationStore.isCurrentRequest(requestGeneration)) return;
        listState = 'empty-error';
        syncList();
        renderConversationHistoryError(function() { void requestNewConversation(); });
        showToast('Could not create a replacement conversation.', 5000);
    }
}

async function requestDelete(conversationId: string): Promise<void> {
    if (pendingLifecycleAction || lifecycleBlocked()) return;
    const wasActive = conversationStore.activeConversationId === conversationId;
    const discardsDraft = wasActive && hasUnsavedDraft();
    const resolveActions = function(): HTMLElement | null {
        return resolveConversationActions(conversationId);
    };
    const dialog = document.getElementById('delete-conversation-dialog') as HTMLDialogElement | null;
    const warning = document.getElementById('delete-conversation-draft-warning');
    if (warning) warning.hidden = !discardsDraft;
    if (dialog) {
        dialog.setAttribute(
            'aria-describedby',
            discardsDraft
                ? 'delete-conversation-message delete-conversation-draft-warning'
                : 'delete-conversation-message',
        );
    }
    if (!dialog || await dialogResult(dialog, resolveActions) !== 'delete') return;
    if (lifecycleBlocked()) return;

    setLifecyclePending(true);
    let resolveFinalFocus: FocusResolver = function() {
        return resolveConversationActions(conversationId) || resolveSurvivingConversation();
    };
    try {
        await deleteConversation(conversationId);
        if (wasActive) clearDraft();
        conversationStore.remove(conversationId);
        if (wasActive) {
            clearConversationSources();
            await selectFallbackConversation(false);
        }
        resolveFinalFocus = resolveSurvivingConversation;
    } catch (error) {
        if (error instanceof ConversationApiError && error.status === 404) {
            if (wasActive) clearDraft();
            conversationStore.remove(conversationId);
            if (wasActive) await selectFallbackConversation(false);
            resolveFinalFocus = resolveSurvivingConversation;
        } else {
            showToast('Could not delete the conversation.', 5000);
        }
    } finally {
        setLifecyclePending(false);
        restoreStableFocus(resolveFinalFocus);
    }
}

async function requestDeleteAll(): Promise<void> {
    if (pendingLifecycleAction || lifecycleBlocked()) return;
    const trigger = resolveDeleteAllButton;
    const discardsDraft = hasUnsavedDraft();
    const dialog = document.getElementById(
        'delete-all-conversations-dialog',
    ) as HTMLDialogElement | null;
    const warning = document.getElementById('delete-all-conversations-draft-warning');
    if (warning) warning.hidden = !discardsDraft;
    if (dialog) {
        if (discardsDraft) {
            dialog.setAttribute('aria-describedby', 'delete-all-conversations-draft-warning');
        } else {
            dialog.removeAttribute('aria-describedby');
        }
    }
    if (!dialog || await dialogResult(dialog, trigger) !== 'delete-all') return;
    if (lifecycleBlocked()) return;

    setLifecyclePending(true);
    let resolveFinalFocus: FocusResolver = trigger;
    try {
        await deleteAllConversations();
        clearDraft();
        clearConversationSources();
        for (const conversation of [...conversationStore.conversations]) {
            conversationStore.remove(conversation.conversation_id);
        }
        listState = 'ready';
        await createFallbackConversation(false);
        resolveFinalFocus = resolveSurvivingConversation;
    } catch {
        showToast('Could not delete conversations.', 5000);
    } finally {
        setLifecyclePending(false);
        restoreStableFocus(resolveFinalFocus);
    }
}

export async function initializeConversations(): Promise<void> {
    bootstrapController?.abort();
    const controller = new AbortController();
    bootstrapController = controller;
    const requestGeneration = conversationStore.beginRequest();
    listState = 'loading';
    syncList();
    try {
        const conversations = await listConversations(controller.signal);
        if (!conversationStore.isCurrentRequest(requestGeneration)) return;
        conversationStore.replaceList(conversations);
        listState = 'ready';
        let selected = conversationStore.initialSelection();
        if (!selected) {
            selected = await createConversation(controller.signal);
            if (!conversationStore.isCurrentRequest(requestGeneration)) return;
            conversationStore.upsertSummary(selected);
        }
        await loadConversation(selected.conversation_id, true, false, requestGeneration);
    } catch (error) {
        if (isAbortError(error) || !conversationStore.isCurrentRequest(requestGeneration)) return;
        listState = conversationStore.conversations.length > 0 ? 'error' : 'empty-error';
        syncList();
        if (conversationStore.conversations.length === 0) {
            renderConversationHistoryError(function() { void initializeConversations(); });
        }
    } finally {
        if (bootstrapController === controller) bootstrapController = null;
    }
}

export function setupConversations(): void {
    document.getElementById('chat-sidebar')?.classList.add(conversationStyles.root);
    document.getElementById('new-conversation-btn')?.addEventListener('click', function() {
        void requestNewConversation();
    });
    resolveDeleteAllButton()?.addEventListener('click', function() {
        void requestDeleteAll();
    });
    document.getElementById('conversation-sidebar-toggle')?.addEventListener('click', toggleSidebar);
    document.getElementById('conversation-sidebar-open')?.addEventListener('click', function(event) {
        openSidebar(event.currentTarget as HTMLElement);
    });
    document.getElementById('conversation-sidebar-backdrop')?.addEventListener('click', function() {
        closeCompactDrawer(true);
    });
    document.getElementById('chat-sidebar')?.addEventListener('keydown', focusTrap);

    const list = conversationList();
    list?.addEventListener('conversation-select', function({detail}) {
        void requestSelectConversation(detail.conversationId);
    });
    list?.addEventListener('conversation-delete', function({detail}) {
        void requestDelete(detail.conversationId);
    });
    list?.addEventListener('conversation-rename', function({detail}) {
        void commitRename(detail.conversationId, detail.title);
    });
    list?.addEventListener('conversation-retry', function({detail}) {
        void (detail.kind === 'reload' ? initializeConversations() : requestNewConversation());
    });

    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && !isDesktop() && drawerOpen) {
            if (document.querySelector('dialog[open]') || conversationList()?.menuOpen) return;
            event.preventDefault();
            closeCompactDrawer(true);
        }
    });
    document.body.addEventListener('panelOpening', function() {
        closeCompactDrawer(false);
    });
    window.addEventListener('resize', applySidebarState);

    // The list re-renders itself from the store; only the in-flight flag is ours.
    bus.on('conversationStreamChanged', syncList);
    bus.on('conversationAnswerSaved', function({conversationId}) {
        if (conversationStore.activeConversationId !== conversationId) return;
        void selectConversation(conversationId, false, false);
    });
    bus.on('conversationSaveCheckRequested', function({conversationId}) {
        if (conversationStore.activeConversationId !== conversationId) return;
        void selectConversation(conversationId, true, false);
    });
    bus.on('conversationDeferredSelectionReady', function({conversationId}) {
        void selectConversation(conversationId, true, true);
    });

    applySidebarState();
    syncList();
    void initializeConversations();
}
