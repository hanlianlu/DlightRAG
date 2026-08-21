// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import conversationStyles from '../styles/conversations.module.css';
import {detachAnswerRun, isSubmissionPending, resumePendingTurn} from './chat.ts';
import {clearMemory} from '../api/memory.ts';
import {
    clearChatViewport,
    renderConversationHistory,
    renderConversationHistoryError,
    renderConversationHistoryLoading,
    renderConversationUnavailable,
} from '../lib/chat_renderer.ts';
import {conversationRoute, newChatRoute, type WebRoute} from '../lib/router.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import './conversation_list.ts';
import type {
    ConversationIntentDetail,
    ConversationRenameDetail,
    ConversationRetryDetail,
} from './conversation_list.ts';
import {hasActiveFileMutation} from './files-panel.ts';
import {clearAttachments, getPendingAttachments} from './attachments.ts';
import {closeConversationPanels, closePanel} from './panel.ts';
import {syncPanelSplitState} from './split_panel.ts';
import {showToast} from './toast.ts';
import {syncShellInert, wrapTabFocus} from '../lib/dom.ts';
import {webRouter} from './router.ts';

const COLLAPSED_KEY = 'dlightrag.conversation_sidebar_collapsed';
const DESKTOP_MEDIA = '(min-width: 1200px)';

type FocusResolver = () => HTMLElement | null;

let pendingLifecycleAction = false;
let renderedViewRevision = -1;
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
        || getPendingAttachments().length > 0
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

export function dialogResult(
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

/**
 * Only an unaccepted submission blocks a conversation change.
 *
 * Once the run is durable, leaving merely detaches this tab's event reader: the
 * run keeps producing and the turn is waiting in history on the way back.
 */
function lifecycleBlocked(): boolean {
    if (!isSubmissionPending()) return false;
    showToast('Wait for the current question to be accepted.', 4000);
    return true;
}

function setLifecyclePending(pending: boolean): void {
    pendingLifecycleAction = pending;
    syncList();
}

/** Pushes the shell state the list cannot derive on its own. */
function syncList(): void {
    const busy = pendingLifecycleAction || conversationStore.mutationPending;
    resolveNewConversationButton()?.toggleAttribute('disabled', busy);
    resolveDeleteAllButton()?.toggleAttribute('disabled', busy);
    const list = conversationList();
    if (!list) return;
    list.busy = busy;
    list.listState = conversationStore.listState;
}

async function commitRename(conversationId: string, title: string): Promise<void> {
    const result = await conversationStore.rename(conversationId, title);
    if (result === 'ok') return;
    if (result === 'missing') {
        showToast('Conversation unavailable.', 4000);
        return;
    }
    showToast(
        title.trim().length > 120
            ? 'Conversation titles must be 1 to 120 characters.'
            : 'Could not rename the conversation.',
        5000,
    );
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
    syncPanelSplitState();
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

function routeFocusTarget(route: WebRoute): FocusResolver {
    if (route.kind === 'conversation') {
        return function() { return resolveConversationSelect(route.conversationId); };
    }
    return resolveNewConversationButton;
}

async function guardNavigation(next: WebRoute): Promise<boolean> {
    if (pendingLifecycleAction || lifecycleBlocked()) return false;
    if (!hasUnsavedDraft()) return true;
    const accepted = await confirmDiscardDraft(routeFocusTarget(next));
    if (accepted) clearDraft();
    return accepted;
}

function renderCurrentConversationView(): void {
    syncList();
    if (renderedViewRevision === conversationStore.viewRevision) return;
    renderedViewRevision = conversationStore.viewRevision;

    if (conversationStore.viewState === 'new') {
        clearChatViewport();
        return;
    }
    if (conversationStore.viewState === 'loading') {
        renderConversationHistoryLoading();
        return;
    }
    if (conversationStore.viewState === 'ready') {
        const history = conversationStore.history;
        if (!history) return;
        const pending = renderConversationHistory(history);
        if (pending) {
            void resumePendingTurn(
                pending.turn,
                history.conversation.conversation_id,
                pending.stored,
            );
        }
        return;
    }
    if (conversationStore.viewState === 'unavailable') {
        const fallback = conversationStore.fallbackConversationId;
        renderConversationUnavailable(
            function() { void webRouter.navigate(newChatRoute()); },
            fallback
                ? function() { void webRouter.navigate(conversationRoute(fallback)); }
                : null,
        );
        return;
    }
    const conversationId = conversationStore.activeConversationId;
    renderConversationHistoryError(function() {
        if (conversationId) void conversationStore.open(conversationId);
        else void conversationStore.loadList();
    });
}

async function applyRoute(route: WebRoute): Promise<void> {
    const previous = conversationStore.activeConversationId;
    const next = route.kind === 'conversation' ? route.conversationId : null;
    if (previous !== next) {
        detachAnswerRun();
        closeConversationPanels();
    }

    if (route.kind === 'conversation') {
        await conversationStore.open(route.conversationId);
    } else if (route.kind === 'new') {
        conversationStore.openNew();
    } else {
        conversationStore.openNew();
        renderConversationUnavailable(
            function() { void webRouter.navigate(newChatRoute(), {replace: true}); },
            null,
        );
    }
    closeCompactDrawer(false);
}

async function requestSelectConversation(conversationId: string): Promise<void> {
    if (conversationId === conversationStore.activeConversationId) {
        closeCompactDrawer(true);
        focusComposer();
        return;
    }
    if (await webRouter.navigate(conversationRoute(conversationId))) {
        closeCompactDrawer(true);
        focusComposer();
    }
}

async function requestNewConversation(): Promise<void> {
    if (webRouter.current.kind === 'new') {
        closeCompactDrawer(true);
        focusComposer();
        return;
    }
    if (await webRouter.navigate(newChatRoute())) {
        closeCompactDrawer(true);
        focusComposer();
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
    if (wasActive) detachAnswerRun();
    let resolveFinalFocus: FocusResolver = function() {
        return resolveConversationActions(conversationId) || resolveSurvivingConversation();
    };
    const result = await conversationStore.delete(conversationId);
    if (result === 'error') {
        showToast('Could not delete the conversation.', 5000);
    } else {
        if (wasActive) {
            clearDraft();
            closeConversationPanels();
            const fallback = conversationStore.fallbackConversationId;
            await webRouter.navigate(
                fallback ? conversationRoute(fallback) : newChatRoute(),
                {replace: true, bypassGuard: true},
            );
        }
        resolveFinalFocus = resolveSurvivingConversation;
    }
    setLifecyclePending(false);
    await restoreStableFocus(resolveFinalFocus);
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

    const alsoClearMemory = (
        document.getElementById('delete-all-also-clear-memory') as HTMLInputElement | null
    )?.checked;

    setLifecyclePending(true);
    detachAnswerRun();
    const result = await conversationStore.deleteAll();
    if (result === 'error') {
        showToast('Could not delete conversations.', 5000);
    } else {
        clearDraft();
        closeConversationPanels();
        if (alsoClearMemory) {
            try {
                await clearMemory();
            } catch {
                showToast('Conversations deleted; could not clear Profile memory.', 5000);
            }
        }
        await webRouter.navigate(newChatRoute(), {replace: true, bypassGuard: true});
    }
    setLifecyclePending(false);
    await restoreStableFocus(result === 'error' ? trigger : resolveNewConversationButton);
}

export async function initializeConversations(): Promise<void> {
    await conversationStore.loadList();
    await applyRoute(webRouter.current);
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
    window.addEventListener('beforeunload', function(event) {
        if (!hasUnsavedDraft()) return;
        event.preventDefault();
        event.returnValue = '';
    });

    conversationStore.subscribe(renderCurrentConversationView);
    webRouter.setGuard(guardNavigation);
    webRouter.start(function(route) { return applyRoute(route); });

    applySidebarState();
    syncList();
    void initializeConversations();
}
