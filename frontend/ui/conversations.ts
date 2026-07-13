// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import conversationStyles from '../styles/conversations.module.css';
import {
    ConversationApiError,
    createConversation,
    deleteConversation,
    getConversationHistory,
    listConversations,
    renameConversation,
    type ConversationSummary,
} from '../api/conversations.ts';
import {bus} from '../events/bus.ts';
import {
    renderConversationHistory,
    renderConversationHistoryError,
    renderConversationHistoryLoading,
} from '../lib/chat_renderer.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {isQueryInFlight, isQueryStopping} from './chat.ts';
import {hasActiveFileMutation} from './files-panel.ts';
import {clearImages, getPendingImageData} from './images.ts';
import {closePanel} from './panel.ts';
import {syncPanelEffectiveWidth} from './resize.ts';
import {showToast} from './toast.ts';

const COLLAPSED_KEY = 'dlightrag.conversation_sidebar_collapsed';
const DESKTOP_MEDIA = '(min-width: 1200px)';

type ListState = 'loading' | 'ready' | 'error' | 'empty-error';
type FocusResolver = () => HTMLElement | null;

let historyController: AbortController | null = null;
let bootstrapController: AbortController | null = null;
let pendingLifecycleAction = false;
let listState: ListState = 'loading';
let listErrorMessage = '';
let openMenuId: string | null = null;
let renameId: string | null = null;
let renameDraft: string | null = null;
let drawerOpen = false;
let desktopCollapsed = window.localStorage.getItem(COLLAPSED_KEY) === 'true';
let drawerReturnFocus: HTMLElement | null = null;

function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
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
    return Boolean(composerInput()?.value) || getPendingImageData().length > 0;
}

function clearDraft(): void {
    const input = composerInput();
    if (input) {
        input.value = '';
        input.dispatchEvent(new Event('input', {bubbles: true}));
    }
    clearImages();
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

function restoreStableFocus(resolveTarget: FocusResolver): void {
    window.requestAnimationFrame(function() { resolveTarget()?.focus(); });
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
            restoreStableFocus(resolveReturnTarget);
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
    renderConversationList();
}

function makeStatus(message: string, retryLabel: string, retry: () => void): HTMLElement {
    const status = document.createElement('div');
    status.className = 'conversation-list-status';
    status.setAttribute('role', 'status');

    const copy = document.createElement('span');
    copy.textContent = message;
    status.appendChild(copy);

    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = retryLabel;
    button.addEventListener('click', retry);
    status.appendChild(button);
    return status;
}

function closeActionsMenu(restoreFocus = false): void {
    const id = openMenuId;
    openMenuId = null;
    renderConversationList();
    if (restoreFocus && id) {
        document.querySelector<HTMLButtonElement>(
            `[data-conversation-id="${CSS.escape(id)}"] [aria-label="Conversation actions"]`,
        )?.focus();
    }
}

function focusMenuItem(conversationId: string, last = false): void {
    window.requestAnimationFrame(function() {
        const menu = document.querySelector<HTMLElement>(
            `[data-conversation-id="${CSS.escape(conversationId)}"] [role="menu"]`,
        );
        const items = menu?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]');
        if (!items || items.length === 0) return;
        items[last ? items.length - 1 : 0].focus();
    });
}

function startRename(conversationId: string): void {
    openMenuId = null;
    renameId = conversationId;
    renameDraft = conversationStore.conversations.find(
        (conversation) => conversation.conversation_id === conversationId,
    )?.title || '';
    renderConversationList();
    window.requestAnimationFrame(function() {
        const input = document.querySelector<HTMLInputElement>(
            `[data-conversation-id="${CSS.escape(conversationId)}"] input`,
        );
        input?.focus();
        input?.select();
    });
}

async function commitRename(conversation: ConversationSummary, title: string): Promise<void> {
    renameId = null;
    renameDraft = null;
    const trimmed = title.trim();
    if (!trimmed || trimmed === (conversation.title || '').trim()) {
        renderConversationList();
        return;
    }
    renderConversationList();
    try {
        const summary = await renameConversation(conversation.conversation_id, trimmed);
        conversationStore.upsertSummary(summary);
    } catch (error) {
        if (error instanceof ConversationApiError && error.status === 404) {
            const wasActive = conversationStore.activeConversationId === conversation.conversation_id;
            conversationStore.remove(conversation.conversation_id);
            if (wasActive) await selectFallbackConversation();
        } else if (error instanceof ConversationApiError && error.status === 422) {
            showToast('Conversation titles must be 1 to 120 characters.', 5000);
        } else {
            showToast('Could not rename the conversation.', 5000);
        }
        renderConversationList();
    }
}

function renderRenameInput(conversation: ConversationSummary): HTMLInputElement {
    const input = document.createElement('input');
    input.type = 'text';
    input.value = renameDraft ?? conversation.title ?? '';
    input.maxLength = 120;
    input.setAttribute('aria-label', 'Conversation title');
    let settled = false;
    const cancel = (): void => {
        if (settled) return;
        settled = true;
        renameId = null;
        renameDraft = null;
        renderConversationList();
    };
    const submit = (): void => {
        if (settled) return;
        settled = true;
        void commitRename(conversation, input.value);
    };
    input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            submit();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            cancel();
        }
    });
    input.addEventListener('input', function() { renameDraft = input.value; });
    input.addEventListener('blur', submit);
    return input;
}

function renderActionsMenu(conversation: ConversationSummary): HTMLElement {
    const menu = document.createElement('div');
    menu.className = 'conversation-actions-menu';
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', 'Conversation actions');

    const rename = document.createElement('button');
    rename.type = 'button';
    rename.setAttribute('role', 'menuitem');
    rename.textContent = 'Rename';
    rename.addEventListener('click', function() { startRename(conversation.conversation_id); });

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.setAttribute('role', 'menuitem');
    remove.className = 'conversation-delete-action';
    remove.textContent = 'Delete';
    remove.disabled = isQueryInFlight() || pendingLifecycleAction;
    remove.addEventListener('click', function() { void requestDelete(conversation.conversation_id); });

    menu.addEventListener('keydown', function(event) {
        const items = Array.from(menu.querySelectorAll<HTMLButtonElement>('[role="menuitem"]'))
            .filter((item) => !item.disabled);
        const index = items.indexOf(document.activeElement as HTMLButtonElement);
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            const delta = event.key === 'ArrowDown' ? 1 : -1;
            items[(index + delta + items.length) % items.length]?.focus();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            closeActionsMenu(true);
        }
    });
    menu.append(rename, remove);
    return menu;
}

function renderConversationRow(conversation: ConversationSummary): HTMLElement {
    const active = conversation.conversation_id === conversationStore.activeConversationId;
    const row = document.createElement('div');
    row.className = 'conversation-row';
    row.setAttribute('role', 'listitem');
    row.dataset.conversationId = conversation.conversation_id;
    if (active) row.setAttribute('aria-current', 'page');

    if (renameId === conversation.conversation_id) {
        row.appendChild(renderRenameInput(conversation));
    } else {
        const select = document.createElement('button');
        select.type = 'button';
        select.className = 'conversation-select';
        select.textContent = conversation.title || 'New chat';
        if (!conversation.title) select.setAttribute('aria-label', 'Open untitled conversation');
        select.disabled = isQueryInFlight() || pendingLifecycleAction;
        select.addEventListener('click', function() {
            void requestSelectConversation(conversation.conversation_id);
        });
        row.appendChild(select);
    }

    const actions = document.createElement('button');
    actions.type = 'button';
    actions.className = 'conversation-actions-button';
    actions.setAttribute('aria-label', 'Conversation actions');
    actions.setAttribute('aria-haspopup', 'menu');
    actions.setAttribute('aria-expanded', openMenuId === conversation.conversation_id ? 'true' : 'false');
    actions.textContent = '•••';
    actions.addEventListener('click', function(event) {
        event.stopPropagation();
        if (openMenuId === conversation.conversation_id) {
            closeActionsMenu();
            return;
        }
        openMenuId = conversation.conversation_id;
        renameId = null;
        renderConversationList();
        focusMenuItem(conversation.conversation_id);
    });
    actions.addEventListener('keydown', function(event) {
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        event.preventDefault();
        if (openMenuId !== conversation.conversation_id) {
            openMenuId = conversation.conversation_id;
            renderConversationList();
        }
        focusMenuItem(conversation.conversation_id, event.key === 'ArrowUp');
    });
    row.appendChild(actions);
    if (openMenuId === conversation.conversation_id) {
        row.appendChild(renderActionsMenu(conversation));
    }
    return row;
}

function renderConversationList(): void {
    const list = document.getElementById('conversation-list');
    if (!list) return;
    list.replaceChildren();
    const newButton = document.getElementById('new-conversation-btn') as HTMLButtonElement | null;
    if (newButton) newButton.disabled = isQueryInFlight() || pendingLifecycleAction;

    if (listState === 'loading' && conversationStore.conversations.length === 0) {
        for (let index = 0; index < 3; index += 1) {
            const skeleton = document.createElement('div');
            skeleton.className = 'conversation-skeleton';
            skeleton.setAttribute('aria-hidden', 'true');
            list.appendChild(skeleton);
        }
        const loading = document.createElement('span');
        loading.className = 'sr-only';
        loading.textContent = 'Loading conversations';
        list.appendChild(loading);
        return;
    }

    if (listState === 'error') {
        list.appendChild(makeStatus(
            listErrorMessage || 'Conversations are unavailable.',
            'Retry',
            function() { void initializeConversations(); },
        ));
    } else if (listState === 'empty-error') {
        list.appendChild(makeStatus(
            'No conversation is open.',
            'Retry New chat',
            function() { void requestNewConversation(); },
        ));
    }

    conversationStore.conversations.forEach(function(conversation) {
        list.appendChild(renderConversationRow(conversation));
    });

}

function closeCompactDrawer(restoreFocus = false): void {
    if (isDesktop() || !drawerOpen) return;
    drawerOpen = false;
    applySidebarState();
    if (restoreFocus) drawerReturnFocus?.focus();
    drawerReturnFocus = null;
}

function setBackgroundInert(inert: boolean): void {
    for (const selector of ['.topbar', '.chat-area', '.composer']) {
        const element = document.querySelector<HTMLElement>(selector);
        if (element) element.inert = inert;
    }
}

function applySidebarState(): void {
    const desktop = isDesktop();
    const sidebar = document.getElementById('chat-sidebar');
    const toggle = document.getElementById('conversation-sidebar-toggle');
    const open = document.getElementById('conversation-sidebar-open');
    const backdrop = document.getElementById('conversation-sidebar-backdrop');
    const expanded = desktop ? !desktopCollapsed : drawerOpen;

    document.body.classList.toggle('conversation-sidebar-open', expanded);
    document.body.classList.toggle('conversation-sidebar-collapsed', desktop && desktopCollapsed);
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
    setBackgroundInert(
        (!desktop && drawerOpen) || document.body.classList.contains('panel-drawer-open'),
    );
}

function openSidebar(trigger: HTMLElement | null): void {
    if (isDesktop()) {
        desktopCollapsed = false;
        window.localStorage.setItem(COLLAPSED_KEY, 'false');
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
        window.localStorage.setItem(COLLAPSED_KEY, 'true');
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
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
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
    renderConversationList();
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
        renderConversationList();
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
    openMenuId = null;
    renderConversationList();
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
        return resolveConversationActions(conversationId)
            || resolveActiveConversationSelect()
            || resolveNewConversationButton();
    };
    try {
        await deleteConversation(conversationId);
        if (wasActive) clearDraft();
        conversationStore.remove(conversationId);
        if (wasActive) {
            clearConversationSources();
            await selectFallbackConversation(false);
        }
        resolveFinalFocus = function() {
            return resolveActiveConversationSelect() || resolveNewConversationButton();
        };
    } catch (error) {
        if (error instanceof ConversationApiError && error.status === 404) {
            if (wasActive) clearDraft();
            conversationStore.remove(conversationId);
            if (wasActive) await selectFallbackConversation(false);
            resolveFinalFocus = function() {
                return resolveActiveConversationSelect() || resolveNewConversationButton();
            };
        } else {
            showToast('Could not delete the conversation.', 5000);
        }
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
    renderConversationList();
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
        listErrorMessage = 'Could not load conversations.';
        renderConversationList();
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
    document.getElementById('conversation-sidebar-toggle')?.addEventListener('click', toggleSidebar);
    document.getElementById('conversation-sidebar-open')?.addEventListener('click', function(event) {
        openSidebar(event.currentTarget as HTMLElement);
    });
    document.getElementById('conversation-sidebar-backdrop')?.addEventListener('click', function() {
        closeCompactDrawer(true);
    });
    document.getElementById('chat-sidebar')?.addEventListener('keydown', focusTrap);

    document.addEventListener('click', function(event) {
        if (!openMenuId || !(event.target instanceof Node)) return;
        const row = document.querySelector(
            `[data-conversation-id="${CSS.escape(openMenuId)}"]`,
        );
        if (row?.contains(event.target)) return;
        closeActionsMenu();
    });
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && !isDesktop() && drawerOpen) {
            if (document.querySelector('dialog[open]') || openMenuId) return;
            event.preventDefault();
            closeCompactDrawer(true);
        }
    });
    document.body.addEventListener('panelOpening', function() {
        closeCompactDrawer(false);
    });
    window.addEventListener('resize', applySidebarState);

    bus.on('conversationListChanged', renderConversationList);
    bus.on('conversationSelected', renderConversationList);
    bus.on('conversationStreamChanged', renderConversationList);
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
    renderConversationList();
    void initializeConversations();
}
