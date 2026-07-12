// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/layout.css';
import '../styles/files.css';
import '../styles/sources.css';
import {
    createConversation,
    getConversationHistory,
    listConversations,
} from '../api/conversations.ts';
import {bus} from '../events/bus.ts';
import {
    renderConversationHistory,
    renderConversationHistoryError,
    renderConversationHistoryLoading,
} from '../lib/chat_renderer.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {closePanel} from './panel.ts';

let historyController: AbortController | null = null;
let bootstrapController: AbortController | null = null;

function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

function clearConversationSources(): void {
    if (document.getElementById('panel-title')?.textContent !== 'SOURCES') return;
    document.getElementById('panel-content')?.replaceChildren();
    closePanel();
}

export async function selectConversation(
    conversationId: string,
    showLoading = true,
    clearSources = true,
): Promise<void> {
    historyController?.abort();
    const controller = new AbortController();
    historyController = controller;
    const generation = conversationStore.beginRequest();
    conversationStore.select(conversationId);
    if (clearSources) clearConversationSources();
    if (showLoading) renderConversationHistoryLoading();

    try {
        const history = await getConversationHistory(conversationId, controller.signal);
        if (conversationStore.setHistory(history, generation)) renderConversationHistory(history);
    } catch (error) {
        if (isAbortError(error) || generation !== conversationStore.generation) return;
        if (showLoading) {
            renderConversationHistoryError(function() {
                void selectConversation(conversationId);
            });
        }
    } finally {
        if (historyController === controller) historyController = null;
    }
}

export async function initializeConversations(): Promise<void> {
    bootstrapController?.abort();
    const controller = new AbortController();
    bootstrapController = controller;
    try {
        const conversations = await listConversations(controller.signal);
        conversationStore.replaceList(conversations);
        const stored = conversationStore.activeConversationId;
        let selected = conversations.find(
            (conversation) => conversation.conversation_id === stored,
        ) || conversations[0];
        if (!selected) {
            selected = await createConversation(controller.signal);
            conversationStore.upsertSummary(selected);
        }
        await selectConversation(selected.conversation_id);
    } catch (error) {
        if (isAbortError(error)) return;
        renderConversationHistoryError(function() { void initializeConversations(); });
    } finally {
        if (bootstrapController === controller) bootstrapController = null;
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    const [
        {setupQueryForm},
        {setupHtmxInteractions},
        {setupImageInputs},
        {setupMathRendering},
        {setupPanel},
        {setupSourcePanel},
        {setupFilesPanel},
        {setupPanelResize},
        {initWorkspaces},
    ] = await Promise.all([
        import('./chat.ts'),
        import('./htmx.ts'),
        import('./images.ts'),
        import('./mathjax.ts'),
        import('./panel.ts'),
        import('./source-panel.ts'),
        import('./files-panel.ts'),
        import('./resize.ts'),
        import('./workspaces.ts'),
    ]);

    initWorkspaces();
    setupPanel();
    setupSourcePanel();
    setupFilesPanel();
    setupPanelResize();
    setupHtmxInteractions();
    setupImageInputs();
    setupQueryForm();
    setupMathRendering();

    bus.on('conversationAnswerSaved', function({conversationId}) {
        if (conversationStore.activeConversationId !== conversationId) return;
        void selectConversation(conversationId, false, false);
    });
    void initializeConversations();

});
