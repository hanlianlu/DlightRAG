// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {ConversationSummary} from '../api/conversations.ts';
import {BusController, LightElement} from '../lib/lit_host.ts';
import {conversationStore} from '../stores/conversationStore.ts';

export type ConversationListState = 'loading' | 'ready' | 'error' | 'empty-error';

export interface ConversationIntentDetail {
    conversationId: string;
}

export interface ConversationRenameDetail extends ConversationIntentDetail {
    title: string;
}

export interface ConversationRetryDetail {
    kind: 'reload' | 'new';
}

const SKELETON_COUNT = 3;

/**
 * Renders the conversation sidebar list from the store and reports intents as
 * events; every lifecycle decision stays with the owning module.
 */
export class ConversationList extends LightElement {
    static properties = {
        listState: {attribute: false},
        busy: {attribute: false},
        openMenuId: {state: true},
        renameId: {state: true},
    };

    declare listState: ConversationListState;
    declare busy: boolean;
    declare openMenuId: string | null;
    declare renameId: string | null;

    #dismiss: AbortController | null = null;

    constructor() {
        super();
        this.listState = 'loading';
        this.busy = false;
        this.openMenuId = null;
        this.renameId = null;
        new BusController(this, 'conversationListChanged', 'conversationSelected');
    }

    override connectedCallback(): void {
        super.connectedCallback();
        const dismiss = new AbortController();
        this.#dismiss = dismiss;
        document.addEventListener('click', (event) => {
            if (!this.openMenuId || !(event.target instanceof Node)) return;
            if (this.#row(this.openMenuId)?.contains(event.target)) return;
            this.openMenuId = null;
        }, {signal: dismiss.signal});
    }

    override disconnectedCallback(): void {
        super.disconnectedCallback();
        this.#dismiss?.abort();
        this.#dismiss = null;
    }

    /** True while a row menu is open, so the shell can leave Escape to us. */
    get menuOpen(): boolean {
        return this.openMenuId !== null;
    }

    closeMenu(restoreFocus = false): void {
        const conversationId = this.openMenuId;
        if (conversationId === null) return;
        this.openMenuId = null;
        if (restoreFocus) void this.#focusAfterRender(`[aria-label="Conversation actions"]`, conversationId);
    }

    #row(conversationId: string): HTMLElement | null {
        return this.querySelector<HTMLElement>(
            `[data-conversation-id="${CSS.escape(conversationId)}"]`,
        );
    }

    async #focusAfterRender(selector: string, conversationId: string, last = false): Promise<void> {
        await this.updateComplete;
        const matches = this.#row(conversationId)?.querySelectorAll<HTMLElement>(selector);
        if (!matches?.length) return;
        matches[last ? matches.length - 1 : 0].focus();
    }

    #emit<D>(type: string, detail: D): void {
        this.dispatchEvent(new CustomEvent<D>(type, {detail, bubbles: true, composed: true}));
    }

    #openMenu(conversationId: string, last = false): void {
        this.openMenuId = conversationId;
        this.renameId = null;
        void this.#focusAfterRender('[role="menuitem"]:not([disabled])', conversationId, last);
    }

    #startRename(conversationId: string): void {
        this.openMenuId = null;
        this.renameId = conversationId;
        void this.updateComplete.then(() => {
            const input = this.#row(conversationId)?.querySelector('input');
            input?.focus();
            input?.select();
        });
    }

    #commitRename(conversation: ConversationSummary, input: HTMLInputElement): void {
        if (this.renameId !== conversation.conversation_id) return;
        this.renameId = null;
        const title = input.value.trim();
        if (!title || title === (conversation.title ?? '').trim()) return;
        this.#emit<ConversationRenameDetail>('conversation-rename', {
            conversationId: conversation.conversation_id,
            title,
        });
    }

    #renderStatus(message: string, retryLabel: string, kind: ConversationRetryDetail['kind']) {
        return html`
            <div class="conversation-list-status" role="status">
                <span>${message}</span>
                <button type="button" @click=${() => {
                    this.#emit<ConversationRetryDetail>('conversation-retry', {kind});
                }}>${retryLabel}</button>
            </div>
        `;
    }

    #renderRenameInput(conversation: ConversationSummary) {
        return html`
            <input
                type="text"
                aria-label="Conversation title"
                maxlength="120"
                .value=${conversation.title ?? ''}
                @keydown=${(event: KeyboardEvent) => {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        (event.currentTarget as HTMLInputElement).blur();
                    } else if (event.key === 'Escape') {
                        event.preventDefault();
                        event.stopPropagation();
                        this.renameId = null;
                    }
                }}
                @blur=${(event: FocusEvent) => {
                    this.#commitRename(conversation, event.currentTarget as HTMLInputElement);
                }}
            >
        `;
    }

    #renderMenu(conversation: ConversationSummary) {
        const conversationId = conversation.conversation_id;
        return html`
            <div
                class="conversation-actions-menu"
                role="menu"
                aria-label="Conversation actions"
                @keydown=${(event: KeyboardEvent) => {
                    const items = [...(event.currentTarget as HTMLElement)
                        .querySelectorAll<HTMLButtonElement>('[role="menuitem"]')]
                        .filter((item) => !item.disabled);
                    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                        event.preventDefault();
                        const index = items.indexOf(document.activeElement as HTMLButtonElement);
                        const delta = event.key === 'ArrowDown' ? 1 : -1;
                        items[(index + delta + items.length) % items.length]?.focus();
                    } else if (event.key === 'Escape') {
                        event.preventDefault();
                        event.stopPropagation();
                        this.closeMenu(true);
                    }
                }}
            >
                <button
                    type="button"
                    role="menuitem"
                    @click=${() => { this.#startRename(conversationId); }}
                >Rename</button>
                <button
                    type="button"
                    role="menuitem"
                    class="conversation-delete-action"
                    ?disabled=${this.busy}
                    @click=${() => {
                        this.openMenuId = null;
                        this.#emit<ConversationIntentDetail>('conversation-delete', {conversationId});
                    }}
                >Delete</button>
            </div>
        `;
    }

    #renderRow(conversation: ConversationSummary) {
        const conversationId = conversation.conversation_id;
        const active = conversationId === conversationStore.activeConversationId;
        const renaming = this.renameId === conversationId;
        const expanded = this.openMenuId === conversationId;
        return html`
            <div
                class="conversation-row"
                role="listitem"
                data-conversation-id=${conversationId}
                aria-current=${active ? 'page' : nothing}
            >
                ${renaming ? this.#renderRenameInput(conversation) : html`
                    <button
                        type="button"
                        class="conversation-select"
                        ?disabled=${this.busy}
                        aria-label=${conversation.title ? nothing : 'Open untitled conversation'}
                        @click=${() => {
                            this.#emit<ConversationIntentDetail>('conversation-select', {conversationId});
                        }}
                    >${conversation.title || 'New chat'}</button>
                `}
                <button
                    type="button"
                    class="conversation-actions-button"
                    aria-label="Conversation actions"
                    aria-haspopup="menu"
                    aria-expanded=${expanded ? 'true' : 'false'}
                    @click=${(event: MouseEvent) => {
                        event.stopPropagation();
                        if (expanded) this.closeMenu();
                        else this.#openMenu(conversationId);
                    }}
                    @keydown=${(event: KeyboardEvent) => {
                        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
                        event.preventDefault();
                        this.#openMenu(conversationId, event.key === 'ArrowUp');
                    }}
                >•••</button>
                ${expanded ? this.#renderMenu(conversation) : nothing}
            </div>
        `;
    }

    protected override render(): TemplateResult | TemplateResult[] {
        const conversations = conversationStore.conversations;
        if (this.listState === 'loading' && conversations.length === 0) {
            return html`
                ${Array.from({length: SKELETON_COUNT}, () => html`
                    <div class="conversation-skeleton" aria-hidden="true"></div>
                `)}
                <span class="sr-only">Loading conversations</span>
            `;
        }
        return html`
            ${this.listState === 'error'
                ? this.#renderStatus('Could not load conversations.', 'Retry', 'reload')
                : nothing}
            ${this.listState === 'empty-error'
                ? this.#renderStatus('No conversation is open.', 'Retry New chat', 'new')
                : nothing}
            ${repeat(
                conversations,
                (conversation) => conversation.conversation_id,
                (conversation) => this.#renderRow(conversation),
            )}
        `;
    }
}

customElements.define('conversation-list', ConversationList);

declare global {
    interface HTMLElementTagNameMap {
        'conversation-list': ConversationList;
    }

    interface HTMLElementEventMap {
        'conversation-select': CustomEvent<ConversationIntentDetail>;
        'conversation-delete': CustomEvent<ConversationIntentDetail>;
        'conversation-rename': CustomEvent<ConversationRenameDetail>;
        'conversation-retry': CustomEvent<ConversationRetryDetail>;
    }
}
