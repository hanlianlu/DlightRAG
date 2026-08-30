// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {ConversationSummary} from '../api/conversations.ts';
import {LightElement, StoreController} from '../lib/lit_host.ts';
import {conversationStore} from '../stores/conversationStore.ts';

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

/** Conversation rows, row accessibility, and item intent. */
export class DlConversationList extends LightElement {
  static properties = {
    busy: {attribute: false},
    openMenuId: {state: true},
    renameId: {state: true},
  };

  declare busy: boolean;
  declare openMenuId: string | null;
  declare renameId: string | null;

  #dismiss: AbortController | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.busy = false;
    this.openMenuId = null;
    this.renameId = null;
    /** Store reads: conversations, listState, loadMoreState, hasOlderConversations, activeConversationId. */
    new StoreController(this, conversationStore);
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

  get menuOpen(): boolean {
    return this.openMenuId !== null;
  }

  closeMenu(restoreFocus = false): void {
    const conversationId = this.openMenuId;
    if (conversationId === null) return;
    this.openMenuId = null;
    if (restoreFocus) void this.focusActions(conversationId);
  }

  async focusConversation(conversationId: string): Promise<boolean> {
    return this.#focusAfterRender('.conversation-select', conversationId);
  }

  async focusActions(conversationId: string): Promise<boolean> {
    return this.#focusAfterRender('.conversation-actions-button', conversationId);
  }

  #row(conversationId: string): HTMLElement | null {
    return this.querySelector<HTMLElement>(
      `[data-conversation-id="${CSS.escape(conversationId)}"]`,
    );
  }

  async #focusAfterRender(
    selector: string,
    conversationId: string,
    last = false,
  ): Promise<boolean> {
    await this.updateComplete;
    const matches = this.#row(conversationId)?.querySelectorAll<HTMLElement>(selector);
    if (!matches?.length) return false;
    matches[last ? matches.length - 1 : 0].focus();
    return true;
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
    this.#emit<ConversationRenameDetail>('dl-conversation-rename', {
      conversationId: conversation.conversation_id,
      title,
    });
  }

  #finishKeyboardRename(
    conversation: ConversationSummary,
    input: HTMLInputElement,
    commit: boolean,
  ): void {
    if (commit) this.#commitRename(conversation, input);
    else this.renameId = null;
    void this.focusActions(conversation.conversation_id);
  }

  #renderStatus(
    message: string,
    retryLabel: string,
    kind: ConversationRetryDetail['kind'],
  ): TemplateResult {
    return html`
      <div class="conversation-list-status" role="status">
        <span>${message}</span>
        <button type="button" @click=${() => {
          this.#emit<ConversationRetryDetail>('dl-conversation-retry', {kind});
        }}>${retryLabel}</button>
      </div>
    `;
  }

  #renderRenameInput(conversation: ConversationSummary): TemplateResult {
    return html`
      <input
        type="text"
        aria-label=${msg('Conversation title', {id: 'conversationList.conversationTitle'})}
        maxlength="120"
        .value=${conversation.title ?? ''}
        @keydown=${(event: KeyboardEvent) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            this.#finishKeyboardRename(
              conversation,
              event.currentTarget as HTMLInputElement,
              true,
            );
          } else if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            this.#finishKeyboardRename(
              conversation,
              event.currentTarget as HTMLInputElement,
              false,
            );
          }
        }}
        @blur=${(event: FocusEvent) => {
          this.#commitRename(conversation, event.currentTarget as HTMLInputElement);
        }}
      >
    `;
  }

  #renderMenu(conversation: ConversationSummary): TemplateResult {
    const conversationId = conversation.conversation_id;
    return html`
      <div
        class="conversation-actions-menu"
        role="menu"
        aria-label=${msg('Conversation actions', {id: 'conversationList.conversationActions'})}
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
        >${msg('Rename', {id: 'conversationList.rename'})}</button>
        <button
          type="button"
          role="menuitem"
          class="conversation-delete-action"
          ?disabled=${this.busy}
          @click=${() => {
            this.openMenuId = null;
            this.#emit<ConversationIntentDetail>('dl-conversation-delete', {conversationId});
          }}
        >${msg('Delete', {id: 'conversationList.delete'})}</button>
      </div>
    `;
  }

  #renderRow(conversation: ConversationSummary): TemplateResult {
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
        <div class="conversation-row-main">
          ${renaming ? this.#renderRenameInput(conversation) : html`
            <button
              type="button"
              class="conversation-select"
              ?disabled=${this.busy}
              aria-label=${conversation.title
                ? nothing
                : msg('Open untitled conversation', {id: 'conversationList.openUntitled'})}
              @click=${() => {
                this.#emit<ConversationIntentDetail>('dl-conversation-select', {conversationId});
              }}
            >${conversation.title || msg('New chat', {id: 'conversationList.newChat'})}</button>
          `}
          ${conversation.forked_from_title ? html`
            <span
              class="conversation-lineage"
              title=${msg('Forked from another conversation', {id: 'conversationList.forkedFromTitle'})}
            >
              ${msg(str`Forked from ${conversation.forked_from_title}`, {id: 'conversationList.forkedFrom'})}
            </span>
          ` : nothing}
        </div>
        <button
          type="button"
          class="conversation-actions-button"
          aria-label=${msg('Conversation actions', {id: 'conversationList.conversationActionsButton'})}
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
    const listState = conversationStore.listState;
    if (listState === 'loading' && conversations.length === 0) {
      return html`
        ${Array.from({length: SKELETON_COUNT}, () => html`
          <div class="conversation-skeleton" aria-hidden="true"></div>
        `)}
        <span class="sr-only">${msg('Loading conversations', {id: 'conversationList.loadingConversations'})}</span>
      `;
    }
    return html`
      ${listState === 'error'
        ? this.#renderStatus(
            msg('Could not load conversations.', {id: 'conversationList.couldNotLoad'}),
            msg('Retry', {id: 'conversationList.retry'}),
            'reload',
          )
        : nothing}
      ${listState === 'empty-error'
        ? this.#renderStatus(
            msg('No conversation is open.', {id: 'conversationList.noConversationOpen'}),
            msg('Retry New chat', {id: 'conversationList.retryNewChat'}),
            'new',
          )
        : nothing}
      <div class="conversation-items" role="list" aria-live="polite">
        ${repeat(
          conversations,
          (conversation) => conversation.conversation_id,
          (conversation) => this.#renderRow(conversation),
        )}
      </div>
      ${conversationStore.loadMoreState === 'error'
        ? html`
          <div class="conversation-list-status" role="status">
            <span>${msg('Could not load older conversations.', {id: 'conversationList.couldNotLoadOlder'})}</span>
            <button type="button" @click=${() => {
              void conversationStore.loadOlder();
            }}>${msg('Retry loading older conversations', {id: 'conversationList.retryLoadOlder'})}</button>
          </div>
        `
        : nothing}
      ${conversationStore.hasOlderConversations && conversationStore.loadMoreState !== 'error'
        ? html`
          <button
            type="button"
            class="conversation-load-older"
            ?disabled=${conversationStore.loadMoreState === 'loading'}
            aria-label=${msg('Load older conversations', {id: 'conversationList.loadOlderAria'})}
            @click=${() => { void conversationStore.loadOlder(); }}
          >${conversationStore.loadMoreState === 'loading'
            ? msg('Loading older…', {id: 'conversationList.loadingOlder'})
            : msg('Load older', {id: 'conversationList.loadOlder'})}</button>
        `
        : nothing}
    `;
  }
}

customElements.define('dl-conversation-list', DlConversationList);

declare global {
  interface HTMLElementTagNameMap {
    'dl-conversation-list': DlConversationList;
  }

  interface HTMLElementEventMap {
    'dl-conversation-select': CustomEvent<ConversationIntentDetail>;
    'dl-conversation-delete': CustomEvent<ConversationIntentDetail>;
    'dl-conversation-rename': CustomEvent<ConversationRenameDetail>;
    'dl-conversation-retry': CustomEvent<ConversationRetryDetail>;
  }
}
