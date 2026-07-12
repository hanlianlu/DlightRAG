// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {ConversationHistory, ConversationSummary} from '../api/conversations.ts';
import {Store} from './base.ts';

const ACTIVE_KEY = 'dlightrag.active_conversation_id';

function storedActiveConversationId(): string | null {
  return typeof window === 'undefined' ? null : window.localStorage.getItem(ACTIVE_KEY);
}

export class ConversationStore extends Store {
  #conversations: ConversationSummary[] = [];
  #activeConversationId: string | null = storedActiveConversationId();
  #history: ConversationHistory | null = null;
  #generation = 0;

  get conversations(): readonly ConversationSummary[] {
    return this.#conversations;
  }

  get activeConversationId(): string | null {
    return this.#activeConversationId;
  }

  get history(): ConversationHistory | null {
    return this.#history;
  }

  get generation(): number {
    return this.#generation;
  }

  replaceList(conversations: ConversationSummary[]): void {
    this.#conversations = [...conversations];
    this.emit('conversationListChanged', {conversations: this.#conversations});
  }

  select(conversationId: string): void {
    if (this.#activeConversationId !== conversationId) this.#history = null;
    this.#activeConversationId = conversationId;
    window.localStorage.setItem(ACTIVE_KEY, conversationId);
    this.emit('conversationSelected', {conversationId});
  }

  setHistory(history: ConversationHistory, generation: number): boolean {
    if (
      generation !== this.#generation
      || history.conversation.conversation_id !== this.#activeConversationId
    ) {
      return false;
    }
    this.#history = history;
    this.upsertSummary(history.conversation);
    this.emit('conversationHistoryChanged', {history});
    return true;
  }

  upsertSummary(summary: ConversationSummary): void {
    const next = this.#conversations.filter(
      (conversation) => conversation.conversation_id !== summary.conversation_id,
    );
    next.push(summary);
    next.sort((left, right) => right.updated_at.localeCompare(left.updated_at));
    this.#conversations = next;
    this.emit('conversationListChanged', {conversations: this.#conversations});
  }

  remove(conversationId: string): boolean {
    const next = this.#conversations.filter(
      (conversation) => conversation.conversation_id !== conversationId,
    );
    if (next.length === this.#conversations.length) return false;
    this.#conversations = next;
    if (this.#activeConversationId === conversationId) {
      this.#activeConversationId = null;
      this.#history = null;
      this.#generation += 1;
      window.localStorage.removeItem(ACTIVE_KEY);
      this.emit('conversationSelected', {conversationId: null});
    }
    this.emit('conversationListChanged', {conversations: this.#conversations});
    return true;
  }

  beginRequest(): number {
    this.#generation += 1;
    return this.#generation;
  }
}

export const conversationStore = new ConversationStore();
