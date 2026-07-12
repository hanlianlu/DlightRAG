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
  #liveAnswerConversationId: string | null = null;
  #pendingSelectionId: string | null = null;
  #answerReady = this.#activeConversationId !== null;

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

  get answerConversationId(): string | null {
    return this.#answerReady ? this.#activeConversationId : null;
  }

  replaceList(conversations: ConversationSummary[]): void {
    this.#conversations = [...conversations];
    this.emit('conversationListChanged', {conversations: this.#conversations});
  }

  select(conversationId: string): boolean {
    if (
      this.#liveAnswerConversationId !== null
      && this.#liveAnswerConversationId !== conversationId
    ) {
      this.#pendingSelectionId = conversationId;
      return false;
    }
    if (this.#activeConversationId !== conversationId) {
      this.#history = null;
      this.#answerReady = false;
    }
    this.#activeConversationId = conversationId;
    window.localStorage.setItem(ACTIVE_KEY, conversationId);
    this.emit('conversationSelected', {conversationId});
    return true;
  }

  setHistory(history: ConversationHistory, generation: number): boolean {
    if (
      !this.canRenderHistory(generation)
      || history.conversation.conversation_id !== this.#activeConversationId
    ) {
      return false;
    }
    this.#history = history;
    this.#answerReady = true;
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
      this.#answerReady = false;
      window.localStorage.removeItem(ACTIVE_KEY);
      this.emit('conversationSelected', {conversationId: null});
    }
    if (this.#pendingSelectionId === conversationId) this.#pendingSelectionId = null;
    this.emit('conversationListChanged', {conversations: this.#conversations});
    return true;
  }

  beginRequest(): number {
    this.#generation += 1;
    return this.#generation;
  }

  beginLiveAnswer(conversationId: string): number {
    this.#liveAnswerConversationId = conversationId;
    this.#generation += 1;
    return this.#generation;
  }

  finishLiveAnswer(conversationId: string, discardPendingSelection = false): string | null {
    if (this.#liveAnswerConversationId !== conversationId) return null;
    this.#liveAnswerConversationId = null;
    this.#generation += 1;
    const pendingSelectionId = this.#pendingSelectionId;
    this.#pendingSelectionId = null;
    return discardPendingSelection ? null : pendingSelectionId;
  }

  canRenderHistory(generation: number): boolean {
    return this.#liveAnswerConversationId === null && generation === this.#generation;
  }
}

export const conversationStore = new ConversationStore();
