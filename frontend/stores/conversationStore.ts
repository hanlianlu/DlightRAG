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
  /**
   * Monotonic request counter and the store's concurrency guard. Bumped at
   * every lifecycle boundary (`beginRequest`, `beginLiveAnswer`,
   * `finishLiveAnswer`, and active-conversation `remove`). Async callers
   * capture the value they started with and re-check it via `isCurrentRequest`
   * / `canRenderHistory`, so results from a superseded request are dropped
   * instead of clobbering newer state.
   */
  #generation = 0;
  /**
   * Conversation currently receiving a streamed answer, or `null` when idle.
   * While non-null it acts as a lock: `select()` of a *different* conversation
   * is deferred into `#pendingSelectionId` instead of switching, and
   * `canRenderHistory` returns false so a history load cannot overwrite the
   * in-progress answer. Cleared only by `finishLiveAnswer`.
   */
  #liveAnswerConversationId: string | null = null;
  /**
   * A selection requested *during* a live answer, held until it ends.
   * `finishLiveAnswer` returns it (unless discarded) so the caller can apply
   * the deferred switch; also cleared if its target conversation is removed.
   */
  #pendingSelectionId: string | null = null;
  /**
   * Whether `#activeConversationId`'s history is loaded and thus answerable.
   * Gates `answerConversationId`; set true by `setHistory`, reset to false when
   * switching conversations (before history arrives) or when the active
   * conversation is removed.
   */
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

  initialSelection(): ConversationSummary | null {
    const stored = this.#activeConversationId;
    return this.#conversations.find(
      (conversation) => conversation.conversation_id === stored,
    ) ?? this.#conversations[0] ?? null;
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

  isCurrentRequest(generation: number): boolean {
    return generation === this.#generation;
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
