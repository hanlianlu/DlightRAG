// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const ACTIVE_CONVERSATION_KEY = 'dlightrag.active_conversation_id';

interface ConversationSummary {
  conversation_id: string;
}

export type ConversationStatus = 'idle' | 'loading' | 'ready' | 'unavailable';

export class ConversationStore {
  #activeConversationId: string | null = null;
  #initializing: Promise<string | null> | null = null;
  #status: ConversationStatus = 'idle';
  #errorMessage: string | null = null;

  get activeConversationId(): string | null {
    return this.#activeConversationId;
  }

  get status(): ConversationStatus {
    return this.#status;
  }

  get errorMessage(): string | null {
    return this.#errorMessage;
  }

  async initialize(): Promise<string | null> {
    return this.ensureActive();
  }

  async ensureActive(): Promise<string | null> {
    if (this.#activeConversationId) return this.#activeConversationId;
    if (this.#initializing) return this.#initializing;
    this.#status = 'loading';
    this.#errorMessage = null;
    this.#initializing = this.#tryLoadOrCreateActive();
    try {
      return await this.#initializing;
    } finally {
      this.#initializing = null;
    }
  }

  async #tryLoadOrCreateActive(): Promise<string | null> {
    try {
      const conversationId = await this.#loadOrCreateActive();
      this.#status = 'ready';
      return conversationId;
    } catch {
      this.#status = 'unavailable';
      this.#errorMessage = 'Conversation service is unavailable. Please try again.';
      return null;
    }
  }

  async #loadOrCreateActive(): Promise<string> {
    const response = await fetch('/web/conversations');
    if (!response.ok) throw new Error('Failed to load conversations');
    const conversations = await response.json() as ConversationSummary[];
    const stored = window.localStorage.getItem(ACTIVE_CONVERSATION_KEY);
    const selected = conversations.find((item) => item.conversation_id === stored)
      || conversations[0];
    if (selected) return this.#select(selected.conversation_id);

    const created = await fetch('/web/conversations', { method: 'POST' });
    if (!created.ok) throw new Error('Failed to create conversation');
    const summary = await created.json() as ConversationSummary;
    return this.#select(summary.conversation_id);
  }

  #select(conversationId: string): string {
    this.#activeConversationId = conversationId;
    this.#errorMessage = null;
    window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId);
    return conversationId;
  }
}

export const conversationStore = new ConversationStore();
