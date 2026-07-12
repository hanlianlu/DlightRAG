// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base';

const ACTIVE_CONVERSATION_KEY = 'dlightrag.active_conversation_id';

interface ConversationSummary {
  conversation_id: string;
}

class ConversationStore extends Store {
  #activeConversationId: string | null = null;
  #initializing: Promise<string> | null = null;

  get activeConversationId(): string | null {
    return this.#activeConversationId;
  }

  async initialize(): Promise<string> {
    return this.ensureActive();
  }

  async ensureActive(): Promise<string> {
    if (this.#activeConversationId) return this.#activeConversationId;
    if (this.#initializing) return this.#initializing;
    this.#initializing = this.#loadOrCreateActive();
    try {
      return await this.#initializing;
    } finally {
      this.#initializing = null;
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
    window.localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId);
    return conversationId;
  }
}

export const conversationStore = new ConversationStore();
