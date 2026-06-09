// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base';

const HISTORY_CAP = 100;

class ConversationStore extends Store {
  #history: Array<{ role: string; content: string }> = [];

  get historyWindow(): readonly Array<{ role: string; content: string }> {
    return this.#history;
  }

  append(query: string, answer: string, imageIds: string[]): void {
    let userContent = query;
    if (imageIds.length > 0) {
      userContent += `\n[attached images: ${imageIds.join(', ')}]`;
    }
    this.#history.push({ role: 'user', content: userContent });
    this.#history.push({ role: 'assistant', content: answer });
    if (this.#history.length > HISTORY_CAP) {
      this.#history.splice(0, this.#history.length - HISTORY_CAP);
    }
    this.emit('chatExchangeComplete', undefined);
  }
}

export const conversationStore = new ConversationStore();
