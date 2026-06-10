// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import { Store } from './base';
import { sessionStore } from './sessionStore';

const HISTORY_CAP = 100;

type ContentBlock = { type: string; text?: string; image_url?: { url: string } };
type MessageContent = string | ContentBlock[];

class ConversationStore extends Store {
  #history: Array<{ role: string; content: MessageContent }> = [];
  #loaded = false;

  get historyWindow(): ReadonlyArray<{ role: string; content: MessageContent }> {
    return this.#history;
  }

  append(
    query: string,
    answer: string,
    imageIds: string[],
    queryImages?: string[],
  ): void {
    let userContent: MessageContent;
    if (queryImages && queryImages.length > 0) {
      const blocks: ContentBlock[] = [
        { type: 'text', text: query },
      ];
      for (const dataUri of queryImages) {
        blocks.push({
          type: 'image_url',
          image_url: { url: dataUri },
        });
      }
      userContent = blocks;
    } else {
      userContent = query;
    }
    this.#history.push({ role: 'user', content: userContent });
    this.#history.push({ role: 'assistant', content: answer });
    if (this.#history.length > HISTORY_CAP) {
      this.#history.splice(0, this.#history.length - HISTORY_CAP);
    }
    this.emit('chatExchangeComplete', undefined);
  }

  async restoreFromCheckpoint(): Promise<void> {
    if (this.#loaded || this.#history.length > 0) return;
    this.#loaded = true;

    const sid = sessionStore.sessionId;
    if (!sid) return;

    try {
      const response = await fetch(`/web/history?session_id=${encodeURIComponent(sid)}`);
      if (!response.ok) return;
      const data = await response.json();
      const history: Array<{ role: string; content: MessageContent }> =
        data?.history || [];
      if (history.length > 0) {
        this.#history = history;
        this.emit('chatHistoryRestored', history.length);
      }
    } catch {
      // best-effort — page works fine without history
    }
  }
}

export const conversationStore = new ConversationStore();
