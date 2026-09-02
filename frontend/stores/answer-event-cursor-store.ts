// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

interface AnswerEventCursor {
  runId: string;
  lastSequence: number;
}

/** Per-conversation SSE cursors; answer submission ownership lives elsewhere. */
export class AnswerEventCursorStore {
  readonly #cursors = new Map<string, AnswerEventCursor>();

  trackRun(conversationId: string, runId: string): void {
    const existing = this.#cursors.get(conversationId);
    if (existing?.runId === runId) return;
    this.#cursors.set(conversationId, {runId, lastSequence: 0});
  }

  runId(conversationId: string): string | null {
    return this.#cursors.get(conversationId)?.runId ?? null;
  }

  lastSequence(conversationId: string, runId: string): number {
    const existing = this.#cursors.get(conversationId);
    return existing?.runId === runId ? existing.lastSequence : 0;
  }

  recordSequence(conversationId: string, runId: string, sequence: number): void {
    const existing = this.#cursors.get(conversationId);
    if (!existing || existing.runId !== runId || sequence <= existing.lastSequence) return;
    existing.lastSequence = sequence;
  }

  clear(conversationId: string): void {
    this.#cursors.delete(conversationId);
  }
}

export const answerEventCursorStore = new AnswerEventCursorStore();
