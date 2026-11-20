// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What this page last saw the server keep of the reader's judgements, message by message.
 *
 * A turn, a roster row, and a child's page each state a judgement as it stood when they were read, and none
 * of them is read again when the reader rates: a settled child's page is kept for as long as its dock lives,
 * and a turn that scrolls out of the window is drawn anew from the history it came from. So the page
 * remembers its own writes: the page's own last write takes precedence over a seed that may be older.
 *
 * This is not a copy of the server's state. The page learns only what its own requests were answered, so a
 * judgement changed from another tab or device shows here only after a reload, which starts from the server's
 * word again. Signing out forgets every judgement, so the next reader on this tab is shown none of them.
 */

import type {AnswerJudgement} from '../api/feedback.ts';

/** Names one message: the Run, the agent that wrote it, and its place in that agent's Session. */
export function judgementKey(runId: string, agent: string, sequence: number): string {
  return JSON.stringify([runId, agent, sequence]);
}

export class KeptJudgements {
  readonly #kept = new Map<string, AnswerJudgement | null>();

  /** The judgement to show: the one this page last saw kept for the message, else the one that was read. */
  current(key: string, read: AnswerJudgement | null): AnswerJudgement | null {
    return this.#kept.has(key) ? this.#kept.get(key) ?? null : read;
  }

  /** Remember what the server answered: the judgement it kept, or null once the reader took it back. */
  keep(key: string, judgement: AnswerJudgement | null): void {
    this.#kept.set(key, judgement);
  }

  /** Forget every judgement, as signing out must: whoever signs in next on this tab has none of them. */
  forget(): void {
    this.#kept.clear();
  }
}
