// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Per-conversation memory of the answer run this tab submitted: the submission
// id that makes a resend idempotent, the run it produced, and the last durable
// sequence this tab consumed so a reconnect resumes without a gap or a
// duplicate. Everything here is derived from the server's authoritative run;
// nothing is a cached answer.

interface AnswerRunProgress {
  submissionId: string;
  payloadFingerprint: string;
  runId: string | null;
  lastSequence: number;
}

export class AnswerRunStore {
  private readonly runs = new Map<string, AnswerRunProgress>();
  private readonly createId: () => string;

  constructor(createId: () => string = () => crypto.randomUUID()) {
    this.createId = createId;
  }

  /** Reuse the submission id of an identical unfinished payload, else mint one. */
  getOrCreateSubmissionId(conversationId: string, payloadFingerprint: string): string {
    const existing = this.runs.get(conversationId);
    if (existing?.payloadFingerprint === payloadFingerprint) return existing.submissionId;

    const submissionId = this.createId();
    this.runs.set(conversationId, {
      submissionId,
      payloadFingerprint,
      runId: null,
      lastSequence: 0,
    });
    return submissionId;
  }

  /** Bind the accepted run to this conversation, restarting its cursor. */
  attachRun(conversationId: string, runId: string): void {
    const existing = this.runs.get(conversationId);
    if (!existing) return;
    if (existing.runId !== runId) existing.lastSequence = 0;
    existing.runId = runId;
  }

  /** Track a run this tab discovered from history rather than from a submission. */
  trackRun(conversationId: string, runId: string): void {
    const existing = this.runs.get(conversationId);
    if (existing?.runId === runId) return;
    this.runs.set(conversationId, {
      submissionId: '',
      payloadFingerprint: '',
      runId,
      lastSequence: 0,
    });
  }

  runId(conversationId: string): string | null {
    return this.runs.get(conversationId)?.runId ?? null;
  }

  /** The durable sequence to resume after; only ever moves forward. */
  lastSequence(conversationId: string, runId: string): number {
    const existing = this.runs.get(conversationId);
    return existing && existing.runId === runId ? existing.lastSequence : 0;
  }

  recordSequence(conversationId: string, runId: string, sequence: number): void {
    const existing = this.runs.get(conversationId);
    if (!existing || existing.runId !== runId) return;
    if (sequence > existing.lastSequence) existing.lastSequence = sequence;
  }

  clear(conversationId: string): void {
    this.runs.delete(conversationId);
  }
}

export async function payloadFingerprint(value: unknown): Promise<string> {
  const encoded = new TextEncoder().encode(JSON.stringify(value));
  const digest = await crypto.subtle.digest('SHA-256', encoded);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export const answerRunStore = new AnswerRunStore();
