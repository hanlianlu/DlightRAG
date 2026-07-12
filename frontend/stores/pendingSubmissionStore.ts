// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export interface ConversationSaveOutcome {
  conversation_saved?: boolean;
  conversation_save_reason?: string | null;
}

interface PendingSubmission {
  submissionId: string;
  payloadFingerprint: string;
}

const DEFINITIVE_NON_COMMIT_REASONS = new Set([
  'conversation_changed',
  'commit_not_found',
  'persistence_failed',
]);

export class PendingSubmissionStore {
  private readonly pending = new Map<string, PendingSubmission>();
  private readonly createId: () => string;

  constructor(createId: () => string = () => crypto.randomUUID()) {
    this.createId = createId;
  }

  getOrCreate(conversationId: string, payloadFingerprint: string): string {
    const existing = this.pending.get(conversationId);
    if (existing?.payloadFingerprint === payloadFingerprint) return existing.submissionId;

    const submissionId = this.createId();
    this.pending.set(conversationId, {submissionId, payloadFingerprint});
    return submissionId;
  }

  clear(conversationId: string): void {
    this.pending.delete(conversationId);
  }
}

export async function payloadFingerprint(value: unknown): Promise<string> {
  const encoded = new TextEncoder().encode(JSON.stringify(value));
  const digest = await crypto.subtle.digest('SHA-256', encoded);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export function isDefinitiveSaveOutcome(outcome: ConversationSaveOutcome | null): boolean {
  if (!outcome) return false;
  if (outcome.conversation_saved === true) return true;
  const reason = outcome.conversation_save_reason;
  return typeof reason === 'string' && DEFINITIVE_NON_COMMIT_REASONS.has(reason);
}

export const pendingSubmissionStore = new PendingSubmissionStore();
