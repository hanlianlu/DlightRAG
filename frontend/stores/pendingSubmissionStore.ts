// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export interface ConversationSaveOutcome {
  conversation_saved?: boolean;
  conversation_save_reason?: string | null;
}

export interface ConversationSavePresentation {
  state: 'saved' | 'not_saved' | 'unknown';
  message: string | null;
  actionLabel: string | null;
}

interface PendingSubmission {
  submissionId: string;
  payloadFingerprint: string;
  warningDelivered: boolean;
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
    this.pending.set(conversationId, {
      submissionId,
      payloadFingerprint,
      warningDelivered: false,
    });
    return submissionId;
  }

  claimWarningDelivery(conversationId: string, submissionId: string): boolean {
    const pending = this.pending.get(conversationId);
    if (!pending || pending.submissionId !== submissionId || pending.warningDelivered) return false;
    pending.warningDelivered = true;
    return true;
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

export function describeConversationSaveOutcome(
  outcome: ConversationSaveOutcome,
): ConversationSavePresentation {
  if (outcome.conversation_saved === true) {
    return {state: 'saved', message: null, actionLabel: null};
  }

  const reason = outcome.conversation_save_reason;
  if (reason === 'conversation_changed') {
    return {
      state: 'not_saved',
      message: 'This answer was not saved because the conversation changed.',
      actionLabel: null,
    };
  }
  if (reason === 'commit_not_found') {
    return {
      state: 'not_saved',
      message: 'This answer was not saved because no committed turn was found.',
      actionLabel: null,
    };
  }
  if (reason === 'persistence_failed') {
    return {
      state: 'not_saved',
      message: 'This answer was not saved because conversation storage failed.',
      actionLabel: null,
    };
  }
  if (reason === 'storage_unavailable') {
    return {
      state: 'unknown',
      message: 'Save status could not be confirmed because conversation storage is unavailable.',
      actionLabel: 'Check save status',
    };
  }
  return {
    state: 'unknown',
    message: 'Save status is unknown. The answer may already be saved; checking history may reconcile it.',
    actionLabel: 'Check save status',
  };
}

export function shouldKeepLiveConversation<T extends ConversationSaveOutcome>(
  outcome: T | null,
): outcome is T & {conversation_saved: true} {
  return outcome?.conversation_saved === true;
}

export const pendingSubmissionStore = new PendingSubmissionStore();
