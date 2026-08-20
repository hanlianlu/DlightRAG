// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from './csrf';

export interface ConversationSummary {
  conversation_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationAttachmentReference {
  attachment_id: string;
  ordinal: number;
  kind: string;
  filename: string;
  mime_type: string;
  byte_size: number;
  url: string;
  thumbnail_url: string | null;
  label: string;
}

export type AnswerRunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface ConversationTurn {
  turn_id: string;
  turn_number: number;
  answer_run_id: string;
  submission_id: string;
  status: AnswerRunStatus;
  cancel_requested: boolean;
  user_text: string;
  assistant_text: string;
  user_attachments: ConversationAttachmentReference[];
  answer_html: string;
  primary_report: string | null;
  error_kind: string | null;
  error_message: string | null;
  created_at: string;
}

export interface AnswerRunDescriptor {
  run_id: string;
  status: AnswerRunStatus;
  cancel_requested: boolean;
  turn_id: string;
  turn_number: number;
  submission_id: string;
  events_url: string;
  status_url: string;
  cancel_url: string;
  conversation: ConversationSummary;
}

export interface ConversationHistory {
  conversation: ConversationSummary;
  turns: ConversationTurn[];
}

export class ConversationApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = 'ConversationApiError';
  }
}

async function responseJson<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) throw new ConversationApiError(response.status, fallback);
  return await response.json() as T;
}

export async function listConversations(signal?: AbortSignal): Promise<ConversationSummary[]> {
  const response = await fetch('/web/conversations', {signal});
  return responseJson<ConversationSummary[]>(response, 'Failed to load conversations');
}

export async function createConversation(signal?: AbortSignal): Promise<ConversationSummary> {
  const response = await fetch('/web/conversations', {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  return responseJson<ConversationSummary>(response, 'Failed to create conversation');
}

export async function getConversationHistory(
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationHistory> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/conversations/${id}/history`, {signal});
  return responseJson<ConversationHistory>(response, 'Failed to load conversation history');
}

export async function renameConversation(
  conversationId: string,
  title: string,
  signal?: AbortSignal,
): Promise<ConversationSummary> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/conversations/${id}`, {
    method: 'PATCH',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({title}),
    signal,
  });
  return responseJson<ConversationSummary>(response, 'Failed to rename conversation');
}

export async function deleteConversation(
  conversationId: string,
  signal?: AbortSignal,
): Promise<void> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/conversations/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) {
    throw new ConversationApiError(response.status, 'Failed to delete conversation');
  }
}

export async function deleteAllConversations(signal?: AbortSignal): Promise<void> {
  const response = await fetch('/web/conversations', {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) {
    throw new ConversationApiError(response.status, 'Failed to delete conversations');
  }
}

export async function getAnswerRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/answer/${id}`, {signal});
  return responseJson<ConversationTurn>(response, 'Failed to load answer run');
}

/** Ask the server to stop a run; disconnecting never does this on its own. */
export async function cancelAnswerRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/answer/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  return responseJson<ConversationTurn>(response, 'Failed to stop the answer');
}
