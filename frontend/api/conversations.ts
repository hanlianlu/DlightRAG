// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from './csrf.ts';

export interface ConversationSummary {
  conversation_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  forked_from_conversation_id?: string | null;
  forked_from_title?: string | null;
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

export interface PresentationImage {
  id: string;
  chunk_id: string;
  source_ref: string;
  url: string;
  thumbnail_url: string;
  label: string;
  answer_image_sent: boolean;
}

export interface PresentationSourceChunk {
  chunk_idx: number | null;
  page_number: number | null;
  content_html: string;
  image_url: string | null;
  thumbnail_url: string | null;
}

export interface PresentationSource {
  id: string;
  title: string;
  source_url: string | null;
  download_url: string | null;
  chunks: PresentationSourceChunk[];
}

export interface AnswerPresentation {
  answer_text: string;
  answer_html: string;
  sources: PresentationSource[];
  answer_images: PresentationImage[];
  primary_report?: string | null;
}

export interface AgentChildStatus {
  child_session_id: string;
  status: string;
  objective?: string;
  model_role?: string;
  usage?: Record<string, number> | null;
}

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
  presentation: AnswerPresentation | null;
  usage: Record<string, unknown>;
  evidence: Record<string, number>;
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
  parent_run_id?: string | null;
  continuation_kind?: string | null;
  conversation: ConversationSummary;
}

export interface ConversationHistory {
  conversation: ConversationSummary;
  turns: ConversationTurn[];
}

export class ConversationApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ConversationApiError';
    this.status = status;
  }
}

async function responseJson<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) throw new ConversationApiError(response.status, fallback);
  return await response.json() as T;
}

export async function listConversations(signal?: AbortSignal): Promise<ConversationSummary[]> {
  const response = await fetch('/web/api/conversations', {signal});
  return responseJson<ConversationSummary[]>(response, 'Failed to load conversations');
}

export async function createConversation(signal?: AbortSignal): Promise<ConversationSummary> {
  const response = await fetch('/web/api/conversations', {
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
  const response = await fetch(`/web/api/conversations/${id}/history`, {signal});
  return responseJson<ConversationHistory>(response, 'Failed to load conversation history');
}

export async function renameConversation(
  conversationId: string,
  title: string,
  signal?: AbortSignal,
): Promise<ConversationSummary> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}`, {
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
  const response = await fetch(`/web/api/conversations/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) {
    throw new ConversationApiError(response.status, 'Failed to delete conversation');
  }
}

export async function deleteAllConversations(signal?: AbortSignal): Promise<void> {
  const response = await fetch('/web/api/conversations', {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) {
    throw new ConversationApiError(response.status, 'Failed to delete conversations');
  }
}

export async function getAnswerReport(
  runId: string,
  signal?: AbortSignal,
): Promise<AnswerPresentation> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/report`, {signal});
  return responseJson<AnswerPresentation>(response, 'Failed to load the report');
}

export async function getAnswerRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}`, {signal});
  return responseJson<ConversationTurn>(response, 'Failed to load answer run');
}

/** Ask the server to stop a run; disconnecting never does this on its own. */
export async function steerAnswerRun(
  runId: string,
  instruction: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/steer`, {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({content: instruction}),
    signal,
  });
  return responseJson<Record<string, unknown>>(response, 'Failed to steer the answer');
}

export async function getAnswerRunChildren(
  runId: string,
  signal?: AbortSignal,
): Promise<AgentChildStatus[]> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/children`, {signal});
  const payload = await responseJson<{children: AgentChildStatus[]}>(
    response,
    'Failed to load child agents',
  );
  return payload.children;
}

export async function continueAnswerRun(
  runId: string,
  operation: 'follow-up' | 'fork',
  content: string,
  submissionId: string,
  signal?: AbortSignal,
): Promise<AnswerRunDescriptor> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/${operation}`, {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({content, submission_id: submissionId}),
    signal,
  });
  return responseJson<AnswerRunDescriptor>(response, `Failed to ${operation} the answer`);
}

export async function resumeAnswerRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/resume`, {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  return responseJson<ConversationTurn>(response, 'Failed to resume the answer');
}

export async function cancelAnswerRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  return responseJson<ConversationTurn>(response, 'Failed to stop the answer');
}
