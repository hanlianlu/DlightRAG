// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export interface ConversationSummary {
  conversation_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationImageReference {
  image_id: string;
  ordinal: number;
  mime_type: string;
  url: string;
  thumbnail_url: string;
  label: string;
}

export interface ConversationTurn {
  turn_id: string;
  turn_number: number;
  user_text: string;
  assistant_text: string;
  user_images: ConversationImageReference[];
  answer_sources: Record<string, unknown>;
  answer_html: string;
  queried_workspaces: string[];
  created_at: string;
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
  const response = await fetch('/web/conversations', {method: 'POST', signal});
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
    headers: {'Content-Type': 'application/json'},
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
  const response = await fetch(`/web/conversations/${id}`, {method: 'DELETE', signal});
  if (!response.ok) {
    throw new ConversationApiError(response.status, 'Failed to delete conversation');
  }
}
