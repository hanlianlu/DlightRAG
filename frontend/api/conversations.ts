// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

const conversationSummary = v.pipe(
  v.object({
    conversation_id: v.string(),
    title: v.nullable(v.string()),
    created_at: v.string(),
    updated_at: v.string(),
    forked_from_conversation_id: v.optional(v.nullable(v.string())),
    forked_from_title: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    conversationId: w.conversation_id,
    title: w.title,
    createdAt: w.created_at,
    updatedAt: w.updated_at,
    forkedFromConversationId: w.forked_from_conversation_id ?? null,
    forkedFromTitle: w.forked_from_title ?? null,
  })),
);
export type ConversationSummary = v.InferOutput<typeof conversationSummary>;

const attachmentReference = v.pipe(
  v.object({
    attachment_id: v.string(),
    ordinal: v.number(),
    kind: v.string(),
    filename: v.string(),
    mime_type: v.string(),
    byte_size: v.number(),
    url: v.string(),
    thumbnail_url: v.nullable(v.string()),
    label: v.string(),
  }),
  v.transform((w) => ({
    attachmentId: w.attachment_id,
    ordinal: w.ordinal,
    kind: w.kind,
    filename: w.filename,
    mimeType: w.mime_type,
    byteSize: w.byte_size,
    url: w.url,
    thumbnailUrl: w.thumbnail_url,
    label: w.label,
  })),
);
export type ConversationAttachmentReference = v.InferOutput<typeof attachmentReference>;

const presentationImage = v.pipe(
  v.object({
    id: v.string(),
    chunk_id: v.string(),
    source_ref: v.string(),
    url: v.string(),
    thumbnail_url: v.string(),
    label: v.string(),
    answer_image_sent: v.boolean(),
  }),
  v.transform((w) => ({
    id: w.id,
    chunkId: w.chunk_id,
    sourceRef: w.source_ref,
    url: w.url,
    thumbnailUrl: w.thumbnail_url,
    label: w.label,
    answerImageSent: w.answer_image_sent,
  })),
);
export type PresentationImage = v.InferOutput<typeof presentationImage>;

const artifactIssue = v.object({
  kind: v.string(),
  description: v.string(),
  resource_id: v.nullable(v.string()),
});
export type ArtifactIssue = v.InferOutput<typeof artifactIssue>;

const artifactOutcome = v.pipe(
  v.object({
    status: v.picklist(['complete', 'partial', 'failed']),
    issues: v.array(artifactIssue),
  }),
  v.transform((w) => ({
    status: w.status,
    issues: w.issues.map((issue) => ({
      kind: issue.kind,
      description: issue.description,
      resourceId: issue.resource_id,
    })),
  })),
);
export type ArtifactOutcome = v.InferOutput<typeof artifactOutcome>;

const answerArtifact = v.pipe(
  v.object({
    resource_id: v.string(),
    media_type: v.string(),
    label: v.string(),
    filename: v.string(),
    byte_size: v.number(),
    digest: v.string(),
    presentation: v.picklist(['image', 'markdown', 'html', 'pdf', 'text', 'download']),
    status: v.picklist(['available', 'unavailable']),
    uri: v.string(),
    width: v.nullable(v.number()),
    height: v.nullable(v.number()),
    data_url: v.nullable(v.string()),
    download_url: v.nullable(v.string()),
    presentation_url: v.nullable(v.string()),
    issue: v.nullable(artifactIssue),
  }),
  v.transform((w) => ({
    resourceId: w.resource_id,
    mediaType: w.media_type,
    label: w.label,
    filename: w.filename,
    byteSize: w.byte_size,
    digest: w.digest,
    presentation: w.presentation,
    status: w.status,
    uri: w.uri,
    width: w.width,
    height: w.height,
    dataUrl: w.data_url,
    downloadUrl: w.download_url,
    presentationUrl: w.presentation_url,
    issue: w.issue === null ? null : {
      kind: w.issue.kind,
      description: w.issue.description,
      resourceId: w.issue.resource_id,
    },
  })),
);
export type AnswerArtifact = v.InferOutput<typeof answerArtifact>;

const presentationPart = v.pipe(
  v.object({
    type: v.picklist(['markdown', 'artifact', 'evidence_image']),
    text: v.string(),
    html: v.string(),
    artifact: v.nullable(answerArtifact),
    evidence_image: v.nullable(presentationImage),
    inline: v.boolean(),
  }),
  v.transform((w) => ({
    type: w.type,
    text: w.text,
    html: w.html,
    artifact: w.artifact,
    evidenceImage: w.evidence_image,
    inline: w.inline,
  })),
);
export type PresentationPart = v.InferOutput<typeof presentationPart>;

const presentationSourceChunk = v.pipe(
  v.object({
    chunk_idx: v.nullable(v.number()),
    page_number: v.nullable(v.number()),
    content_html: v.string(),
    image_url: v.nullable(v.string()),
    thumbnail_url: v.nullable(v.string()),
  }),
  v.transform((w) => ({
    chunkIdx: w.chunk_idx,
    pageNumber: w.page_number,
    contentHtml: w.content_html,
    imageUrl: w.image_url,
    thumbnailUrl: w.thumbnail_url,
  })),
);
export type PresentationSourceChunk = v.InferOutput<typeof presentationSourceChunk>;

const presentationSource = v.pipe(
  v.object({
    id: v.string(),
    title: v.string(),
    source_url: v.nullable(v.string()),
    download_url: v.nullable(v.string()),
    chunks: v.array(presentationSourceChunk),
  }),
  v.transform((w) => ({
    id: w.id,
    title: w.title,
    sourceUrl: w.source_url,
    downloadUrl: w.download_url,
    chunks: w.chunks,
  })),
);
export type PresentationSource = v.InferOutput<typeof presentationSource>;

const answerPresentation = v.pipe(
  v.object({
    answer_text: v.string(),
    parts: v.array(presentationPart),
    sources: v.array(presentationSource),
    evidence_images: v.array(presentationImage),
    artifacts: v.array(answerArtifact),
    artifact_outcome: artifactOutcome,
  }),
  v.transform((w) => ({
    answerText: w.answer_text,
    parts: w.parts,
    sources: w.sources,
    evidenceImages: w.evidence_images,
    artifacts: w.artifacts,
    artifactOutcome: w.artifact_outcome,
  })),
);
export type AnswerPresentation = v.InferOutput<typeof answerPresentation>;

const agentChildStatus = v.pipe(
  v.object({
    child_session_id: v.optional(v.string()),
    status: v.string(),
    objective: v.optional(v.nullable(v.string())),
    model_role: v.optional(v.nullable(v.string())),
    usage: v.optional(v.nullable(v.record(v.string(), v.number()))),
    operation_id: v.optional(v.nullable(v.string())),
    operation_sequence: v.optional(v.nullable(v.number())),
    operation_status: v.optional(v.nullable(v.string())),
    cancellation_origin: v.optional(v.nullable(v.string())),
    summary: v.optional(v.nullable(v.string())),
    result_handles: v.optional(v.array(v.string())),
  }),
  v.transform((w) => ({
    childSessionId: w.child_session_id,
    status: w.status,
    objective: w.objective ?? undefined,
    modelRole: w.model_role ?? undefined,
    usage: w.usage ?? null,
    operationId: w.operation_id ?? null,
    operationSequence: w.operation_sequence ?? null,
    operationStatus: w.operation_status ?? null,
    cancellationOrigin: w.cancellation_origin ?? null,
    summary: w.summary ?? null,
    resultHandles: w.result_handles ?? [],
  })),
);
export type AgentChildStatus = v.InferOutput<typeof agentChildStatus>;

const childControlReceipt = v.pipe(
  v.object({
    run_id: v.string(),
    child_session_id: v.optional(v.string()),
    request_id: v.optional(v.string()),
    action: v.string(),
    outcome: v.string(),
    operation_id: v.optional(v.nullable(v.string())),
    operation_sequence: v.optional(v.nullable(v.number())),
    control_sequence: v.optional(v.nullable(v.number())),
    consumed_at: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    runId: w.run_id,
    childSessionId: w.child_session_id ?? null,
    requestId: w.request_id ?? null,
    action: w.action,
    outcome: w.outcome,
    operationId: w.operation_id ?? null,
    operationSequence: w.operation_sequence ?? null,
    controlSequence: w.control_sequence ?? null,
    consumedAt: w.consumed_at ?? null,
  })),
);
export type ChildControlReceipt = v.InferOutput<typeof childControlReceipt>;

const childTranscriptMessage = v.pipe(
  v.object({
    role: v.string(),
    content: v.optional(v.unknown()),
    tool_calls: v.optional(v.array(v.unknown())),
    tool_call_id: v.optional(v.string()),
    name: v.optional(v.string()),
    is_error: v.optional(v.boolean()),
  }),
  v.transform((w) => ({
    role: w.role,
    content: typeof w.content === 'string' ? w.content : '',
    toolCalls: w.tool_calls ?? [],
    toolCallId: w.tool_call_id ?? '',
    name: w.name ?? '',
    isError: w.is_error ?? false,
  })),
);
export type ChildTranscriptMessage = v.InferOutput<typeof childTranscriptMessage>;

const childControlRecord = v.pipe(
  v.object({
    control_sequence: v.number(),
    kind: v.string(),
    content: v.string(),
    origin: v.string(),
    consumed: v.boolean(),
    consumed_at: v.optional(v.nullable(v.string())),
    created_at: v.optional(v.nullable(v.string())),
    operation_id: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    controlSequence: w.control_sequence,
    kind: w.kind,
    content: w.content,
    origin: w.origin,
    consumed: w.consumed,
    consumedAt: w.consumed_at ?? null,
    createdAt: w.created_at ?? null,
    operationId: w.operation_id ?? null,
  })),
);
export type ChildControlRecord = v.InferOutput<typeof childControlRecord>;

const childQuestion = v.pipe(
  v.object({
    request_id: v.string(),
    question: v.string(),
    status: v.string(),
    reply: v.optional(v.nullable(v.string())),
    reply_origin: v.optional(v.nullable(v.string())),
    expires_at: v.optional(v.nullable(v.string())),
    created_at: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    requestId: w.request_id,
    question: w.question,
    status: w.status,
    reply: w.reply ?? null,
    replyOrigin: w.reply_origin ?? null,
    expiresAt: w.expires_at ?? null,
    createdAt: w.created_at ?? null,
  })),
);
export type ChildQuestion = v.InferOutput<typeof childQuestion>;

const childResultLineage = v.pipe(
  v.nullable(v.object({
    status: v.optional(v.string()),
    summary: v.optional(v.string()),
    handles: v.optional(v.array(v.string())),
    operation_id: v.optional(v.nullable(v.string())),
  })),
  v.transform((w) => w === null ? null : {
    status: w.status ?? '',
    summary: w.summary ?? '',
    handles: w.handles ?? [],
    operationId: w.operation_id ?? null,
  }),
);

const childObservation = v.pipe(
  v.object({
    run_id: v.string(),
    child: agentChildStatus,
    transcript: v.array(childTranscriptMessage),
    controls: v.array(childControlRecord),
    questions: v.array(childQuestion),
    result: v.optional(childResultLineage),
  }),
  v.transform((w) => ({
    runId: w.run_id,
    child: w.child,
    transcript: w.transcript,
    controls: w.controls,
    questions: w.questions,
    result: w.result ?? null,
  })),
);
export type ChildObservation = v.InferOutput<typeof childObservation>;

const conversationTurn = v.pipe(
  v.object({
    turn_id: v.string(),
    turn_number: v.number(),
    answer_run_id: v.string(),
    submission_id: v.string(),
    status: v.picklist(['queued', 'running', 'succeeded', 'failed', 'cancelled']),
    cancel_requested: v.boolean(),
    user_text: v.string(),
    assistant_text: v.string(),
    user_attachments: v.array(attachmentReference),
    presentation: v.nullable(answerPresentation),
    usage: v.record(v.string(), v.unknown()),
    evidence: v.record(v.string(), v.number()),
    error_kind: v.nullable(v.string()),
    error_message: v.nullable(v.string()),
    created_at: v.string(),
  }),
  v.transform((w) => ({
    turnId: w.turn_id,
    turnNumber: w.turn_number,
    answerRunId: w.answer_run_id,
    submissionId: w.submission_id,
    status: w.status,
    cancelRequested: w.cancel_requested,
    userText: w.user_text,
    assistantText: w.assistant_text,
    userAttachments: w.user_attachments,
    presentation: w.presentation,
    usage: w.usage,
    evidence: w.evidence,
    errorKind: w.error_kind,
    errorMessage: w.error_message,
    createdAt: w.created_at,
  })),
);
export type ConversationTurn = v.InferOutput<typeof conversationTurn>;

export const acceptedAnswer = v.object({
  conversation: conversationSummary,
  turn: conversationTurn,
});
export type AcceptedAnswer = v.InferOutput<typeof acceptedAnswer>;

const conversationPage = v.pipe(
  v.object({
    items: v.array(conversationSummary),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({items: w.items, nextCursor: w.next_cursor ?? null})),
);
export type ConversationPage = v.InferOutput<typeof conversationPage>;

const conversationHistory = v.pipe(
  v.object({
    conversation: conversationSummary,
    turns: v.array(conversationTurn),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({conversation: w.conversation, turns: w.turns, nextCursor: w.next_cursor ?? null})),
);
export type ConversationHistory = v.InferOutput<typeof conversationHistory>;

const agentChildRosterPage = v.pipe(
  v.object({
    children: v.array(agentChildStatus),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({children: w.children, nextCursor: w.next_cursor ?? null})),
);
export type AgentChildRosterPage = v.InferOutput<typeof agentChildRosterPage>;

export class ConversationApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ConversationApiError';
    this.status = status;
  }
}

export class ChildControlRejectedError extends ConversationApiError {
  readonly outcome: string;

  constructor(status: number, outcome: string) {
    super(status, outcome);
    this.name = 'ChildControlRejectedError';
    this.outcome = outcome;
  }
}

function makeError(status: number, message: string): Error {
  return new ConversationApiError(status, message);
}

export async function listConversations(
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<ConversationPage> {
  const query = cursor === null ? '' : `?cursor=${encodeURIComponent(cursor)}`;
  const response = await fetch(`/web/api/conversations${query}`, {signal});
  return parseWire(response, conversationPage, makeError, 'Failed to load conversations');
}

export async function createConversation(signal?: AbortSignal): Promise<ConversationSummary> {
  const response = await fetch('/web/api/conversations', {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  return parseWire(response, conversationSummary, makeError, 'Failed to create conversation');
}

export async function getConversationHistory(
  conversationId: string,
  cursor: string | null = null,
  limit?: number,
  signal?: AbortSignal,
): Promise<ConversationHistory> {
  const id = encodeURIComponent(conversationId);
  const query = new URLSearchParams();
  if (cursor !== null) query.set('cursor', cursor);
  if (limit !== undefined) query.set('limit', String(limit));
  const suffix = query.size > 0 ? `?${query.toString()}` : '';
  const response = await fetch(`/web/api/conversations/${id}/history${suffix}`, {signal});
  return parseWire(response, conversationHistory, makeError, 'Failed to load conversation history');
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
  return parseWire(response, conversationSummary, makeError, 'Failed to rename conversation');
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

/** Parse the presentation served at a server-provided artifact URL. The URL
 *  stays server-owned; the response still crosses the validated edge. */
export async function getArtifactPresentationAt(
  presentationUrl: string,
  signal?: AbortSignal,
): Promise<AnswerPresentation> {
  const target = new URL(presentationUrl, window.location.origin);
  if (target.origin !== window.location.origin) {
    throw new ConversationApiError(0, 'Untrusted artifact presentation URL');
  }
  const response = await fetch(target.pathname + target.search, {signal});
  return parseWire(response, answerPresentation, makeError, 'Failed to load the Artifact');
}

export async function getRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/runs/${id}`, {signal});
  return parseWire(response, conversationTurn, makeError, 'Failed to load answer run');
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
  const schema = v.record(v.string(), v.unknown());
  return parseWire(response, schema, makeError, 'Failed to steer the answer');
}

export async function getAnswerRunChildren(
  runId: string,
  signal?: AbortSignal,
): Promise<AgentChildStatus[]> {
  const page = await getAnswerRunChildrenPage(runId, null, signal);
  return page.children;
}

export async function getAnswerRunChildrenPage(
  runId: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<AgentChildRosterPage> {
  const id = encodeURIComponent(runId);
  const query = new URLSearchParams();
  if (cursor !== null) query.set('cursor', cursor);
  const suffix = query.size > 0 ? `?${query.toString()}` : '';
  const response = await fetch(`/web/api/answer/${id}/children${suffix}`, {signal});
  return parseWire(response, agentChildRosterPage, makeError, 'Failed to load child agents');
}

export async function getAnswerRunChild(
  runId: string,
  childSessionId: string,
  signal?: AbortSignal,
): Promise<ChildObservation> {
  const run = encodeURIComponent(runId);
  const child = encodeURIComponent(childSessionId);
  const response = await fetch(`/web/api/answer/${run}/children/${child}`, {signal});
  return parseWire(response, childObservation, makeError, 'Failed to load child agent');
}

async function parseChildCommand(
  response: Response,
  fallback: string,
): Promise<ChildControlReceipt> {
  if (response.status === 409) {
    let outcome = 'rejected';
    try {
      const body: unknown = await response.json();
      if (
        body !== null
        && typeof body === 'object'
        && 'detail' in body
        && typeof body.detail === 'string'
        && body.detail.trim()
      ) {
        outcome = body.detail;
      }
    } catch {
      // Keep the generic rejected outcome when the 409 body is unreadable.
    }
    throw new ChildControlRejectedError(409, outcome);
  }
  return parseWire(response, childControlReceipt, makeError, fallback);
}

export async function controlAnswerChild(
  runId: string,
  childSessionId: string,
  action: 'steer' | 'continue' | 'cancel',
  content: string,
  submissionId: string,
  reauthorizeUserCancelled = false,
  signal?: AbortSignal,
): Promise<ChildControlReceipt> {
  const run = encodeURIComponent(runId);
  const child = encodeURIComponent(childSessionId);
  const response = await fetch(`/web/api/answer/${run}/children/${child}/control`, {
    method: 'POST',
    headers: {...csrfHeaders('application/json'), 'Idempotency-Key': submissionId},
    body: JSON.stringify({
      action,
      content,
      reauthorize_user_cancelled: reauthorizeUserCancelled,
    }),
    signal,
  });
  return parseChildCommand(response, 'Failed to control child agent');
}

export async function replyAnswerChild(
  runId: string,
  requestId: string,
  content: string,
  submissionId: string,
  signal?: AbortSignal,
): Promise<ChildControlReceipt> {
  const run = encodeURIComponent(runId);
  const guidance = encodeURIComponent(requestId);
  const response = await fetch(`/web/api/answer/${run}/child-guidance/${guidance}/reply`, {
    method: 'POST',
    headers: {...csrfHeaders('application/json'), 'Idempotency-Key': submissionId},
    body: JSON.stringify({content}),
    signal,
  });
  return parseChildCommand(response, 'Failed to reply to child agent');
}

export async function continueAnswerRun(
  runId: string,
  operation: 'follow-up' | 'fork',
  content: string,
  submissionId: string,
  signal?: AbortSignal,
): Promise<AcceptedAnswer> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/${operation}`, {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({content, submission_id: submissionId}),
    signal,
  });
  return parseWire(response, acceptedAnswer, makeError, `Failed to ${operation} the answer`);
}

export async function cancelRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/runs/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  return parseWire(response, conversationTurn, makeError, 'Failed to stop the answer');
}
