// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from './csrf.ts';
import {acceptedAnswer as acceptedAnswerSchema, type AcceptedAnswer} from './conversations.ts';
import {buildAnswerRequest, type AnswerMode} from '../lib/answer-request.ts';
import * as v from 'valibot';

const WEB_COMMAND_ERROR_KINDS = [
  'invalid_request',
  'attachment_rejected',
  'scope_forbidden',
  'conversation_missing',
  'submission_conflict',
  'service_unavailable',
] as const;

export type WebCommandErrorKind = (typeof WEB_COMMAND_ERROR_KINDS)[number];

export interface AnswerSubmissionIntent {
  readonly submissionId: string;
  readonly conversationId: string | null;
  readonly query: string;
  readonly workspaces: readonly string[];
  readonly mode: AnswerMode | null;
  readonly requestedSkill?: string | null;
}

export class AnswerSubmissionError extends Error {
  readonly status: number;
  readonly kind: WebCommandErrorKind | 'ambiguous';

  constructor(
    status: number,
    kind: WebCommandErrorKind | 'ambiguous',
    message: string,
  ) {
    super(message);
    this.name = 'AnswerSubmissionError';
    this.status = status;
    this.kind = kind;
  }
}

function acceptedAnswer(value: unknown): AcceptedAnswer {
  return v.parse(acceptedAnswerSchema, value);
}

async function failure(response: Response): Promise<AnswerSubmissionError> {
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    value = null;
  }
  const body = value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
  const kind = WEB_COMMAND_ERROR_KINDS.includes(body.kind as WebCommandErrorKind)
    ? body.kind as WebCommandErrorKind
    : 'ambiguous';
  const message = typeof body.message === 'string' ? body.message : 'Answer submission failed';
  return new AnswerSubmissionError(response.status, kind, message);
}

export interface AnswerSubmissionAdapter {
  submit(
    intent: AnswerSubmissionIntent,
    files: readonly File[],
    signal: AbortSignal,
  ): Promise<AcceptedAnswer>;
  lookup(submissionId: string, signal: AbortSignal): Promise<AcceptedAnswer | null>;
}

export class BrowserAnswerSubmissionAdapter implements AnswerSubmissionAdapter {
  async submit(
    intent: AnswerSubmissionIntent,
    files: readonly File[],
    signal: AbortSignal,
  ): Promise<AcceptedAnswer> {
    const {body, headers} = buildAnswerRequest({
      query: intent.query,
      workspaces: [...intent.workspaces],
      conversationId: intent.conversationId,
      submissionId: intent.submissionId,
      ...(intent.mode ? {mode: intent.mode} : {}),
      ...(intent.requestedSkill ? {requestedSkill: intent.requestedSkill} : {}),
    }, [...files]);
    let response: Response;
    try {
      response = await fetch('/web/api/answer', {
        method: 'POST',
        headers: {...csrfHeaders(), ...(headers ?? {})},
        body,
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new AnswerSubmissionError(0, 'ambiguous', 'Answer submission was interrupted');
    }
    if (!response.ok) throw await failure(response);
    try {
      return acceptedAnswer(await response.json());
    } catch (error) {
      if (error instanceof AnswerSubmissionError) throw error;
      throw new AnswerSubmissionError(0, 'ambiguous', 'Malformed answer acceptance');
    }
  }

  async lookup(submissionId: string, signal: AbortSignal): Promise<AcceptedAnswer | null> {
    let response: Response;
    try {
      response = await fetch(
        `/web/api/answer-submissions/${encodeURIComponent(submissionId)}`,
        {signal},
      );
    } catch (error) {
      if (signal.aborted) throw error;
      throw new AnswerSubmissionError(0, 'ambiguous', 'Answer lookup was interrupted');
    }
    if (response.status === 404) return null;
    if (!response.ok) throw await failure(response);
    return acceptedAnswer(await response.json());
  }
}
