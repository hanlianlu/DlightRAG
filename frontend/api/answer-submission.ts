// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import type {AgentEffort} from './agent-effort.ts';
import {
  buildAnswerRequest,
  type AnswerDelivery,
  type AnswerMode,
} from './answer-request.ts';
import {
  acceptedAnswer as acceptedAnswerSchema,
  answerSubmissionResponse as answerSubmissionResponseSchema,
  type AcceptedAnswer,
  type AnswerSubmissionResponse,
} from './conversations.ts';
import {csrfHeaders} from './csrf.ts';
import {AnswerSubmissionError, webCommandError} from './web-command-error.ts';

export interface AnswerSubmissionIntent {
  readonly submissionId: string;
  readonly conversationId: string | null;
  readonly query: string;
  readonly workspaces: readonly string[];
  readonly mode: AnswerMode | null;
  readonly requestedSkill?: string | null;
  /** The caller's own agent effort; omitted runs use the deployment default. */
  readonly effort?: AgentEffort | null;
  /** The delivery the reader chose for a send that arrives while a Run is live. */
  readonly delivery?: AnswerDelivery;
}

function acceptedAnswer(value: unknown): AcceptedAnswer {
  return v.parse(acceptedAnswerSchema, value);
}

function submissionResponse(value: unknown): AnswerSubmissionResponse {
  return v.parse(answerSubmissionResponseSchema, value);
}

export interface AnswerSubmissionAdapter {
  submit(
    intent: AnswerSubmissionIntent,
    files: readonly File[],
    signal: AbortSignal,
  ): Promise<AnswerSubmissionResponse>;
  lookup(submissionId: string, signal: AbortSignal): Promise<AcceptedAnswer | null>;
}

export class BrowserAnswerSubmissionAdapter implements AnswerSubmissionAdapter {
  async submit(
    intent: AnswerSubmissionIntent,
    files: readonly File[],
    signal: AbortSignal,
  ): Promise<AnswerSubmissionResponse> {
    const {body, headers} = buildAnswerRequest({
      query: intent.query,
      workspaces: [...intent.workspaces],
      conversationId: intent.conversationId,
      submissionId: intent.submissionId,
      ...(intent.mode ? {mode: intent.mode} : {}),
      ...(intent.requestedSkill ? {requestedSkill: intent.requestedSkill} : {}),
      ...(intent.effort ? {effort: intent.effort} : {}),
      ...(intent.delivery ? {delivery: intent.delivery} : {}),
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
      throw new AnswerSubmissionError(0, 'ambiguous', '');
    }
    if (!response.ok) throw await webCommandError(response);
    try {
      return submissionResponse(await response.json());
    } catch (error) {
      if (error instanceof AnswerSubmissionError) throw error;
      throw new AnswerSubmissionError(0, 'ambiguous', '');
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
      throw new AnswerSubmissionError(0, 'ambiguous', '');
    }
    if (response.status === 404) return null;
    if (!response.ok) throw await webCommandError(response);
    // As for a submit: an answer that cannot be read may already be committed, so it is not known.
    try {
      return acceptedAnswer(await response.json());
    } catch {
      throw new AnswerSubmissionError(0, 'ambiguous', '');
    }
  }
}
