// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The typed failure envelope shared by the browser answer commands.

 * Submit, submission lookup, and fork answer `{kind, message, error_kind?}`;
 * `error_kind` names the stable answer error kind when admission rejected the
 * input or the server classified the failure. The UI localizes; this module
 * never invents user-visible copy.
 */

import {noteRefusal} from './wire.ts';

const WEB_COMMAND_ERROR_KINDS = [
  'invalid_request',
  'attachment_rejected',
  'scope_forbidden',
  'conversation_missing',
  'submission_conflict',
  'service_unavailable',
] as const;

export type WebCommandErrorKind = (typeof WEB_COMMAND_ERROR_KINDS)[number];

export class AnswerSubmissionError extends Error {
  readonly status: number;
  readonly kind: WebCommandErrorKind | 'ambiguous';
  /** The stable answer error kind the server named for this failure, if any. */
  readonly errorKind: string | null;

  constructor(
    status: number,
    kind: WebCommandErrorKind | 'ambiguous',
    message: string,
    errorKind: string | null = null,
  ) {
    super(message);
    this.name = 'AnswerSubmissionError';
    this.status = status;
    this.kind = kind;
    this.errorKind = errorKind;
  }
}

/** Parse one failed command response; an empty message leaves the UI's fallback. */
export async function webCommandError(response: Response): Promise<AnswerSubmissionError> {
  noteRefusal(response.status);
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
  const message = typeof body.message === 'string' ? body.message : '';
  const errorKind = typeof body.error_kind === 'string' ? body.error_kind : null;
  return new AnswerSubmissionError(response.status, kind, message, errorKind);
}
