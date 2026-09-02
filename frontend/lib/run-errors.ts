// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Localized projection of durable answer-run error kinds.

 * The server taxonomy lives in `src/dlightrag/application/answer_runs/errors.py`.
 * Kinds whose public message is static map to catalog copy whose English
 * source is the server message verbatim; kinds with dynamic payloads
 * (filenames, limits, mode names) keep the server message so no detail is
 * lost. Unknown kinds and unmapped payloads fall back to the server message,
 * then to the localized generic failure copy.
 */

import {msg} from '@lit/localize';
import {answerErrorMessage} from './errors.ts';

const RUN_ERROR_KIND_COPY: Record<string, string> = {
  MODEL_CAPABILITY_UNAVAILABLE:
    'The configured query model cannot use the tools required for this answer request.',
  UNSUPPORTED_RESOURCE_CAPABILITY:
    'This request needs a resource capability that no answer mode can provide.',
  ANSWER_RESOURCE_INVALID: 'An answer attachment or link could not be admitted safely.',
  memory_disabled: 'Profile Memory is not active for this owner.',
  invalid_tool_configuration: 'Answer tooling is misconfigured.',
  ANSWER_IMAGE_CAPABILITY_UNKNOWN:
    'Answer-model image capability is unknown: the startup probe did not confirm image '
    + 'support. Provide a vision-capable query model or retry once the model is reachable.',
  CURRENT_IMAGES_UNSUPPORTED:
    'Current model does not support image input. Use a vision-capable model or remove images.',
};

function kindSource(kind: unknown): {kind: string; source: string} | null {
  if (typeof kind !== 'string') return null;
  const source = RUN_ERROR_KIND_COPY[kind];
  return source === undefined ? null : {kind, source};
}

/** Project one live SSE error payload to user-facing copy. */
export function localizedRunErrorPayload(payload: unknown, fallback?: string): string {
  const known = kindSource(
    payload !== null && typeof payload === 'object' && !Array.isArray(payload)
      ? (payload as {kind?: unknown}).kind
      : undefined,
  );
  if (known) return msg(known.source, {id: `errors.kind.${known.kind}`});
  return answerErrorMessage(payload, fallback);
}

/** Project one stored turn's terminal error fields to user-facing copy. */
export function localizedStoredRunError(kind: string | null, message: string | null): string {
  const known = kindSource(kind);
  if (known) return msg(known.source, {id: `errors.kind.${known.kind}`});
  if (typeof message === 'string' && message.trim()) return message;
  return answerErrorMessage(null);
}
