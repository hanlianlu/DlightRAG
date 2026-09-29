// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Localized projection of answer error kinds, stored on Runs or named by rejections.

 * The server taxonomy lives in `src/dlightrag/engine/answer/errors.py`.
 * MODEL_CAPABILITY_UNAVAILABLE is historical: no current path raises it, but
 * Runs stored by earlier releases still carry it. Kinds whose public message
 * is static map to catalog copy whose English source is the server message
 * without its bracketed marker; kinds with dynamic payloads
 * (filenames, limits, mode names) keep the server message so no detail is
 * lost. Unknown kinds and unmapped payloads fall back to the server message,
 * then to the localized generic failure copy.
 * tests/unit/test_run_error_kind_vocabulary.py locks the keys to the server.
 */

import {msg} from '@lit/localize';

/** The public message of an answer error payload, or the localized generic failure. */
export function answerErrorMessage(
  payload: unknown,
  fallback: string = msg('Service error. Please try again.', {id: 'errors.service'}),
): string {
  const message =
    payload !== null && typeof payload === 'object' && !Array.isArray(payload)
      ? (payload as {message?: unknown}).message
      : undefined;
  return typeof message === 'string' && message.trim() ? message : fallback;
}

const RUN_ERROR_KIND_COPY: Record<string, string> = {
  MODEL_CAPABILITY_UNAVAILABLE:
    'The configured query model cannot use the tools required for this answer request.',
  unsupported_resource_capability:
    'This request needs a resource capability that no answer mode can provide.',
  ANSWER_RESOURCE_INVALID: 'An answer attachment or link could not be admitted safely.',
  invalid_tool_configuration: 'Answer tooling is misconfigured.',
  ANSWER_IMAGE_CAPABILITY_UNKNOWN:
    'Answer-model image capability is unknown: the startup probe did not confirm image '
    + 'support. Provide a vision-capable query model or retry once the model is reachable.',
  CURRENT_IMAGES_UNSUPPORTED:
    'Current model does not support image input. Use a vision-capable model or remove images.',
};

function kindSource(kind: string | null): {kind: string; source: string} | null {
  if (kind === null || !Object.hasOwn(RUN_ERROR_KIND_COPY, kind)) return null;
  return {kind, source: RUN_ERROR_KIND_COPY[kind]};
}

/** Localized copy for a known error kind, or null when the kind is unmapped. */
export function localizedErrorKind(kind: string | null): string | null {
  const known = kindSource(kind);
  return known ? msg(known.source, {id: `errors.kind.${known.kind}`}) : null;
}

/** Project one live SSE error payload to user-facing copy. */
export function localizedRunErrorPayload(payload: unknown, fallback?: string): string {
  const kind = payload !== null && typeof payload === 'object' && !Array.isArray(payload)
    ? (payload as {kind?: unknown}).kind
    : undefined;
  return localizedErrorKind(typeof kind === 'string' ? kind : null)
    ?? answerErrorMessage(payload, fallback);
}

/** Project one stored turn's terminal error fields to user-facing copy. */
export function localizedStoredRunError(kind: string | null, message: string | null): string {
  const localized = localizedErrorKind(kind);
  if (localized) return localized;
  if (typeof message === 'string' && message.trim()) return message;
  return answerErrorMessage(null);
}
