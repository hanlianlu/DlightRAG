// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Localized projection of answer error kinds, stored on Runs or named by rejections.

 * The server taxonomy lives in `src/dlightrag/engine/answer/errors.py`.
 * Kinds whose public message is static map to catalog copy whose English
 * source is the server message without its bracketed marker; kinds with
 * dynamic payloads (filenames, limits, mode names) keep the server message so
 * no detail is lost. Unknown kinds and unmapped payloads fall back to the
 * server message, then to the localized generic failure copy.
 * tests/unit/test_run_error_kind_vocabulary.py locks the keys to the server.
 *
 * Every function here returns the source and its catalog id together, so a view
 * can hold the copy and a component applies the words with `localizedText` when it
 * draws them: a language change then redraws the message.
 */

import {catalogText, serverText, type LocalizedText} from './localized-text.ts';

/** The generic failure copy, which is what an unmapped payload falls back to. */
export const SERVICE_ERROR_TEXT = catalogText('Service error. Please try again.', 'errors.service');

/** The public message of an answer error payload, or the fallback. */
export function answerErrorText(
  payload: unknown,
  fallback: LocalizedText = SERVICE_ERROR_TEXT,
): LocalizedText {
  const message =
    payload !== null && typeof payload === 'object' && !Array.isArray(payload)
      ? (payload as {message?: unknown}).message
      : undefined;
  return typeof message === 'string' && message.trim() ? serverText(message) : fallback;
}

const RUN_ERROR_KIND_COPY: Record<string, string> = {
  unsupported_resource_capability:
    'This request needs a resource capability that no answer mode can provide.',
  ANSWER_RESOURCE_INVALID: 'An answer attachment or link could not be admitted safely.',
  invalid_tool_configuration: 'Answer tooling is misconfigured.',
  ANSWER_IMAGE_CAPABILITY_UNKNOWN:
    'Answer-model image capability is unknown: the startup probe did not confirm image '
    + 'support. Provide a vision-capable query model or retry once the model is reachable.',
  CURRENT_IMAGES_UNSUPPORTED:
    'Current model does not support image input. Use a vision-capable model or remove images.',
  active_run_limit:
    'You have as many answers in progress as you may run at once. '
    + 'Wait for one to finish or stop one, then try again.',
  session_busy:
    'Another answer in this conversation was using the agent lane. This answer did not run. '
    + 'Retry; if the other answer is still working, send this message again and it will be '
    + 'queued after that answer.',
};

function kindSource(kind: string | null): {kind: string; source: string} | null {
  if (kind === null || !Object.hasOwn(RUN_ERROR_KIND_COPY, kind)) return null;
  return {kind, source: RUN_ERROR_KIND_COPY[kind]};
}

/** Copy for a known error kind as a source a view can hold, or null when unmapped. */
export function errorKindText(kind: string | null): LocalizedText | null {
  const known = kindSource(kind);
  return known ? catalogText(known.source, `errors.kind.${known.kind}`) : null;
}

/** Project one live SSE error payload to copy a view can hold. */
export function runErrorText(
  payload: unknown,
  fallback: LocalizedText = SERVICE_ERROR_TEXT,
): LocalizedText {
  const kind = payload !== null && typeof payload === 'object' && !Array.isArray(payload)
    ? (payload as {kind?: unknown}).kind
    : undefined;
  return errorKindText(typeof kind === 'string' ? kind : null)
    ?? answerErrorText(payload, fallback);
}

/** Project one stored turn's terminal error fields to copy a view can hold. */
export function storedRunErrorText(kind: string | null, message: string | null): LocalizedText {
  const localized = errorKindText(kind);
  if (localized) return localized;
  if (typeof message === 'string' && message.trim()) return serverText(message);
  return answerErrorText(null);
}
