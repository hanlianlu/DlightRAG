// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The single seam where wire JSON becomes typed domain objects.

 * Every REST response is parsed through a valibot schema declared beside its
 * client function: the schema is the runtime check, the inferred type, and the
 * one place snake_case Wire Format is translated. Failures cross the same edge:
 * a refusal answering the general envelope `{detail, error_type, error_kind?}`
 * becomes one `ApiError`, parsed here and nowhere else, and every refusal's
 * status is noted here, so a 401 from any route signs the page out once. See
 * docs/adr/0002-browser-wire-validation.md.
 */

import * as v from 'valibot';

const signIn = new EventTarget();

/** Hear that the server no longer accepts this page's sign-in; returns the unsubscribe. */
export function onSignedOut(listener: () => void): () => void {
  signIn.addEventListener('signed-out', listener);
  return () => signIn.removeEventListener('signed-out', listener);
}

/** Every reader of a refused response notes its status here, whatever envelope
 *  the body answers. A 401 from any Web route means the browser's sign-in
 *  expired or was never there: the shell then asks the reader to sign in again,
 *  so no caller handles a 401 of its own. */
export function noteRefusal(status: number): void {
  if (status === 401) signIn.dispatchEvent(new Event('signed-out'));
}

/** The server's public `error_type` vocabulary (docs/interfaces.md). */
const API_ERROR_TYPES = [
  'validation',
  'auth',
  'not_found',
  'conflict',
  'unavailable',
  'configuration',
  'internal',
] as const;

export type ApiErrorType = (typeof API_ERROR_TYPES)[number];

export interface ApiErrorFields {
  readonly detail?: string | null;
  readonly errorType?: ApiErrorType | null;
  readonly errorKind?: string | null;
}

/** A refused or unreadable response from a route answering the general envelope.

 * `status` is the HTTP status even when a successful response was unreadable,
 * so a caller can tell a refusal from an answer it could not parse. `detail`
 * is the server's public reason and `errorType` the type it names, each null
 * when the body gave none: a status alone is never read as a type, so a
 * proxy's or guard's bare 403 is not an authorization refusal. This carries
 * no user-visible copy: the UI chooses localized text from these fields.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | null;
  readonly errorType: ApiErrorType | null;
  readonly errorKind: string | null;

  constructor(status: number, fields: ApiErrorFields = {}) {
    super(fields.detail ?? `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = fields.detail ?? null;
    this.errorType = fields.errorType ?? null;
    this.errorKind = fields.errorKind ?? null;
  }
}

function envelopeOf(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

/** Read a refusal body already parsed as JSON (null when it was not), noting its
 *  status; a body that is not the envelope keeps only its status. */
export function apiErrorFromBody(status: number, body: unknown): ApiError {
  noteRefusal(status);
  const {detail, error_type: errorType, error_kind: errorKind} = envelopeOf(body);
  return new ApiError(status, {
    // FastAPI's own request validation answers a list here; that is no reason.
    detail: typeof detail === 'string' && detail.trim() ? detail : null,
    errorType: API_ERROR_TYPES.includes(errorType as ApiErrorType) ? errorType as ApiErrorType : null,
    errorKind: typeof errorKind === 'string' && errorKind ? errorKind : null,
  });
}

/** Parse one refused response through the general envelope. */
export async function apiError(response: Response): Promise<ApiError> {
  return apiErrorFromBody(response.status, await response.json().catch(() => null));
}

/** Parse a response through its schema; refusals go through `refused`. */
export async function parseWire<Input, Output>(
  response: Response,
  schema: v.GenericSchema<Input, Output>,
  refused: (response: Response) => Promise<Error> = apiError,
): Promise<Output> {
  if (!response.ok) throw await refused(response);
  try {
    return v.parse(schema, await response.json());
  } catch {
    // Malformed JSON and schema violations are the same failure to callers.
    throw new ApiError(response.status);
  }
}
