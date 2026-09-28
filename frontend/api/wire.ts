// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The single seam where wire JSON becomes typed domain objects.

 * Every REST response is parsed through a valibot schema declared beside its
 * client function: the schema is the runtime check, the inferred type, and the
 * one place snake_case Wire Format is translated. Failures cross the same edge:
 * a refusal answering the general envelope `{detail, error_type, error_kind?}`
 * becomes one `ApiError`, parsed here and nowhere else. See
 * docs/adr/0002-browser-wire-validation.md.
 */

import * as v from 'valibot';

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

/** Classify a status the way the server does when a body names no type. */
function errorTypeForStatus(status: number): ApiErrorType {
  if (status === 401 || status === 403) return 'auth';
  if (status === 404 || status === 410) return 'not_found';
  if (status === 409 || status === 412) return 'conflict';
  if (status === 429 || status === 503) return 'unavailable';
  if (status >= 400 && status < 500) return 'validation';
  return 'internal';
}

export interface ApiErrorFields {
  readonly detail?: string | null;
  readonly errorType?: ApiErrorType;
  readonly errorKind?: string | null;
}

/** A refused or unreadable response from a route answering the general envelope.

 * `status` is the HTTP status even when a successful response was unreadable,
 * so a caller can tell a refusal from an answer it could not parse. `detail`
 * is the server's public reason, or null when it gave none. This carries no
 * user-visible copy: the UI chooses localized text from these fields.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | null;
  readonly errorType: ApiErrorType;
  readonly errorKind: string | null;

  constructor(status: number, fields: ApiErrorFields = {}) {
    super(fields.detail ?? `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = fields.detail ?? null;
    this.errorType = fields.errorType ?? errorTypeForStatus(status);
    this.errorKind = fields.errorKind ?? null;
  }
}

function envelopeOf(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

/** Parse one refused response; a body that is not the envelope keeps only its status. */
export async function apiError(response: Response): Promise<ApiError> {
  const envelope = envelopeOf(await response.json().catch(() => null));
  const {detail, error_type: errorType, error_kind: errorKind} = envelope;
  return new ApiError(response.status, {
    // FastAPI's own request validation answers a list here; that is no reason.
    detail: typeof detail === 'string' && detail.trim() ? detail : null,
    errorType: API_ERROR_TYPES.includes(errorType as ApiErrorType)
      ? errorType as ApiErrorType
      : errorTypeForStatus(response.status),
    errorKind: typeof errorKind === 'string' && errorKind ? errorKind : null,
  });
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
