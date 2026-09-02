// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The single seam where wire JSON becomes typed domain objects.

 * Every REST response is parsed through a valibot schema declared beside its
 * client function: the schema is the runtime check, the inferred type, and the
 * one place snake_case Wire Format is translated. See
 * docs/adr/0002-browser-wire-validation.md.
 */

import * as v from 'valibot';

export type ErrorFactory = (status: number, message: string) => Error;

export async function parseWire<Input, Output>(
  response: Response,
  schema: v.GenericSchema<Input, Output>,
  makeError: ErrorFactory,
  fallback: string,
): Promise<Output> {
  if (!response.ok) throw makeError(response.status, fallback);
  try {
    return v.parse(schema, await response.json());
  } catch {
    // Malformed JSON and schema violations are the same failure to callers.
    throw makeError(response.status, fallback);
  }
}
