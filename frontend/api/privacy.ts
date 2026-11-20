// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Settings Privacy wire: whether the owner lets their conversations improve the product.
 *
 * The permission is the owner's own and off until they turn it on. A read and a write answer alike
 * with what the server stores: on, with the moment it was given (`since`), or off, with none. The
 * store holds the two together (a row is on exactly when it has a moment), so a reply that splits
 * them is not one this client can trust. It is refused like any other bad reply, and the page shows
 * its failed read and not a state it guessed.
 */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire, timestamp} from './wire.ts';

const privacySettings = v.variant('contribute', [
  v.object({contribute: v.literal(true), since: timestamp}),
  v.object({contribute: v.literal(false), since: v.null()}),
]);

export type PrivacySettings = v.InferOutput<typeof privacySettings>;

const PATH = '/web/api/privacy/settings';

export async function getPrivacySettings(signal?: AbortSignal): Promise<PrivacySettings> {
  return parseWire(await fetch(PATH, {signal}), privacySettings);
}

/** Turn the owner's permission on or off; the reply is what the server now stores. */
export async function putPrivacySettings(contribute: boolean, signal?: AbortSignal): Promise<PrivacySettings> {
  return parseWire(await fetch(PATH, {
    method: 'PUT',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({contribute}),
    signal,
  }), privacySettings);
}
