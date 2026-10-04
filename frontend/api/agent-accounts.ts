// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Settings Agent Accounts wire: a redacted view, the owner's sign-up switch, and removal.

 * Credentials are write-only everywhere. DlightRAG mints and seals each password, so no reply
 * carries a password, an envelope, a key id, or an account id. A schema that is not strict already
 * drops a field it does not know from what it returns; these are strict, so a reply that names one
 * is refused instead. A backend that began to send a secret here fails where it can be seen, and
 * not quietly in a field nobody reads.
 */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

const timestamp = v.pipe(v.string(), v.isoTimestamp(), v.transform((value) => new Date(value)));

const account = v.pipe(
  v.strictObject({
    site: v.string(),
    email: v.nullable(v.string()),
    username: v.nullable(v.string()),
    created_at: timestamp,
    last_used_at: v.nullable(timestamp),
  }),
  v.transform((w) => ({
    site: w.site,
    email: w.email,
    username: w.username,
    createdAt: w.created_at,
    lastUsedAt: w.last_used_at,
  })),
);

const view = v.pipe(
  v.strictObject({
    available: v.boolean(),
    registration: v.strictObject({allowed: v.boolean(), enabled: v.boolean()}),
    accounts: v.array(account),
  }),
  v.transform((w) => ({
    available: w.available,
    registration: w.registration,
    accounts: w.accounts,
  })),
);

/** One registered website: how the agent signs in there, and when it last did. */
export type AgentAccount = v.InferOutput<typeof account>;

/** The owner's accounts and sign-up switch. `available` is whether the deployment composed Agent
 *  Accounts at all, `registration.allowed` its ceiling, and `registration.enabled` the owner's own
 *  switch; the agent registers only where both are on. */
export type AgentAccountsView = v.InferOutput<typeof view>;

const BASE = '/web/api/agent-accounts';

export async function getAgentAccounts(signal?: AbortSignal): Promise<AgentAccountsView> {
  return parseWire(await fetch(BASE, {signal}), view);
}

/** Turn the owner's "allow new sign-ups" switch; the reply is the fresh view. */
export async function setAgentAccountRegistration(
  enabled: boolean,
  signal?: AbortSignal,
): Promise<AgentAccountsView> {
  return parseWire(await fetch(`${BASE}/settings`, {
    method: 'PUT',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({registration_enabled: enabled}),
    signal,
  }), view);
}

/** Remove the owner's account for one website; a 404 means it is already gone. */
export async function removeAgentAccount(site: string, signal?: AbortSignal): Promise<AgentAccountsView> {
  return parseWire(await fetch(`${BASE}/${encodeURIComponent(site)}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  }), view);
}
