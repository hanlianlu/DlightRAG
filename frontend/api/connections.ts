// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The redacted Settings Connections wire; no credential response is accepted. */
import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

const tool = v.pipe(v.strictObject({
  remote_name: v.string(), local_name: v.string(), description: v.string(),
  input_schema: v.record(v.string(), v.unknown()),
}), v.transform((w) => ({remoteName: w.remote_name, localName: w.local_name,
  description: w.description, inputSchema: w.input_schema})));
const connection = v.pipe(v.strictObject({
  connection_id: v.string(), label: v.string(), endpoint: v.string(), enabled: v.boolean(),
  activation_epoch: v.number(), generation: v.number(), authentication: v.picklist(['none', 'bearer', 'oauth']),
  authorization_status: v.nullable(v.picklist(['pending', 'succeeded', 'failed'])),
  status: v.picklist(['disabled', 'ready', 'refreshing', 'degraded', 'needs-auth', 'revoked']),
  last_error_kind: v.nullable(v.string()), catalogue_created_at: v.nullable(v.string()), tools: v.array(tool),
}), v.transform((w) => ({connectionId: w.connection_id, label: w.label, endpoint: w.endpoint,
  enabled: w.enabled, activationEpoch: w.activation_epoch, generation: w.generation,
  authorizationStatus: w.authorization_status, authentication: w.authentication, status: w.status, lastErrorKind: w.last_error_kind,
  catalogueCreatedAt: w.catalogue_created_at, tools: w.tools})));
const view = v.pipe(v.strictObject({revision: v.string(), single_user: v.boolean(), connections: v.array(connection)}),
  v.transform((w) => ({revision: w.revision, singleUser: w.single_user, connections: w.connections})));
export type ConnectionsView = v.InferOutput<typeof view>;
export type Connection = v.InferOutput<typeof connection>;

export class ConnectionsApiError extends Error {
  readonly status: number;
  constructor(status: number) {
    super(status === 409 ? 'Connections changed or authorization is required. Reload before retrying.' : 'Connection request failed.');
    this.status = status;
  }
}

export async function getConnections(signal?: AbortSignal): Promise<ConnectionsView> {
  return parseWire(await fetch('/web/api/connections/mcp', {signal}), view,
    (status) => new ConnectionsApiError(status), 'Connection request failed.');
}

type Change = {kind: 'create'; label: string; endpoint: string}
  | {kind: 'edit'; connectionId: string; label: string; endpoint: string}
  | {kind: 'enable' | 'disable' | 'delete' | 'probe' | 'revoke'; connectionId: string; consentVersion?: 1}
  | {kind: 'bearer'; connectionId: string; bearer: string; endpoint?: string};

export async function changeConnection(revision: string, command: Change, signal?: AbortSignal): Promise<ConnectionsView> {
  const base = '/web/api/connections/mcp';
  let url = command.kind === 'create' ? base : `${base}/${encodeURIComponent(command.connectionId)}`;
  let method = 'PATCH';
  const body: Record<string, unknown> = {expected_revision: revision};
  switch (command.kind) {
    case 'create':
      method = 'POST'; Object.assign(body, {label: command.label, endpoint: command.endpoint}); break;
    case 'edit':
      Object.assign(body, {kind: 'edit', label: command.label, endpoint: command.endpoint}); break;
    case 'enable': case 'disable':
      Object.assign(body, {kind: command.kind, ...(command.kind === 'enable' ? {consent_version: command.consentVersion} : {})}); break;
    case 'delete': method = 'DELETE'; break;
    case 'bearer': method = 'PUT'; url += '/bearer'; body.bearer = command.bearer; if (command.endpoint !== undefined) body.endpoint = command.endpoint; break;
    case 'probe': case 'revoke': method = 'POST'; url += `/${command.kind}`; break;
  }
  return parseWire(await fetch(url, {method, headers: csrfHeaders('application/json'), body: JSON.stringify(body), signal}),
    view, (status) => new ConnectionsApiError(status), 'Connection request failed.');
}

const authorization = v.strictObject({authorization_url: v.pipe(v.string(), v.url(), v.check((url) => {
  const parsed = new URL(url);
  return ['https:', 'http:'].includes(parsed.protocol) && !parsed.username && !parsed.password && !parsed.hash;
}))});
export async function beginConnectionAuthorization(revision: string, connectionId: string, endpoint: string, signal?: AbortSignal): Promise<string> {
  const result = await parseWire(await fetch(`/web/api/connections/mcp/${encodeURIComponent(connectionId)}/oauth`, {
    method: 'POST', headers: csrfHeaders('application/json'),
    body: JSON.stringify({expected_revision: revision, endpoint}), signal,
  }), authorization, (status) => new ConnectionsApiError(status), 'Authorization failed. Restart from Settings.');
  return result.authorization_url;
}
