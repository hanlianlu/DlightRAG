// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The redacted Settings Connections wire; no credential or catalogue reply is accepted. */
import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {type ApiError, apiErrorFromBody, parseWire, refusalBody, refusalEnvelope} from './wire.ts';

/** `[[name, value], ...]`: what an endpoint needs stated, which is configuration and not a secret. */
const endpointHeaders = v.array(v.tuple([v.string(), v.string()]));
const connection = v.pipe(v.strictObject({
  connection_id: v.string(), label: v.string(), endpoint: v.string(), enabled: v.boolean(),
  activation_epoch: v.number(), generation: v.number(),
  authentication: v.picklist(['none', 'bearer', 'header', 'oauth']),
  endpoint_headers: endpointHeaders,
  // The credential's own header and prefix, readable back because they are not secrets. Null
  // when the deployment cannot read its key ring: the card says so rather than looking unset.
  credential_header: v.nullable(v.string()), credential_prefix: v.nullable(v.string()),
  authorization_status: v.nullable(v.picklist(['pending', 'succeeded', 'failed', 'changed'])),
  status: v.picklist(['disabled', 'ready', 'refreshing', 'degraded', 'needs-auth', 'no-tools', 'revoked']),
  // The published catalogue's size and what its declarations cost one Research request, both
  // estimated with the runtime's own token estimator. Null until a catalogue is published; a
  // disabled Connection's cost is hypothetical and still sent, so the card can state it.
  tool_count: v.nullable(v.number()), declaration_tokens: v.nullable(v.number()),
}), v.transform((w) => ({connectionId: w.connection_id, label: w.label, endpoint: w.endpoint,
  enabled: w.enabled, activationEpoch: w.activation_epoch, generation: w.generation,
  authorizationStatus: w.authorization_status, authentication: w.authentication, status: w.status,
  endpointHeaders: w.endpoint_headers, credentialHeader: w.credential_header,
  credentialPrefix: w.credential_prefix, toolCount: w.tool_count,
  declarationTokens: w.declaration_tokens})));
const preset = v.pipe(v.strictObject({
  preset_id: v.string(), label: v.string(), endpoint: v.string(),
  default_authentication: v.picklist(['none', 'bearer', 'header', 'oauth']),
  endpoint_headers: endpointHeaders,
  credential_header: v.string(), credential_prefix: v.string(),
}), v.transform((w) => ({presetId: w.preset_id, label: w.label, endpoint: w.endpoint,
  defaultAuthentication: w.default_authentication, endpointHeaders: w.endpoint_headers,
  credentialHeader: w.credential_header, credentialPrefix: w.credential_prefix})));
const view = v.pipe(v.strictObject({revision: v.string(), connections: v.array(connection),
  presets: v.array(preset)}),
  v.transform((w) => ({revision: w.revision, connections: w.connections, presets: w.presets})));
export type ConnectionsView = v.InferOutput<typeof view>;
export type Connection = v.InferOutput<typeof connection>;
/** A starter fill for the create form; it carries no credential and grants nothing. */
export type Preset = v.InferOutput<typeof preset>;
/** What a Connection states about its endpoint, as the form edits it. */
export type EndpointHeaderPair = [string, string];

/** A refusal of the Connections module itself, which answers `{kind, message}`.
 *  The route's other refusals (authorization, request validation) answer the
 *  general envelope and arrive as ApiError. */
export class ConnectionsApiError extends Error {
  readonly status: number;
  /** The Connections module's refusal kind. */
  readonly kind: string;
  constructor(status: number, kind: string, message: string) {
    super(message || `HTTP ${status}`);
    this.name = 'ConnectionsApiError';
    this.status = status;
    this.kind = kind;
  }
}

async function connectionsRefusal(response: Response): Promise<ConnectionsApiError | ApiError> {
  const body = await refusalBody(response);
  const {kind, message} = refusalEnvelope(body);
  if (typeof kind !== 'string') return apiErrorFromBody(response.status, body);
  return new ConnectionsApiError(response.status, kind, typeof message === 'string' ? message : '');
}

export async function getConnections(signal?: AbortSignal): Promise<ConnectionsView> {
  return parseWire(await fetch('/web/api/connections/mcp', {signal}), view, connectionsRefusal);
}

type Change = {kind: 'create'; label: string; endpoint: string; endpointHeaders: EndpointHeaderPair[]}
  | {kind: 'edit'; connectionId: string; label: string; endpoint: string; endpointHeaders?: EndpointHeaderPair[]}
  | {kind: 'enable' | 'disable' | 'delete' | 'probe' | 'revoke'; connectionId: string; consentVersion?: 1}
  | {kind: 'bearer'; connectionId: string; bearer: string; endpoint?: string}
  | {kind: 'header'; connectionId: string; header: string; prefix: string; token: string; endpoint?: string};

export async function changeConnection(revision: string, command: Change, signal?: AbortSignal): Promise<ConnectionsView> {
  const base = '/web/api/connections/mcp';
  let url = command.kind === 'create' ? base : `${base}/${encodeURIComponent(command.connectionId)}`;
  let method = 'PATCH';
  const body: Record<string, unknown> = {expected_revision: revision};
  switch (command.kind) {
    case 'create':
      method = 'POST';
      Object.assign(body, {label: command.label, endpoint: command.endpoint, endpoint_headers: command.endpointHeaders});
      break;
    case 'edit':
      Object.assign(body, {kind: 'edit', label: command.label, endpoint: command.endpoint, endpoint_headers: command.endpointHeaders});
      break;
    case 'enable': case 'disable':
      Object.assign(body, {kind: command.kind, ...(command.kind === 'enable' ? {consent_version: command.consentVersion} : {})}); break;
    case 'delete': method = 'DELETE'; break;
    case 'bearer': method = 'PUT'; url += '/bearer'; body.bearer = command.bearer; if (command.endpoint !== undefined) body.endpoint = command.endpoint; break;
    case 'header':
      method = 'PUT'; url += '/header';
      Object.assign(body, {header: command.header, prefix: command.prefix, token: command.token});
      if (command.endpoint !== undefined) body.endpoint = command.endpoint;
      break;
    case 'probe': case 'revoke': method = 'POST'; url += `/${command.kind}`; break;
  }
  return parseWire(await fetch(url, {method, headers: csrfHeaders('application/json'), body: JSON.stringify(body), signal}),
    view, connectionsRefusal);
}

const authorization = v.strictObject({authorization_url: v.pipe(v.string(), v.url(), v.check((url) => {
  const parsed = new URL(url);
  return ['https:', 'http:'].includes(parsed.protocol) && !parsed.username && !parsed.password && !parsed.hash;
}))});
export async function beginConnectionAuthorization(revision: string, connectionId: string, endpoint: string, signal?: AbortSignal): Promise<string> {
  const result = await parseWire(await fetch(`/web/api/connections/mcp/${encodeURIComponent(connectionId)}/oauth`, {
    method: 'POST', headers: csrfHeaders('application/json'),
    body: JSON.stringify({expected_revision: revision, endpoint}), signal,
  }), authorization, connectionsRefusal);
  return result.authorization_url;
}
