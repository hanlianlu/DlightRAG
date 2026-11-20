// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {getConnections, ConnectionsApiError} from './connections.ts';
import {ApiError} from './wire.ts';
const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;
test.beforeEach(() => {Object.defineProperty(globalThis, 'document', {configurable: true, value: {cookie: ''}});});
test.afterEach(() => {globalThis.fetch = originalFetch; Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});});

test('Connections wire normalizes owner projection and rejects credential-shaped replies', async () => {
  globalThis.fetch = async () => Response.json({revision: '0', connections: [], presets: []});
  assert.deepEqual(await getConnections(), {revision: '0', connections: [], presets: []});
  globalThis.fetch = async () => Response.json({revision: '0', connections: [], presets: [], bearer: 'must-not-enter-ui'});
  await assert.rejects(getConnections(), ApiError);
});

test('a Connections refusal keeps its own {kind, message} envelope', async () => {
  globalThis.fetch = async () => Response.json(
    {kind: 'requires_reauthorization', message: 'Endpoint candidate needs a new grant'},
    {status: 409},
  );
  await assert.rejects(getConnections(), (error: unknown) => error instanceof ConnectionsApiError
    && error.status === 409
    && error.kind === 'requires_reauthorization'
    && error.message === 'Endpoint candidate needs a new grant');
});

test('the route\'s other refusals answer the general envelope and keep their reason', async () => {
  globalThis.fetch = async () => Response.json(
    {detail: 'Invalid Connection command', error_type: 'validation'},
    {status: 422},
  );
  await assert.rejects(getConnections(), (error: unknown) => error instanceof ApiError
    && error.status === 422
    && error.errorType === 'validation'
    && error.detail === 'Invalid Connection command');
  globalThis.fetch = async () => Response.json({detail: 'Not authenticated', error_type: 'auth'}, {status: 401});
  await assert.rejects(getConnections(), (error: unknown) => error instanceof ApiError
    && error.errorType === 'auth'
    && error.detail === 'Not authenticated');
  globalThis.fetch = async () => new Response('', {status: 503});
  await assert.rejects(getConnections(), (error: unknown) => error instanceof ApiError
    && !(error instanceof ConnectionsApiError)
    && error.status === 503
    && error.detail === null);
});

test('presets arrive as display copy plus the tab they imply, and an unknown tab is rejected', async () => {
  const preset = {preset_id: 'notion', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp', default_authentication: 'oauth',
    endpoint_headers: [], credential_header: 'Authorization', credential_prefix: 'Bearer '};
  globalThis.fetch = async () => Response.json({revision: '0', connections: [], presets: [preset]});
  assert.deepEqual((await getConnections()).presets, [
    {presetId: 'notion', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp', defaultAuthentication: 'oauth',
      endpointHeaders: [], credentialHeader: 'Authorization', credentialPrefix: 'Bearer '},
  ]);
  globalThis.fetch = async () => Response.json({revision: '0', connections: [], presets: [{...preset, default_authentication: 'token'}]});
  await assert.rejects(getConnections(), ApiError);
});

test('a preset may state the headers its endpoint needs and the header a credential uses', async () => {
  const feishu = {preset_id: 'feishu', label: 'Feishu/Lark', endpoint: 'https://mcp.feishu.cn/mcp', default_authentication: 'header',
    endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']], credential_header: 'X-Lark-MCP-TAT', credential_prefix: ''};
  globalThis.fetch = async () => Response.json({revision: '0', connections: [], presets: [feishu]});
  assert.deepEqual((await getConnections()).presets, [
    {presetId: 'feishu', label: 'Feishu/Lark', endpoint: 'https://mcp.feishu.cn/mcp', defaultAuthentication: 'header',
      endpointHeaders: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']], credentialHeader: 'X-Lark-MCP-TAT', credentialPrefix: ''},
  ]);
});

test('a Connection reply states its endpoint headers and refuses a reply that hides them', async () => {
  const connection = {connection_id: 'one', label: 'Feishu/Lark', endpoint: 'https://mcp.feishu.cn/mcp', enabled: false,
    activation_epoch: 1, generation: 0, authentication: 'header',
    endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc']],
    credential_header: 'X-Lark-MCP-TAT', credential_prefix: '', authorization_status: null, status: 'disabled',
    tool_count: 3, declaration_tokens: 512};
  globalThis.fetch = async () => Response.json({revision: '0', connections: [connection], presets: []});
  assert.deepEqual((await getConnections()).connections[0]?.endpointHeaders, [['X-Lark-MCP-Allowed-Tools', 'create-doc']]);
  // The credential's own header and prefix are read back, because they are not secrets; the
  // token they travel with is not on the wire at all.
  assert.deepEqual(
    [(await getConnections()).connections[0]?.credentialHeader,
      (await getConnections()).connections[0]?.credentialPrefix],
    ['X-Lark-MCP-TAT', ''],
  );
  globalThis.fetch = async () => Response.json({revision: '0', connections: [{...connection, credential_header: null}], presets: []});
  assert.equal((await getConnections()).connections[0]?.credentialHeader, null);
  const {endpoint_headers: _hidden, ...without} = connection;
  globalThis.fetch = async () => Response.json({revision: '0', connections: [without], presets: []});
  await assert.rejects(getConnections(), ApiError);
});

test('a Connection states its published catalogue size and what declaring it costs a request', async () => {
  const connection = {connection_id: 'one', label: 'Feishu/Lark', endpoint: 'https://mcp.feishu.cn/mcp',
    enabled: false, activation_epoch: 1, generation: 0, authentication: 'header',
    endpoint_headers: [], credential_header: null, credential_prefix: null,
    authorization_status: null, status: 'disabled', tool_count: 12, declaration_tokens: 12_300};
  globalThis.fetch = async () => Response.json({revision: '0', connections: [connection], presets: []});
  const published = (await getConnections()).connections[0]!;
  assert.equal(published.toolCount, 12);
  assert.equal(published.declarationTokens, 12_300);

  // Until a catalogue is published both facts are null, never a claimed zero or cost.
  globalThis.fetch = async () => Response.json({revision: '0',
    connections: [{...connection, tool_count: null, declaration_tokens: null}], presets: []});
  const unpublished = (await getConnections()).connections[0]!;
  assert.equal(unpublished.toolCount, null);
  assert.equal(unpublished.declarationTokens, null);

  // The reply is still strict: hiding the new facts is refused, not read as "no catalogue".
  const {tool_count: _count, declaration_tokens: _tokens, ...hidden} = connection;
  globalThis.fetch = async () => Response.json({revision: '0', connections: [hidden], presets: []});
  await assert.rejects(getConnections(), ApiError);
});

test('OAuth wire rejects unsafe redirects and credential-shaped replies', async () => {
  const {beginConnectionAuthorization} = await import('./connections.ts');
  for (const authorization_url of ['javascript:alert(1)', 'https://user:secret@as.example/', 'https://as.example/#secret']) {
    globalThis.fetch = async () => Response.json({authorization_url});
    await assert.rejects(beginConnectionAuthorization('revision', 'connection', 'https://candidate.example/mcp'), ApiError);
  }
  globalThis.fetch = async () => Response.json({authorization_url: 'https://as.example/authorize', access_token: 'forbidden'});
  await assert.rejects(beginConnectionAuthorization('revision', 'connection', 'https://candidate.example/mcp'), ApiError);
});

test('bearer candidate sends explicit new endpoint and only the write-only credential', async () => {
  const {changeConnection} = await import('./connections.ts');
  globalThis.fetch = async (url, init) => {
    assert.equal(url, '/web/api/connections/mcp/connection/bearer');
    assert.deepEqual(JSON.parse(String(init?.body)), {expected_revision: 'revision', bearer: 'new-test-token', endpoint: 'https://candidate.example/mcp'});
    return Response.json({revision: 'next', connections: [], presets: []});
  };
  await changeConnection('revision', {kind: 'bearer', connectionId: 'connection', bearer: 'new-test-token', endpoint: 'https://candidate.example/mcp'});
});

test('a header credential sends its own header and prefix beside the write-only token', async () => {
  const {changeConnection} = await import('./connections.ts');
  globalThis.fetch = async (url, init) => {
    assert.equal(url, '/web/api/connections/mcp/connection/header');
    assert.equal(init?.method, 'PUT');
    assert.deepEqual(JSON.parse(String(init?.body)), {
      expected_revision: 'revision', header: 'X-Lark-MCP-TAT', prefix: '', token: 'new-test-token',
      endpoint: 'https://mcp.feishu.cn/mcp',
    });
    return Response.json({revision: 'next', connections: [], presets: []});
  };
  await changeConnection('revision', {kind: 'header', connectionId: 'connection', header: 'X-Lark-MCP-TAT',
    prefix: '', token: 'new-test-token', endpoint: 'https://mcp.feishu.cn/mcp'});
});

test('create and edit state endpoint headers as non-secret pairs', async () => {
  const {changeConnection} = await import('./connections.ts');
  globalThis.fetch = async (url, init) => {
    assert.equal(url, '/web/api/connections/mcp');
    assert.equal(init?.method, 'POST');
    assert.deepEqual(JSON.parse(String(init?.body)), {expected_revision: 'revision', label: 'Feishu/Lark',
      endpoint: 'https://mcp.feishu.cn/mcp', endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']]});
    return Response.json({revision: 'next', connections: [], presets: []});
  };
  await changeConnection('revision', {kind: 'create', label: 'Feishu/Lark', endpoint: 'https://mcp.feishu.cn/mcp',
    endpointHeaders: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']]});
  globalThis.fetch = async (url, init) => {
    assert.equal(url, '/web/api/connections/mcp/connection');
    assert.deepEqual(JSON.parse(String(init?.body)), {expected_revision: 'revision', kind: 'edit', label: 'Feishu/Lark',
      endpoint: 'https://mcp.feishu.cn/mcp', endpoint_headers: []});
    return Response.json({revision: 'next', connections: [], presets: []});
  };
  await changeConnection('revision', {kind: 'edit', connectionId: 'connection', label: 'Feishu/Lark',
    endpoint: 'https://mcp.feishu.cn/mcp', endpointHeaders: []});
});
