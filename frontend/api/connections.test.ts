// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {getConnections, ConnectionsApiError} from './connections.ts';
const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;
test.beforeEach(() => {Object.defineProperty(globalThis, 'document', {configurable: true, value: {cookie: ''}});});
test.afterEach(() => {globalThis.fetch = originalFetch; Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});});

test('Connections wire normalizes owner projection and rejects credential-shaped replies', async () => {
  globalThis.fetch = async () => Response.json({revision: '0', single_user: true, connections: []});
  assert.deepEqual(await getConnections(), {revision: '0', singleUser: true, connections: []});
  globalThis.fetch = async () => Response.json({revision: '0', single_user: false, connections: [], bearer: 'must-not-enter-ui'});
  await assert.rejects(getConnections(), ConnectionsApiError);
});

test('OAuth wire rejects unsafe redirects and credential-shaped replies', async () => {
  const {beginConnectionAuthorization} = await import('./connections.ts');
  for (const authorization_url of ['javascript:alert(1)', 'https://user:secret@as.example/', 'https://as.example/#secret']) {
    globalThis.fetch = async () => Response.json({authorization_url});
    await assert.rejects(beginConnectionAuthorization('revision', 'connection', 'https://candidate.example/mcp'), ConnectionsApiError);
  }
  globalThis.fetch = async () => Response.json({authorization_url: 'https://as.example/authorize', access_token: 'forbidden'});
  await assert.rejects(beginConnectionAuthorization('revision', 'connection', 'https://candidate.example/mcp'), ConnectionsApiError);
});

test('bearer candidate sends explicit new endpoint and only the write-only credential', async () => {
  const {changeConnection} = await import('./connections.ts');
  globalThis.fetch = async (url, init) => {
    assert.equal(url, '/web/api/connections/mcp/connection/bearer');
    assert.deepEqual(JSON.parse(String(init?.body)), {expected_revision: 'revision', bearer: 'new-test-token', endpoint: 'https://candidate.example/mcp'});
    return Response.json({revision: 'next', single_user: false, connections: []});
  };
  await changeConnection('revision', {kind: 'bearer', connectionId: 'connection', bearer: 'new-test-token', endpoint: 'https://candidate.example/mcp'});
});
