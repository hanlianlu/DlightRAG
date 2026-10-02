// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {getWebBootstrap} from './bootstrap.ts';
import {ApiError} from './wire.ts';

const originalFetch = globalThis.fetch;

test.afterEach(() => {
  globalThis.fetch = originalFetch;
});

test('bootstrap rejects a non-success response through its typed error', async () => {
  globalThis.fetch = async () => new Response('', {status: 503});

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof ApiError && error.status === 503,
  );
});

test('bootstrap rejects malformed success JSON through its typed error', async () => {
  globalThis.fetch = async () => new Response('<html>', {
    status: 200,
    headers: {'Content-Type': 'text/html'},
  });

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof ApiError && error.status === 200,
  );
});

test('bootstrap v5 requires the agent effort offer', async () => {
  const fixture = {
    contract_version: 5, workspaces: [], primary_workspace: '', active_workspaces: [],
    answer_attachments: {count_limit: 0, image_max_bytes: 1, document_max_bytes: 1,
      extensions: [], image_capability: 'unknown', image_limit: 0, accept: ''},
    active_html_preview_enabled: false,
    agent_effort: {levels: ['low', 'high', 'max'], default: 'high'},
  };
  globalThis.fetch = async () => Response.json(fixture);
  const bootstrap = await getWebBootstrap();
  assert.deepEqual(bootstrap.agentEffort, {levels: ['low', 'high', 'max'], default: 'high'});
  globalThis.fetch = async () => Response.json({...fixture, contract_version: 4});
  await assert.rejects(getWebBootstrap(), ApiError);
  const {agent_effort: _offer, ...missing} = fixture;
  globalThis.fetch = async () => Response.json(missing);
  await assert.rejects(getWebBootstrap(), ApiError);
  // A deployment that names no level, or one the three-level control cannot
  // name, reports no default rather than a level it would not run.
  globalThis.fetch = async () => Response.json({...fixture, agent_effort: {levels: ['low', 'high', 'max']}});
  assert.equal((await getWebBootstrap()).agentEffort.default, null);
  globalThis.fetch = async () => Response.json({...fixture, agent_effort: {levels: ['low'], default: null}});
  assert.deepEqual((await getWebBootstrap()).agentEffort, {levels: ['low'], default: null});
});
