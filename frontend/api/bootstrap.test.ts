// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {BootstrapApiError, getWebBootstrap} from './bootstrap.ts';

const originalFetch = globalThis.fetch;

test.afterEach(() => {
  globalThis.fetch = originalFetch;
});

test('bootstrap rejects a non-success response through its typed error', async () => {
  globalThis.fetch = async () => new Response('', {status: 503});

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof BootstrapApiError && error.status === 503,
  );
});

test('bootstrap rejects malformed success JSON through its typed error', async () => {
  globalThis.fetch = async () => new Response('<html>', {
    status: 200,
    headers: {'Content-Type': 'text/html'},
  });

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof BootstrapApiError && error.status === 200,
  );
});

test('bootstrap v2 requires an explicit personal Connections capability', async () => {
  const fixture = {
    contract_version: 2, workspaces: [], primary_workspace: '', active_workspaces: [],
    answer_attachments: {count_limit: 0, image_max_bytes: 1, document_max_bytes: 1,
      extensions: [], image_capability: 'unknown', image_limit: 0, accept: ''},
    active_html_preview_enabled: false, personal_mcp_connections: true,
  };
  globalThis.fetch = async () => Response.json(fixture);
  assert.equal((await getWebBootstrap()).personalMcpConnections, true);
  globalThis.fetch = async () => Response.json({...fixture, contract_version: 1});
  await assert.rejects(getWebBootstrap(), BootstrapApiError);
  const {personal_mcp_connections: _capability, ...missing} = fixture;
  globalThis.fetch = async () => Response.json(missing);
  await assert.rejects(getWebBootstrap(), BootstrapApiError);
});
