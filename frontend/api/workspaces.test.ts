// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {WorkspaceApiError, createWorkspaceRequest, resetWorkspaceRequest} from './workspaces.ts';

const originalDocument = globalThis.document;
const originalFetch = globalThis.fetch;

test.beforeEach(() => {
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: 'dlightrag_web_csrf=test-token'},
  });
});

test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: originalDocument,
  });
});

test('a refused workspace command carries the server detail and status', async () => {
  const remedy = 'This deployment is a read-only replica of the knowledge base: it accepts no '
    + 'corpus writes. Send the workspace creation to a writer.';
  globalThis.fetch = async () => Response.json(
    {detail: remedy, error_type: 'unavailable'},
    {status: 503},
  );

  await assert.rejects(
    createWorkspaceRequest('Finance'),
    (error: unknown) => error instanceof WorkspaceApiError
      && error.status === 503
      && error.message === remedy,
  );
});

test('a refusal without a readable detail falls back to the command copy', async () => {
  globalThis.fetch = async () => new Response('upstream failure', {status: 502});

  await assert.rejects(
    resetWorkspaceRequest('finance'),
    (error: unknown) => error instanceof WorkspaceApiError
      && error.status === 502
      && error.message === 'Could not accept Corpus reset.',
  );
});
