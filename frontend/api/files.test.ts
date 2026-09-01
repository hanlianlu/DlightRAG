// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  getFailedFileRetryStatus,
  getFailedFiles,
  getFilePanel,
  startFailedFileRetry,
} from './files.ts';

const originalDocument = globalThis.document;
const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;

test.beforeEach(() => {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {location: {origin: 'http://localhost'}},
  });
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
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: originalWindow,
  });
});

test('file pages encode the opaque cursor, pass abort, and normalize older payloads', async () => {
  const seen: Array<{url: string; signal: AbortSignal | null | undefined}> = [];
  globalThis.fetch = async (input, init) => {
    seen.push({url: String(input), signal: init?.signal});
    return new Response(JSON.stringify({
      workspace: 'finance',
      files: [],
      ingest: {busy: false, message: '', progress_percent: null},
    }), {headers: {'Content-Type': 'application/json'}});
  };
  const controller = new AbortController();

  const first = await getFilePanel('finance', null, controller.signal);
  const older = await getFilePanel('finance', 'opaque/cursor +', controller.signal);

  assert.equal(first.next_cursor, null);
  assert.equal(older.next_cursor, null);
  assert.deepEqual(seen, [
    {
      url: '/web/api/files?workspace=finance',
      signal: controller.signal,
    },
    {
      url: '/web/api/files?workspace=finance&cursor=opaque%2Fcursor+%2B',
      signal: controller.signal,
    },
  ]);
});

test('failed-file recovery uses bounded pages, CSRF POST, and job polling', async () => {
  const seen: Array<{url: string; method: string; headers?: HeadersInit}> = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    seen.push({url, method, headers: init?.headers});
    if (method === 'POST') {
      return new Response(JSON.stringify({
        job_id: 'retry-1',
        workspace: 'finance',
        status: 'queued',
        retried: 0,
        succeeded: 0,
        failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    if (url.includes('/retry/')) {
      return new Response(JSON.stringify({
        job_id: 'retry-1',
        workspace: 'finance',
        status: 'succeeded',
        retried: 2,
        succeeded: 2,
        failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    return new Response(JSON.stringify({
      workspace: 'finance',
      failed: [],
      next_cursor: null,
      active_recovery: null,
    }), {headers: {'Content-Type': 'application/json'}});
  };

  await getFailedFiles('finance', 'opaque cursor');
  await startFailedFileRetry('finance');
  await getFailedFileRetryStatus('finance', 'retry/1');

  assert.deepEqual(seen, [
    {
      url: '/web/api/files/failed?workspace=finance&cursor=opaque+cursor',
      method: 'GET',
      headers: undefined,
    },
    {
      url: '/web/api/files/retry?workspace=finance',
      method: 'POST',
      headers: {'X-CSRF-Token': 'test-token'},
    },
    {
      url: '/web/api/files/retry/retry%2F1?workspace=finance',
      method: 'GET',
      headers: undefined,
    },
  ]);
});
