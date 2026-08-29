// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {getFilePanel} from './files.ts';

const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;

test.beforeEach(() => {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {location: {origin: 'http://localhost'}},
  });
});

test.afterEach(() => {
  globalThis.fetch = originalFetch;
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
