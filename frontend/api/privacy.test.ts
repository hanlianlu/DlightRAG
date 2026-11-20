// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {getPrivacySettings, putPrivacySettings} from './privacy.ts';
import {ApiError} from './wire.ts';

const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;
test.beforeEach(() => {
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: 'dlightrag_web_csrf=csrf-token'},
  });
});
test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});
});

test('the permission is read with the moment it was given, as the server writes it, and as a date', async () => {
  // The server writes a moment with microseconds; a date holds milliseconds.
  globalThis.fetch = async () => Response.json({contribute: true, since: '2026-10-09T20:15:03.123456Z'});
  assert.deepEqual(await getPrivacySettings(), {contribute: true, since: new Date('2026-10-09T20:15:03.123Z')});
  globalThis.fetch = async () => Response.json({contribute: false, since: null});
  assert.deepEqual(await getPrivacySettings(), {contribute: false, since: null});
});

test('a reply that cannot be trusted is refused like any other bad reply, never read as off or as on', async () => {
  const replies: unknown[] = [
    {since: null},
    {contribute: 'yes', since: null},
    {contribute: null},
    [],
    // The store holds the switch and its moment together; a reply that splits them is not its answer.
    {contribute: true},
    {contribute: true, since: null},
    {contribute: true, since: 'not a time'},
    {contribute: false, since: '2026-10-09T20:15:03.123456Z'},
  ];
  for (const reply of replies) {
    globalThis.fetch = async () => Response.json(reply);
    await assert.rejects(getPrivacySettings(), ApiError, JSON.stringify(reply));
  }
});

test('turning it on is one PUT with the CSRF token, and the reply is what the server stores', async () => {
  const sent: Array<{url: string; init: RequestInit | undefined}> = [];
  globalThis.fetch = async (url, init) => {
    sent.push({url: String(url), init});
    return Response.json({contribute: true, since: '2026-10-09T20:15:03.123456Z'});
  };

  assert.deepEqual(await putPrivacySettings(true), {contribute: true, since: new Date('2026-10-09T20:15:03.123Z')});
  assert.equal(sent.length, 1);
  assert.equal(sent[0]!.url, '/web/api/privacy/settings');
  assert.equal(sent[0]!.init?.method, 'PUT');
  assert.equal(sent[0]!.init?.body, '{"contribute":true}');
  assert.deepEqual(sent[0]!.init?.headers, {'Content-Type': 'application/json', 'X-CSRF-Token': 'csrf-token'});
});
