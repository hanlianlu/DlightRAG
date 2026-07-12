// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import { ConversationStore } from './conversationStore.ts';

const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;

function installStorage(): void {
  const values = new Map<string, string>();
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      localStorage: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
      },
    },
  });
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: originalWindow,
  });
});

test('conversation bootstrap exposes 503 and recovers on retry', async () => {
  installStorage();
  let attempts = 0;
  globalThis.fetch = async () => {
    attempts += 1;
    if (attempts === 1) return new Response('unavailable', { status: 503 });
    return Response.json([{ conversation_id: 'server-conversation' }]);
  };
  const store = new ConversationStore();

  assert.equal(await store.initialize(), null);
  assert.equal(store.status, 'unavailable');
  assert.ok(store.errorMessage);
  assert.match(store.errorMessage, /unavailable/i);

  assert.equal(await store.ensureActive(), 'server-conversation');
  assert.equal(store.status, 'ready');
  assert.equal(store.errorMessage, null);
});

test('conversation bootstrap contains network failures without inventing an id', async () => {
  installStorage();
  globalThis.fetch = async () => {
    throw new TypeError('network down');
  };
  const store = new ConversationStore();

  assert.equal(await store.initialize(), null);
  assert.equal(store.activeConversationId, null);
  assert.equal(store.status, 'unavailable');
});

test('conversation bootstrap exposes server create failure without inventing an id', async () => {
  installStorage();
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    if (calls === 1) return Response.json([]);
    return new Response('unavailable', { status: 503 });
  };
  const store = new ConversationStore();

  assert.equal(await store.initialize(), null);
  assert.equal(calls, 2);
  assert.equal(store.activeConversationId, null);
  assert.equal(store.status, 'unavailable');
});
