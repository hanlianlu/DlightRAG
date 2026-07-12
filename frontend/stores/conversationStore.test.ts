// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import { ConversationStore } from './conversationStore.ts';

const originalWindow = globalThis.window;

function installStorage(initial: Record<string, string> = {}): Map<string, string> {
  const values = new Map(Object.entries(initial));
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      localStorage: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
        removeItem: (key: string) => values.delete(key),
      },
    },
  });
  return values;
}

afterEach(() => {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: originalWindow,
  });
});

const FIRST = {
  conversation_id: 'first',
  title: 'First',
  created_at: '2026-07-13T08:00:00Z',
  updated_at: '2026-07-13T10:00:00Z',
};

const SECOND = {
  conversation_id: 'second',
  title: null,
  created_at: '2026-07-13T09:00:00Z',
  updated_at: '2026-07-13T11:00:00Z',
};

test('conversation store restores only the active server id from local storage', () => {
  installStorage({'dlightrag.active_conversation_id': 'stored'});

  const store = new ConversationStore();

  assert.equal(store.activeConversationId, 'stored');
  assert.deepEqual(store.conversations, []);
  assert.equal(store.history, null);
});

test('conversation store exposes only server projection mutations', () => {
  installStorage();
  const store = new ConversationStore();

  for (const mutation of [
    'replaceList',
    'select',
    'setHistory',
    'upsertSummary',
    'remove',
    'beginRequest',
    'beginLiveAnswer',
    'finishLiveAnswer',
    'canRenderHistory',
  ] as const) {
    assert.equal(typeof store[mutation], 'function', mutation);
  }
});

test('late bootstrap history cannot replace a live answer viewport', () => {
  installStorage();
  const store = new ConversationStore();
  store.replaceList([FIRST]);
  store.select('first');
  const settledGeneration = store.beginRequest();
  const settled = {conversation: FIRST, turns: []};
  assert.equal(store.setHistory(settled, settledGeneration), true);

  store.beginLiveAnswer('first');
  const lateBootstrapGeneration = store.beginRequest();

  assert.equal(store.canRenderHistory(lateBootstrapGeneration), false);
  let attachedViewport = 'live-answer';
  if (store.setHistory(settled, lateBootstrapGeneration)) attachedViewport = 'history';
  assert.equal(attachedViewport, 'live-answer');
  assert.deepEqual(store.history, settled);

  store.finishLiveAnswer('first');
  assert.equal(store.canRenderHistory(lateBootstrapGeneration), false);

  const recoveryGeneration = store.beginRequest();
  assert.equal(store.canRenderHistory(recoveryGeneration), true);
  assert.equal(store.setHistory(settled, recoveryGeneration), true);
});

test('conversation store persists selection and rejects stale history responses', () => {
  const storage = installStorage();
  const store = new ConversationStore();
  store.replaceList([FIRST, SECOND]);
  store.select('first');
  const firstGeneration = store.beginRequest();
  store.select('second');
  const secondGeneration = store.beginRequest();

  const stale = {conversation: FIRST, turns: []};
  const current = {conversation: SECOND, turns: []};

  assert.equal(storage.get('dlightrag.active_conversation_id'), 'second');
  assert.equal(store.setHistory(stale, firstGeneration), false);
  assert.equal(store.history, null);
  assert.equal(store.setHistory(current, secondGeneration), true);
  assert.deepEqual(store.history, current);
});

test('conversation store upserts authoritative summaries and removes projections', () => {
  const storage = installStorage();
  const store = new ConversationStore();
  store.replaceList([FIRST]);
  store.select('first');

  store.upsertSummary({...FIRST, title: 'Renamed', updated_at: '2026-07-13T12:00:00Z'});
  store.upsertSummary(SECOND);

  assert.deepEqual(store.conversations.map((item) => item.conversation_id), ['first', 'second']);
  assert.equal(store.conversations[0].title, 'Renamed');
  assert.equal(store.remove('first'), true);
  assert.equal(store.activeConversationId, null);
  assert.equal(store.history, null);
  assert.equal(storage.has('dlightrag.active_conversation_id'), false);
  assert.equal(store.remove('missing'), false);
});
