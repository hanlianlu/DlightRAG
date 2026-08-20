// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {ConversationApiError, type ConversationHistory, type ConversationSummary} from '../api/conversations.ts';
import {ConversationStore, type ConversationApi} from './conversationStore.ts';

function summary(id: string, updated = '2026-08-20T00:00:00Z'): ConversationSummary {
  return {
    conversation_id: id,
    title: id,
    created_at: '2026-08-19T00:00:00Z',
    updated_at: updated,
  };
}

function history(id: string): ConversationHistory {
  return {conversation: summary(id), turns: []};
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function api(overrides: Partial<ConversationApi> = {}): ConversationApi {
  return {
    list: async () => [],
    history: async (id) => history(id),
    rename: async (id, title) => ({...summary(id), title}),
    delete: async () => {},
    deleteAll: async () => {},
    ...overrides,
  };
}

test('new chat is answerable without inventing or restoring a conversation id', () => {
  const store = new ConversationStore(api());

  store.openNew();

  assert.equal(store.viewState, 'new');
  assert.equal(store.canAnswer, true);
  assert.equal(store.answerConversationId, null);
  assert.equal(store.activeConversationId, null);
});

test('list loading is server-owned, sorted, and observable', async () => {
  const store = new ConversationStore(api({
    list: async () => [summary('older'), summary('newer', '2026-08-21T00:00:00Z')],
  }));
  let changes = 0;
  store.subscribe(() => { changes += 1; });

  await store.loadList();

  assert.deepEqual(store.conversations.map((item) => item.conversation_id), ['newer', 'older']);
  assert.equal(store.listState, 'ready');
  assert.ok(changes >= 2);
});

test('opening a route drops a superseded history response', async () => {
  const first = deferred<ConversationHistory>();
  const second = deferred<ConversationHistory>();
  const store = new ConversationStore(api({
    history: async (id) => id === 'first' ? first.promise : second.promise,
  }));

  const openingFirst = store.open('first');
  const openingSecond = store.open('second');
  first.resolve(history('first'));
  second.resolve(history('second'));

  assert.equal(await openingFirst, 'stale');
  assert.equal(await openingSecond, 'ready');
  assert.equal(store.activeConversationId, 'second');
  assert.equal(store.history?.conversation.conversation_id, 'second');
});

test('missing and malformed route ids share one unavailable state', async () => {
  for (const status of [404, 422]) {
    const store = new ConversationStore(api({
      history: async () => { throw new ConversationApiError(status, 'hidden'); },
    }));

    assert.equal(await store.open('opaque'), 'unavailable');
    assert.equal(store.viewState, 'unavailable');
    assert.equal(store.canAnswer, false);
    assert.equal(store.activeConversationId, 'opaque');
  }
});

test('adopting an atomically created conversation updates routing state without viewport revision', () => {
  const store = new ConversationStore(api());
  store.openNew();
  const before = store.viewRevision;

  store.adoptCreatedConversation(summary('created'));

  assert.equal(store.activeConversationId, 'created');
  assert.equal(store.answerConversationId, 'created');
  assert.equal(store.viewRevision, before);
  assert.deepEqual(store.conversations.map((item) => item.conversation_id), ['created']);
});

test('background refresh preserves visible history on transient failure', async () => {
  let fail = false;
  const store = new ConversationStore(api({
    history: async (id) => {
      if (fail) throw new ConversationApiError(503, 'down');
      return history(id);
    },
  }));
  await store.open('one');
  const before = store.viewRevision;
  fail = true;

  assert.equal(await store.refreshActive(), 'error');
  assert.equal(store.viewState, 'ready');
  assert.equal(store.history?.conversation.conversation_id, 'one');
  assert.equal(store.viewRevision, before);
});

test('rename validation errors do not masquerade as missing conversations', async () => {
  const store = new ConversationStore(api({
    list: async () => [summary('one')],
    rename: async () => { throw new ConversationApiError(422, 'invalid title'); },
  }));
  await store.loadList();

  assert.equal(await store.rename('one', ''), 'error');
  assert.deepEqual(store.conversations.map((item) => item.conversation_id), ['one']);
});

test('delete mutates the server before removing the local summary', async () => {
  const calls: string[] = [];
  const store = new ConversationStore(api({
    list: async () => [summary('one')],
    delete: async (id) => { calls.push(id); },
  }));
  await store.loadList();

  assert.equal(await store.delete('one'), 'ok');
  assert.deepEqual(calls, ['one']);
  assert.deepEqual(store.conversations, []);
  assert.equal(store.mutationPending, false);
});
