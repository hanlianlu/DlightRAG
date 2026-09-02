// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ConversationApiError,
  type ConversationHistory,
  type ConversationPage,
  type ConversationSummary,
} from '../api/conversations.ts';
import {ConversationStore, type ConversationApi} from './conversation-store.ts';

function summary(id: string, updated = '2026-08-20T00:00:00Z'): ConversationSummary {
  return {
    conversationId: id,
    title: id,
    createdAt: '2026-08-19T00:00:00Z',
    updatedAt: updated,
    forkedFromConversationId: null,
    forkedFromTitle: null,
  };
}

function history(id: string): ConversationHistory {
  return {conversation: summary(id), turns: [], nextCursor: null};
}

function page(items: ConversationSummary[], nextCursor: string | null = null): ConversationPage {
  return {items, nextCursor: nextCursor};
}

function turn(number: number) {
  return {
    turnId: `turn-${number}`,
    turnNumber: number,
    answerRunId: `run-${number}`,
    submissionId: `submission-${number}`,
    status: 'succeeded' as const,
    cancelRequested: false,
    userText: `Question ${number}`,
    assistantText: `Answer ${number}`,
    userAttachments: [],
    presentation: null,
    usage: {},
    evidence: {},
    errorKind: null,
    errorMessage: null,
    createdAt: '2026-08-20T00:00:00Z',
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function api(overrides: Partial<ConversationApi> = {}): ConversationApi {
  return {
    list: async () => page([]),
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
    list: async () => page([
      summary('older'),
      summary('newer', '2026-08-21T00:00:00Z'),
    ]),
  }));
  let changes = 0;
  store.subscribe(() => { changes += 1; });

  await store.loadList();

  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['newer', 'older']);
  assert.equal(store.listState, 'ready');
  assert.ok(changes >= 2);
});

test('load older coalesces overlap and appends deduped deterministic pages', async () => {
  const older = deferred<ConversationPage>();
  const cursors: Array<string | null> = [];
  const store = new ConversationStore(api({
    list: async (cursor) => {
      cursors.push(cursor);
      if (cursor === null) {
        return page([
          summary('00000000-0000-0000-0000-000000000002'),
          summary('00000000-0000-0000-0000-000000000001'),
        ], 'older-cursor');
      }
      return older.promise;
    },
  }));
  await store.loadList();

  const firstFlight = store.loadOlder();
  const overlappingFlight = store.loadOlder();
  assert.equal(firstFlight, overlappingFlight);
  assert.equal(store.loadMoreState, 'loading');
  older.resolve(page([
    summary('00000000-0000-0000-0000-000000000001'),
    summary('00000000-0000-0000-0000-000000000003', '2026-08-19T00:00:00Z'),
  ]));
  await firstFlight;

  assert.deepEqual(cursors, [null, 'older-cursor']);
  assert.deepEqual(store.conversations.map((item) => item.conversationId), [
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000003',
  ]);
  assert.equal(store.hasOlderConversations, false);
  assert.equal(store.loadMoreState, 'idle');
});

test('load older errors preserve loaded rows and remain retryable', async () => {
  let olderAttempts = 0;
  const store = new ConversationStore(api({
    list: async (cursor) => {
      if (cursor === null) return page([summary('new')], 'older-cursor');
      olderAttempts += 1;
      if (olderAttempts === 1) throw new ConversationApiError(503, 'down');
      return page([summary('old', '2026-08-19T00:00:00Z')]);
    },
  }));
  await store.loadList();

  await store.loadOlder();
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['new']);
  assert.equal(store.loadMoreState, 'error');
  await store.loadOlder();

  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['new', 'old']);
  assert.equal(store.loadMoreState, 'idle');
});

test('reload cancels an older-page request and replaces it with a fresh first page', async () => {
  let firstPages = 0;
  let olderAborted = false;
  const store = new ConversationStore(api({
    list: async (cursor, signal) => {
      if (cursor === null) {
        firstPages += 1;
        return firstPages === 1
          ? page([summary('first')], 'older-cursor')
          : page([summary('reloaded')]);
      }
      return await new Promise<ConversationPage>((_resolve, reject) => {
        signal?.addEventListener('abort', () => {
          olderAborted = true;
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    },
  }));
  await store.loadList();
  const older = store.loadOlder();

  await store.loadList();
  await older;

  assert.equal(olderAborted, true);
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['reloaded']);
  assert.equal(store.loadMoreState, 'idle');
});

test('dispose aborts a pending page without applying late state', async () => {
  let aborted = false;
  const store = new ConversationStore(api({
    list: async (cursor, signal) => {
      if (cursor === null) return page([summary('first')], 'older-cursor');
      return await new Promise<ConversationPage>((_resolve, reject) => {
        signal?.addEventListener('abort', () => {
          aborted = true;
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    },
  }));
  await store.loadList();
  const older = store.loadOlder();

  store.dispose();
  await older;

  assert.equal(aborted, true);
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['first']);
});

test('microsecond timestamps and UUID ties retain exact newest-first order', async () => {
  const store = new ConversationStore(api({
    list: async () => page([
      summary('00000000-0000-0000-0000-000000000001', '2026-08-20T00:00:00Z'),
      summary('00000000-0000-0000-0000-000000000002', '2026-08-20T00:00:00.000001Z'),
      summary('00000000-0000-0000-0000-000000000003', '2026-08-20T00:00:00.000001Z'),
    ]),
  }));

  await store.loadList();

  assert.deepEqual(store.conversations.map((item) => item.conversationId), [
    '00000000-0000-0000-0000-000000000003',
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
  ]);
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
  assert.equal(store.history?.conversation.conversationId, 'second');
});

test('message history traverses 205 turns by prepend with coalescing and overlap dedup', async () => {
  const calls: Array<string | null> = [];
  const store = new ConversationStore(api({
    history: async (id, cursor) => {
      calls.push(cursor ?? null);
      const pages: Record<string, {start: number; end: number; next: string | null}> = {
        recent: {start: 166, end: 205, next: 'before-166'},
        'before-166': {start: 126, end: 166, next: 'before-126'},
        'before-126': {start: 86, end: 125, next: 'before-86'},
        'before-86': {start: 46, end: 85, next: 'before-46'},
        'before-46': {start: 1, end: 45, next: null},
      };
      const page = pages[cursor ?? 'recent']!;
      return {
        conversation: summary(id),
        turns: Array.from(
          {length: page.end - page.start + 1},
          (_, index) => turn(page.start + index),
        ),
        nextCursor: page.next,
      };
    },
  }));
  await store.open('one');

  const first = store.loadOlderMessages();
  const coalesced = store.loadOlderMessages();
  assert.equal(first, coalesced);
  await first;
  while (store.hasOlderMessages) await store.loadOlderMessages();

  assert.equal(store.history?.turns.length, 205);
  assert.deepEqual(store.history?.turns.map((item) => item.turnNumber),
    Array.from({length: 205}, (_, index) => index + 1));
  assert.deepEqual(calls, [null, 'before-166', 'before-126', 'before-86', 'before-46']);
});

test('a recent refresh replaces a disconnected loaded range and restores its older cursor', async () => {
  let request = 0;
  const store = new ConversationStore(api({
    history: async (id) => {
      request += 1;
      return {
        conversation: summary(id),
        turns: request === 1
          ? Array.from({length: 40}, (_, index) => turn(index + 1))
          : Array.from({length: 40}, (_, index) => turn(index + 51)),
        nextCursor: request === 1 ? null : 'before-51',
      };
    },
  }));
  await store.open('one');

  await store.refreshActive();

  assert.deepEqual(
    store.history?.turns.map((item) => item.turnNumber),
    Array.from({length: 40}, (_, index) => index + 51),
  );
  assert.equal(store.hasOlderMessages, true);
  assert.equal(store.history?.nextCursor, 'before-51');
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
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['created']);
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
  assert.equal(store.history?.conversation.conversationId, 'one');
  assert.equal(store.viewRevision, before);
});

test('rename updates and reorders an already loaded summary', async () => {
  const store = new ConversationStore(api({
    list: async () => page([
      summary('one', '2026-08-19T00:00:00Z'),
      summary('two', '2026-08-20T00:00:00Z'),
    ]),
    rename: async (id, title) => ({
      ...summary(id, '2026-08-21T00:00:00Z'),
      title,
    }),
  }));
  await store.loadList();

  assert.equal(await store.rename('one', 'renamed'), 'ok');
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['one', 'two']);
  assert.equal(store.conversations[0]?.title, 'renamed');
});

test('rename validation errors do not masquerade as missing conversations', async () => {
  const store = new ConversationStore(api({
    list: async () => page([summary('one')]),
    rename: async () => { throw new ConversationApiError(422, 'invalid title'); },
  }));
  await store.loadList();

  assert.equal(await store.rename('one', ''), 'error');
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['one']);
});

test('an aborted mutation settles pending state without applying local changes', async () => {
  let receivedSignal: AbortSignal | undefined;
  const store = new ConversationStore(api({
    list: async () => page([summary('one')]),
    deleteAll: async (signal) => {
      receivedSignal = signal;
      await new Promise<void>((_resolve, reject) => {
        signal?.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    },
  }));
  await store.loadList();
  const controller = new AbortController();

  const deletion = store.deleteAll(controller.signal);
  assert.equal(store.mutationPending, true);
  controller.abort();

  assert.equal(await deletion, 'error');
  assert.equal(receivedSignal, controller.signal);
  assert.equal(store.mutationPending, false);
  assert.deepEqual(store.conversations.map((item) => item.conversationId), ['one']);
});

test('delete mutates the server before removing the local summary', async () => {
  const calls: string[] = [];
  const store = new ConversationStore(api({
    list: async () => page([summary('one')]),
    delete: async (id) => { calls.push(id); },
  }));
  await store.loadList();

  assert.equal(await store.delete('one'), 'ok');
  assert.deepEqual(calls, ['one']);
  assert.deepEqual(store.conversations, []);
  assert.equal(store.mutationPending, false);
});
