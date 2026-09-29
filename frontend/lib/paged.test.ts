// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {KeysetPager} from './paged.ts';

interface Page {
  items: string[];
  nextCursor: string | null;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return {promise, resolve, reject};
}

/** A pager whose every request waits for the test to answer it. */
function harness() {
  const requests: Array<{
    cursor: string | null;
    signal: AbortSignal;
    answer: ReturnType<typeof deferred<Page>>;
  }> = [];
  let notified = 0;
  const pager = new KeysetPager<Page>((cursor, signal) => {
    const answer = deferred<Page>();
    requests.push({cursor, signal, answer});
    return answer.promise;
  }, () => { notified += 1; });
  return {pager, requests, notified: () => notified};
}

test('start loads the first page and anchors the cursor for the next one', async () => {
  const {pager, requests, notified} = harness();
  const pages: string[][] = [];

  const first = pager.start((page) => { pages.push(page.items); });
  assert.equal(pager.state, 'loading');
  assert.equal(pager.hasOlder, false);
  requests[0]!.answer.resolve({items: ['c', 'b'], nextCursor: 'after-b'});
  await first;
  assert.deepEqual(requests.map(({cursor}) => cursor), [null]);
  assert.equal(pager.state, 'idle');
  assert.equal(pager.hasOlder, true);
  assert.equal(pager.outcome, null, 'a first page is not a next-page outcome');

  const next = pager.loadNext((page) => { pages.push(page.items); });
  assert.equal(pager.loadNext(() => { pages.push(['duplicate']); }), next, 'one flight at a time');
  requests[1]!.answer.resolve({items: ['a'], nextCursor: null});
  await next;

  assert.deepEqual(requests.map(({cursor}) => cursor), [null, 'after-b']);
  assert.deepEqual(pages, [['c', 'b'], ['a']]);
  assert.equal(pager.hasOlder, false);
  assert.equal(pager.outcome, 'loaded');
  assert.equal(notified(), 4, 'each load reports its start and its end');
  await pager.loadNext(() => { pages.push(['past the end']); });
  assert.equal(requests.length, 2);
});

test('a newer start aborts and drops the page it replaces', async () => {
  const {pager, requests} = harness();
  const pages: string[][] = [];
  pager.reset('older');
  const older = pager.loadNext((page) => { pages.push(page.items); });

  const restarted = pager.start((page) => { pages.push(page.items); });
  assert.equal(requests[0]!.signal.aborted, true);
  requests[0]!.answer.resolve({items: ['stale'], nextCursor: 'stale-cursor'});
  requests[1]!.answer.resolve({items: ['fresh'], nextCursor: null});
  await Promise.all([older, restarted]);

  assert.deepEqual(pages, [['fresh']]);
  assert.equal(pager.hasOlder, false);
  assert.equal(pager.outcome, null);
});

test('a failed next page keeps its cursor for a retry and reports the error', async () => {
  const {pager, requests} = harness();
  const errors: unknown[] = [];
  const pages: string[][] = [];
  pager.reset('older');

  const failing = pager.loadNext(() => {}, (error) => { errors.push(error); });
  const failure = new Error('unavailable');
  requests[0]!.answer.reject(failure);
  await failing;
  assert.equal(pager.state, 'error');
  assert.equal(pager.outcome, 'failed');
  assert.equal(pager.hasOlder, true);
  assert.deepEqual(errors, [failure]);

  const retry = pager.loadNext((page) => { pages.push(page.items); });
  assert.equal(pager.outcome, null, 'a retry in flight is no longer a failure');
  assert.equal(requests[1]!.cursor, 'older');
  requests[1]!.answer.resolve({items: ['old'], nextCursor: null});
  await retry;
  assert.deepEqual(pages, [['old']]);
  assert.equal(pager.state, 'idle');
});

test('a failed first page reports its error without a next-page outcome', async () => {
  const {pager, requests} = harness();
  let failed = false;
  const starting = pager.start(() => {}, () => { failed = true; });
  requests[0]!.answer.reject(new Error('down'));
  await starting;

  assert.equal(failed, true);
  assert.equal(pager.state, 'error');
  assert.equal(pager.outcome, null);
  assert.equal(pager.hasOlder, false);
});

test('cancel reports the flight it drops and keeps the cursor; reset re-anchors it silently', async () => {
  const {pager, requests, notified} = harness();
  const pages: string[][] = [];
  pager.reset('older');

  const cancelled = pager.loadNext((page) => { pages.push(page.items); });
  const before = notified();
  pager.cancel();
  assert.equal(notified(), before + 1, 'the list learns its next page is no longer loading');
  assert.equal(requests[0]!.signal.aborted, true);
  assert.equal(pager.state, 'idle');
  requests[0]!.answer.reject(new DOMException('Aborted', 'AbortError'));
  await cancelled;
  assert.equal(pager.hasOlder, true);
  assert.equal(pager.state, 'idle');
  assert.equal(notified(), before + 1, 'the dropped flight reports nothing more');
  pager.cancel();
  assert.equal(notified(), before + 1, 'nothing to drop, nothing to report');

  const replaced = pager.loadNext(() => { pages.push(['replaced']); });
  const beforeReset = notified();
  pager.reset(null);
  assert.equal(notified(), beforeReset);
  assert.equal(requests[1]!.signal.aborted, true);
  assert.equal(pager.hasOlder, false);
  requests[1]!.answer.reject(new DOMException('Aborted', 'AbortError'));
  await replaced;
  pager.reset('elsewhere');
  const next = pager.loadNext((page) => { pages.push(page.items); });
  assert.equal(requests[2]!.cursor, 'elsewhere');
  requests[2]!.answer.resolve({items: ['kept'], nextCursor: null});
  await next;
  assert.deepEqual(pages, [['kept']]);
});

test('a start leaves the list it replaces whole until its first page lands', async () => {
  const {pager, requests} = harness();
  const pages: string[][] = [];
  const loaded = pager.start((page) => { pages.push(page.items); });
  requests[0]!.answer.resolve({items: ['b'], nextCursor: 'after-b'});
  await loaded;

  const refreshing = pager.start((page) => { pages.push(page.items); });
  assert.equal(pager.state, 'loading');
  assert.equal(pager.starting, true);
  assert.equal(pager.hasOlder, true, 'the shown list still has its next page');
  assert.deepEqual(pager.snapshot(), {state: 'loading', starting: true, hasOlder: true, outcome: null});
  assert.equal(
    pager.loadNext(() => { pages.push(['from a replaced cursor']); }),
    refreshing,
    'a next page waits for the first page instead of reading past the old list',
  );
  assert.equal(requests.length, 2);

  requests[1]!.answer.resolve({items: ['c', 'b'], nextCursor: null});
  await refreshing;
  assert.deepEqual(pages, [['b'], ['c', 'b']]);
  assert.equal(pager.starting, false);
  assert.equal(pager.hasOlder, false, 'the landed first page decides');
});

test('a failed start keeps the cursor of the list it meant to replace', async () => {
  const {pager, requests} = harness();
  const loaded = pager.start(() => {});
  requests[0]!.answer.resolve({items: ['b'], nextCursor: 'after-b'});
  await loaded;

  const errors: unknown[] = [];
  const refreshing = pager.start(() => {}, (error) => { errors.push(error); });
  requests[1]!.answer.reject(new Error('down'));
  await refreshing;

  assert.equal(errors.length, 1);
  assert.equal(pager.state, 'error');
  assert.equal(pager.starting, false);
  assert.equal(pager.outcome, null, 'a first page is not a next-page outcome');
  assert.equal(pager.hasOlder, true);
  const next = pager.loadNext(() => {});
  assert.equal(requests[2]!.cursor, 'after-b');
  requests[2]!.answer.resolve({items: ['a'], nextCursor: null});
  await next;
});

test('cancel ends a start and lets the kept cursor load again', async () => {
  const {pager, requests, notified} = harness();
  pager.reset('older');
  const refreshing = pager.start(() => {});
  const before = notified();

  pager.cancel();
  assert.equal(notified(), before + 1);
  assert.equal(pager.starting, false);
  assert.equal(pager.state, 'idle');
  requests[0]!.answer.reject(new DOMException('Aborted', 'AbortError'));
  await refreshing;
  assert.equal(pager.hasOlder, true);
  void pager.loadNext(() => {});
  assert.equal(requests[1]!.cursor, 'older');
});
